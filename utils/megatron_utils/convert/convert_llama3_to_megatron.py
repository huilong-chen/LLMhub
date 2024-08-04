# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import re
import sys
import types
from functools import partial

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, LlamaConfig, is_safetensors_available
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

try:
    from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
except ModuleNotFoundError:
    print("Unable to import Megatron, please specify the path to Megatron using './Megatron-LM'. Exiting.")
    exit(1)


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
            list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}


def convert_tokens_to_string(tokens):
    """Converts a sequence of tokens (string) in a single string."""
    text = "".join(tokens)
    text = bytearray([byte_decoder[c] for c in text]).decode("utf-8", errors="replace")
    return text

# Copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/megatron_gpt2/checkpoint_reshaping_and_interoperability.py
def add_checkpointing_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer",
    )
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument("--norm-head", action="store_true")
    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    return parser


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="3GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser

#  1 LlamaForCausalLM(
#  2  (model): LlamaModel(
#  3     (embed_tokens): Embedding(128256, 4096)
#  4     (layers): ModuleList(
#  5       (0-31): 32 x LlamaDecoderLayer(
#  6         (self_attn): LlamaSdpaAttention(
#  7           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#  8           (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
#  9           (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
# 10           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
# 11           (rotary_emb): LlamaRotaryEmbedding()
# 12         )
# 13         (mlp): LlamaMLP(
# 14           (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
# 15           (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
# 16           (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
# 17           (act_fn): SiLU()
# 18         )
# 19         (input_layernorm): LlamaRMSNorm()
# 20         (post_attention_layernorm): LlamaRMSNorm()
# 21       )
# 22     )
# 23     (norm): LlamaRMSNorm()
# 24   )
# 25   (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
# 26 )

# The simple map of names for "automated" rules.
megatron_to_transformers = {
    "self_attention.query_key_value": ".self_attn.qkv_proj.",  # 7, 8, 9
    "self_attention.dense": ".self_attn.o_proj.",  # 10
    "mlp.dense_h_to_4h_1": ".mlp.gate_proj.",  # 14
    "mlp.dense_h_to_4h_2": ".mlp.up_proj.",  # 15
    "mlp.dense_4h_to_h": ".mlp.down_proj.",  # 16
    "input_layernorm": ".input_layernorm.",  # 19
    "post_attention_layernorm": ".post_attention_layernorm.",  # 20
}

transformers_to_megatron = {v[1:-1]: k for k, v in megatron_to_transformers.items()}

tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.bias",
    "self_attention.query_key_value.weight",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h_1.weight",
    "mlp.dense_h_to_4h_2.weight",
    "mlp.dense_4h_to_h.weight",
    # transformers layers to split across tp ranks
    "self_attn.qkv_proj.bias",
    "self_attn.qkv_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
]


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def merge_transformers_sharded_states(path):
    # small model without index
    if os.path.isfile(os.path.join(path, WEIGHTS_NAME)):
        return torch.load(os.path.join(path, WEIGHTS_NAME), "cpu")
    # Load the index
    index_file = os.path.join(path, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(path, SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {path}.")

    if safe_index_present and is_safetensors_available():
        from safetensors.torch import load_file as loader

        load_index = safe_index_file
    else:
        loader = partial(torch.load, map_location="cpu")
        load_index = index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    state_dict = {}
    for shard_file in shard_files:
        current_chunk = loader(os.path.join(path, shard_file))
        state_dict.update(current_chunk)
    return state_dict


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = "model_optim_rng.pt"
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def get_element_from_dict_by_path(d, path: str):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def load_and_check_transformers_model(path):
    """
    Load and check transformers models. Accept transformer model path only.

    """
    try:
        model = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        print("Error with this model conversion: ", path)
        raise e


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-LM checkpoint arguments from the state dict
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = "model_optim_rng.pt"
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # Create Transformers Llama config from Megatron-LM arguments
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_config = get_tokenizer_config(args.tokenizer_path)

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=megatron_args.hidden_size,
        intermediate_size=megatron_args.ffn_hidden_size,
        num_hidden_layers=megatron_args.num_layers,
        num_attention_heads=megatron_args.num_attention_heads,
        max_position_embeddings=megatron_args.max_position_embeddings,
        rope_theta=500000,
        rms_norm_eps=1e-5,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        architectures=["LlamaForCausalLM"],
    )
    if hasattr(megatron_args, "norm_epsilon"):
        config.rms_norm_eps = megatron_args.norm_epsilon
    if hasattr(megatron_args, "rope_base"):
        config.rope_theta = megatron_args.rope_base

    # 自定义的tokenizer，需要把auto_map添加至config.json，否则加载tokenizer会报错。
    if "auto_map" in tokenizer_config:
        config.auto_map = tokenizer_config["auto_map"]

    if hasattr(megatron_args, "group_query_attention") and megatron_args.group_query_attention:
        num_query_groups = megatron_args.num_query_groups
        config.num_key_value_heads = num_query_groups
    else:
        num_query_groups = megatron_args.num_attention_heads

    output_state_dict = {}

    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    dtype = torch.bfloat16
    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z_]+)")

    # Convert.
    print("Converting")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    # Convert and store the word embeddings.
    word_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    # cut the word_embeddings to the correct size
    word_embeddings = word_embeddings[: config.vocab_size, :]
    output_state_dict["model.embed_tokens.weight"] = word_embeddings.to(dtype)

    # Transformer Layers
    print("Converting transformer layers")
    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    query_head_per_group = heads // num_query_groups
    num_layers = config.num_hidden_layers // pp_size

    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_rank)

        # The transformer.
        path = "model.language_model.encoder"
        # Extract the layers.
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            # Match the name.
            m = layer_re.match(key)
            # Stop if that's not a layer
            if m is None:
                break
            if key.endswith("_extra_state"):
                continue
            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            # The name of the layer.
            layer_name = f"model.layers.{layer_idx}"

            if op_name + "." + weight_or_bias not in tensor_parallel_params:
                params = val.to(dtype)
            else:
                dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h"] else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            # Transpose the QKV matrix.
            if op_name == "self_attention.query_key_value":
                params = [val] + [
                    get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                    for tp_rank in range(1, tp_size)
                ]
                params = [
                    p.view(
                        num_query_groups // tp_size,
                        (query_head_per_group + 2),
                        hidden_size_per_head,
                        -1,
                    )
                    for p in params
                ]
                params = torch.cat(params, dim=0).to(dtype)
                q, k, v = torch.split(params, [query_head_per_group, 1, 1], dim=1)

                if weight_or_bias == "weight":
                    q, k, v = (
                        q.reshape(-1, config.hidden_size),
                        k.reshape(-1, config.hidden_size),
                        v.reshape(-1, config.hidden_size),
                    )
                else:
                    # For bias convert
                    # q: query_head_per_group; k: 1; v: 1
                    q, k, v = q.reshape(-1), k.reshape(-1), v.reshape(-1)

                output_state_dict[f"{layer_name}.self_attn.q_proj.{weight_or_bias}"] = q.clone()
                output_state_dict[f"{layer_name}.self_attn.k_proj.{weight_or_bias}"] = k.clone()
                output_state_dict[f"{layer_name}.self_attn.v_proj.{weight_or_bias}"] = v.clone()
            else:
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + weight_or_bias] = params

    if config.num_hidden_layers != (layer_idx + 1):
        raise ValueError(f"Expected {config.num_hidden_layers} layers but found {layer_idx + 1}")

    # The final layernorm.
    print("Converting final layernorm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    output_state_dict["model.norm.weight"] = params["final_layernorm.weight"].to(dtype)

    print("Converting LM head")
    lm_head = torch.cat(
        [
            get_element_from_dict_by_path(tp_state_dicts[tp_rank], "model.language_model.output_layer.weight")
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )

    # cut the head to the size of the vocab
    lm_head = lm_head[: config.vocab_size, :]

    if args.norm_head:
        print("normalizing lm head!!")
        lm_head = torch.nn.functional.normalize(lm_head)
    output_state_dict["lm_head.weight"] = lm_head.to(dtype)

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Store the config to file.
    print("Saving config")
    config.save_pretrained(args.save_path)

    # Save tokenizer based on args
    tokenizer.save_pretrained(args.save_path)

    # Store the state_dict to file.
    # 保存前 clone 所有的 tensor，避免 save 文件过大
    output_state_dict = {k: v.clone() for k, v in output_state_dict.items()}
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # Save the model
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )

    # check that config and weight are consistent.
    load_and_check_transformers_model(args.save_path)


def convert_checkpoint_from_transformers_to_megatron(args):
    """
    Convert a checkpoint from HuggingFace Transformers to Megatron-LM. This allows converted checkpoints with variable
    tensor parallelism and pipeline parallelism sizes. It takes as input a checkpoint from HuggingFace Transformers
    which can have multiple shards.

    Args:
        args (argparse.Namespace): the arguments to the script

    """
    os.makedirs(args.save_path, exist_ok=True)
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

    # load the transformers model state dict and config
    state_dict = merge_transformers_sharded_states(args.load_path)

    config = AutoConfig.from_pretrained(args.load_path)

    # Saving the tracker file
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # megatron args
    megatron_args = {"orig_vocab_size": config.vocab_size, "max_position_embeddings": config.max_position_embeddings,
                     "hidden_size": config.hidden_size, "num_layers": config.num_hidden_layers,
                     "num_attention_heads": config.num_attention_heads, "ffn_hidden_size": config.intermediate_size,
                     "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
                     "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
                     "data_parallel_size": args.target_data_parallel_size,
                     "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by, "rank": 0,
                     "tokenizer_type": "PreTrainedTokenizerFast",
                     "group_query_attention": False, "rope_base": config.rope_theta}
    if hasattr(config, "num_key_value_heads") and config.num_key_value_heads:
        num_query_groups = config.num_key_value_heads
        if config.num_key_value_heads % args.target_tensor_model_parallel_size != 0:
            raise ValueError(
                f"Number of key value heads ({config.num_key_value_heads}) is not divisible by "
                f"target_tensor_model_parallel_size ({args.target_tensor_model_parallel_size})"
            )
        if config.num_key_value_heads:
            megatron_args["group_query_attention"] = True
    else:
        num_query_groups = config.num_attention_heads
    megatron_args["num_query_groups"] = num_query_groups

    # Llama use silu as activation function
    megatron_args["bias_gelu_fusion"] = False
    megatron_args["openai_gelu"] = False

    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    setattr(margs, "params_dtype", dtype)

    # save dummy optim state dict
    dummy_optim_state_dict = {}
    dummy_optim_state_dict["optimizer"] = {
        "step": 0,
        "param_groups": [
            {
                "lr": 0.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "eps": 0.0,
                "weight_decay": 0.0,
                "correct_bias": False,
                "params": [],
            }
        ],
    }
    if args.use_distributed_optimizer:
        for i in range(args.target_pipeline_model_parallel_size):
            for j in range(args.target_tensor_model_parallel_size):
                for k in range(args.target_data_parallel_size):
                    if args.target_pipeline_model_parallel_size == 1:
                        checkpoint_dir = f"mp_rank_{j:02d}_{i:03d}"
                    else:
                        checkpoint_dir = f"mp_rank_{j:02d}_{i:03d}_{k:03d}"
                    checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        dummy_optim_state_dict,
                        os.path.join(checkpoint_dir, "optim.pt"),
                    )

    # Convert.
    print("Converting")
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})

    # Embedding layer
    print("converting embedding layer")
    word_embedding = state_dict["model.embed_tokens.weight"].to(dtype)
    lm_head = state_dict["lm_head.weight"].to(dtype)
    orig_vocab_size = config.vocab_size

    # Load tokenizer and get real original vocab_size and padded it.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    padded_vocab_size = _vocab_size_with_padding(len(tokenizer), margs)

    setattr(margs, "padded_vocab_size", padded_vocab_size)
    # Cut out extra padding we don't need
    if orig_vocab_size > padded_vocab_size:
        full_word_embed = word_embedding[0:padded_vocab_size, :].clone()
        full_lm_head = lm_head[0:padded_vocab_size, :].clone()
    # Expanding embedding to larger size by replicating final entry
    elif orig_vocab_size < padded_vocab_size:
        input_embeddings_avg = word_embedding[:orig_vocab_size].mean(dim=0, keepdim=True)
        output_embeddings_avg = lm_head[:orig_vocab_size].mean(dim=0, keepdim=True)
        padding_size = padded_vocab_size - orig_vocab_size
        full_word_embed = torch.cat((word_embedding, input_embeddings_avg.expand(padding_size, -1)))
        full_lm_head = torch.cat((lm_head, output_embeddings_avg.expand(padding_size, -1)))
    # Same size!
    else:
        full_word_embed = word_embedding.clone()
        full_lm_head = lm_head.clone()

    origin_tokenizer = AutoTokenizer.from_pretrained(args.load_path, use_fast=False, trust_remote_code=True)
    # 如果tokenizer不一致，说明需要从头开始初始化token
    if len(origin_tokenizer) != len(tokenizer):
        word_embedding = word_embedding.cuda()
        lm_head = lm_head.cuda()
        for token_idx in range(0, len(tokenizer)):
            target_token = tokenizer.convert_ids_to_tokens(token_idx)
            ori_token_idx = origin_tokenizer.convert_tokens_to_ids(target_token)
            if ori_token_idx is not None and ori_token_idx == token_idx:
                continue
            elif ori_token_idx is not None and ori_token_idx != token_idx:
                full_word_embed[token_idx] = word_embedding[ori_token_idx].cpu()
                full_lm_head[token_idx] = lm_head[ori_token_idx].cpu()
                print("move token {} origin token_id {} to new token_id {}".format(target_token, ori_token_idx,
                                                                                   token_idx))
            else:
                real_target_token = convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token_idx))
                origin_token_ids = origin_tokenizer.encode(real_target_token)
                origin_tokens = [origin_tokenizer.decode(i) for i in origin_token_ids]
                print(
                    f"target: {token_idx:<5} {repr(real_target_token):<15} origin: {origin_token_ids} {origin_tokens}")
                input_embeddings_avg = word_embedding[origin_token_ids].mean(dim=0, keepdim=True).cpu()
                output_embeddings_avg = lm_head[origin_token_ids].mean(dim=0, keepdim=True).cpu()
                full_word_embed[token_idx] = input_embeddings_avg
                full_lm_head[token_idx] = output_embeddings_avg

    # Split into new tensor model parallel sizes
    # (39424, 5120) -> [(19712, 5120), (19712, 5120)]
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_model_parallel_size, dim=0)
    for i in range(args.target_tensor_model_parallel_size):
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embed[i].clone()

    # Transformer layers
    print("converting transformer layers")
    if config.num_attention_heads % args.target_tensor_model_parallel_size != 0:
        raise ValueError(
            f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of tensor parallelism"
            f" ({args.target_tensor_model_parallel_size})"
        )

    if config.num_hidden_layers % args.target_pipeline_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.num_hidden_layers}) must be divisible by number of pipeline parallelism"
            f" ({args.target_pipeline_model_parallel_size})"
        )
    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size

    layer_re = re.compile(r"model\.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z_]+)")
    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // heads
    query_head_per_group = heads // num_query_groups

    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})

        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [
                layer_name for layer_name in state_dict.keys() if layer_name.startswith(f"model.layers.{pp_layer_id}.")
            ]

            # combine q,k,v
            for weight_or_bias in ["weight", "bias"]:
                # If Bias in attention, and convert it
                if f"model.layers.{pp_layer_id}.self_attn.q_proj.{weight_or_bias}" not in state_dict:
                    continue
                q, k, v = (
                    state_dict[f"model.layers.{pp_layer_id}.self_attn.{op}_proj.{weight_or_bias}"] for op in "qkv"
                )
                for op in "qkv":
                    layers_to_copy.remove(f"model.layers.{pp_layer_id}.self_attn.{op}_proj.{weight_or_bias}")
                layer_name = f"model.layers.{pp_layer_id}.self_attn.qkv_proj.{weight_or_bias}"
                layers_to_copy.append(layer_name)
                if weight_or_bias == "weight":
                    # q: (num_heads * hidden_size_per_head, hidden_size)
                    # k: (num_kv_heads * hidden_size_per_head, hidden_size)
                    # v: (num_kv_heads * hidden_size_per_head, hidden_size)
                    q_grouped = q.view(num_query_groups, query_head_per_group, hidden_size_per_head, -1)
                    k_grouped = k.view(num_query_groups, 1, hidden_size_per_head, -1)
                    v_grouped = v.view(num_query_groups, 1, hidden_size_per_head, -1)
                    # qkv: (num_query_groups * (query_head_per_group + 2) * hidden_size_per_head)
                    state_dict[layer_name] = torch.cat((q_grouped, k_grouped, v_grouped), dim=1).view(
                        -1, config.hidden_size
                    )
                else:
                    # If Bias in attention, and convert it
                    q_bias = q.view([num_query_groups, -1])
                    k_bias = k.view([num_query_groups, -1])
                    v_bias = v.view([num_query_groups, -1])
                    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).view(-1).contiguous()
                    state_dict[layer_name] = qkv_bias

            for layer_name in layers_to_copy:
                m = layer_re.match(layer_name)
                # Stop if that's not a layer
                if m is None:
                    break

                # The index of the layer.
                _ = int(m.group(1))
                # The name of the operation.
                op_name = m.group(2)
                # Is it a weight or a bias?
                weight_or_bias = m.group(3)

                params = state_dict[layer_name].to(dtype)
                out_name = transformers_to_megatron.get(op_name, None)
                if out_name is None:
                    continue

                layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

                if op_name + "." + weight_or_bias in tensor_parallel_params:
                    dim = 1 if op_name in ["self_attn.o_proj", "mlp.down_proj"] else 0
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)

                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = (
                        params[i].clone() if (op_name + "." + weight_or_bias in tensor_parallel_params) else params
                    )

        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            # handle final layernorm
            weight_or_bias = "weight"
            params = state_dict[f"model.norm.{weight_or_bias}"].to(dtype)
            layer_name = f"final_layernorm.{weight_or_bias}"
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                params_dict[layer_name] = params

            full_lm_head = torch.chunk(full_lm_head, args.target_tensor_model_parallel_size, dim=0)

            # add the LM head
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.output_layer")
                params_dict["weight"] = full_lm_head[i].clone()

        # saving the state dict as per the tp_rank and pp_rank
        for tp_rank in range(args.target_tensor_model_parallel_size):
            output_state_dict[tp_rank]["checkpoint_version"] = 3.0
            output_state_dict[tp_rank]["args"] = margs
            checkpoint_dir = (
                f"mp_rank_{tp_rank:02d}"
                if args.target_pipeline_model_parallel_size == 1
                else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
            )
            checkpoint_name = "model_optim_rng.pt"

            if not args.use_distributed_optimizer:
                output_state_dict[tp_rank]["optimizer"] = dummy_optim_state_dict["optimizer"]
            checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if args.print_checkpoint_structure:
                print(
                    f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank"
                    f" {pp_rank}:"
                )
                recursive_print(None, output_state_dict[tp_rank])
            torch.save(output_state_dict[tp_rank], checkpoint_path)


def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()
    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()
