from functools import partial

import datasets
import torch
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.training import get_args, get_timers, get_tokenizer, global_vars, initialize_megatron, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import average_losses_across_data_parallel_group, get_ltor_masks_and_position_ids

from utils.megatron_utils.patch import patch_all
from utils.megatron_utils.arguments import get_tasks_args
from utils.megatron_utils.megatron_finetune_utils import finetune


def get_gpt_layer_with_transformer_engine_spec() -> ModuleSpec:
    """
    Generates a spec for a GPT transformer layer using Transformer Engine modules.

    Returns:
        A ModuleSpec object that specifies how to construct a GPT transformer layer with
        the appropriate submodules for self-attention and MLP/MoE using Transformer Engine optimizations.
    """
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0("building GPT model ...")
    args = get_args()
    config = core_transformer_config_from_args(args)

    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    model = GPTModel(
        config=config,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rope_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )
    return model


def train_valid_datasets_provider(*args, **kwargs):
    """Build train and validation dataset."""
    args = global_vars.get_args()
    train_dataset = datasets.load_from_disk(args.train_data, keep_in_memory=False)
    return train_dataset, None


def get_batch(data):
    """Generate a batch"""
    args = global_vars.get_args()
    tokenizer = get_tokenizer()
    data = next(data)
    # Unpack.
    tokens = data["input_ids"].long()
    labels = data["labels"].long()
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = torch.cat([shift_labels, torch.full_like(shift_labels[:, -1:], -100)], dim=1)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eos_token_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    loss_mask[shift_labels == -100] = 0

    return tokens, shift_labels, loss_mask, attention_mask, position_ids


def loss_func(output_tensor, loss_mask):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask)
    if loss_mask.sum() != 0:
        loss = loss / loss_mask.sum()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask=loss_mask)


if __name__ == "__main__":
    patch_all(finetune=True, reset_attention_mask=True)
    initialize_megatron(extra_args_provider=get_tasks_args)

    model = model_provider()
    print(model)

    finetune(
        train_valid_datasets_provider,
        model_provider,
    )


# model = model_provider()

# GPTModel(
#   (embedding): LanguageModelEmbedding(
#     (word_embeddings): VocabParallelEmbedding()
#     (embedding_dropout): Dropout(p=0.0, inplace=False)
#   )
#   (rotary_pos_emb): RotaryEmbedding()
#   (decoder): TransformerBlock(
#     (layers): ModuleList(
#       (0-31): 32 x TransformerLayer(
#         (input_layernorm): IdentityOp()
#         (self_attention): SelfAttention(
#           (core_attention): TEDotProductAttention(
#             (flash_attention): FlashAttention()
#             (fused_attention): FusedAttention()
#             (unfused_attention): UnfusedDotProductAttention(
#               (scale_mask_softmax): FusedScaleMaskSoftmax()
#               (attention_dropout): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (linear_proj): TERowParallelLinear()
#           (linear_qkv): TELayerNormColumnParallelLinear()
#         )
#         (pre_cross_attn_layernorm): IdentityOp()
#         (cross_attention): IdentityOp()
#         (cross_attn_bda): IdentityFuncOp()
#         (pre_mlp_layernorm): IdentityOp()
#         (mlp): MLP(
#           (linear_fc1): TELayerNormColumnParallelLinear()
#           (linear_fc2): TERowParallelLinear()
#         )
#       )
#     )
#     (final_layernorm): RMSNorm()
#   )
#   (output_layer): ColumnParallelLinear()
# )