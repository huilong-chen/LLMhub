from transformers import AutoTokenizer
from functools import partial
from torch import Tensor

from megatron.training import global_vars
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.custom_layers.transformer_engine import te_checkpoint
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.transformer_block import TransformerBlock

def patch_all(finetune=False, reset_attention_mask=False):
    global_vars.build_tokenizer = build_tokenizer
    if reset_attention_mask:
        if finetune:
            Attention.forward = forward_sft
        else:
            Attention.forward = forward
        TransformerBlock._checkpointed_forward = _checkpointed_forward

def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print("> building Huggingface tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token_id = 0
    if args.vocab_size is None:
        args.padded_vocab_size = _vocab_size_with_padding(len(tokenizer), args)
    else:
        args.padded_vocab_size = args.vocab_size
    return tokenizer

def forward_sft(
    self,
    hidden_states,
    attention_mask,
    key_value_states=None,
    inference_params=None,
    rotary_pos_emb=None,
    packed_seq_params=None
):
    # hidden_states: [sq, b, h]

    # For self attention we just duplicate the rotary_pos_emb if it isn't already
    if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
        rotary_pos_emb = (rotary_pos_emb,) * 2

    # =====================
    # Query, Key, and Value
    # =====================
    # Get the query, key and value tensors based on the type of attention -
    # self or cross attn.
    query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

    # ===================================================
    # Adjust key, value, and rotary_pos_emb for inference
    # ===================================================
    key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
        inference_params, key, value, rotary_pos_emb
    )

    sq, bq, hq, dq = query.shape
    sk, bk, hk, dk = key.shape

    if packed_seq_params is not None:
        query = query.reshape(sq * bq, hq, dq)
        key = key.reshape(sk * bk, hk, dk)

    # ================================================
    # relative positional embedding (rotary embedding)
    # ================================================
    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        query = apply_rotary_pos_emb(
            query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
        )
        key = apply_rotary_pos_emb(
            key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
        )

        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

    if packed_seq_params is not None:
        query = query.reshape(sq, bq, hq, dq)
        key = key.reshape(sk, bk, hk, dk)
    # ==================================
    # core attention computation
    # ==================================

    if self.checkpoint_core_attention and self.training:
        core_attn_out = self._checkpointed_attention_forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
        )
    else:
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
        )

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.linear_proj(core_attn_out)

    return output, bias


def forward(
    self,
    hidden_states,
    attention_mask,
    key_value_states=None,
    inference_params=None,
    rotary_pos_emb=None,
    packed_seq_params=None
):
    # hidden_states: [sq, b, h]

    # For self attention we just duplicate the rotary_pos_emb if it isn't already
    if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
        rotary_pos_emb = (rotary_pos_emb,) * 2

    # =====================
    # Query, Key, and Value
    # =====================
    # Get the query, key and value tensors based on the type of attention -
    # self or cross attn.
    query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

    # ===================================================
    # Adjust key, value, and rotary_pos_emb for inference
    # ===================================================
    key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
        inference_params, key, value, rotary_pos_emb
    )

    # ================================================
    # relative positional embedding (rotary embedding)
    # ================================================
    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb

        query = apply_rotary_pos_emb(
            query, q_pos_emb, config=self.config, cu_seqlens=None,
        )
        key = apply_rotary_pos_emb(
            key, k_pos_emb, config=self.config, cu_seqlens=None,
        )

        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

    # ==================================
    # core attention computation
    # ==================================

    if self.checkpoint_core_attention and self.training:
        core_attn_out = self._checkpointed_attention_forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
        )
    else:
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
        )

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.linear_proj(core_attn_out)

    return output, bias


def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams,
):
    """Forward method with activation checkpointing."""

    def custom(start: int, end: int):
        def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
        ):
            for index in range(start, end):
                layer = self._get_layer(index)
                hidden_states, context = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    inference_params=None,
                    packed_seq_params=packed_seq_params,
                )
            return hidden_states, context

        return custom_forward

    def checkpoint_handler(forward_func):
        if self.config.fp8:
            return te_checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                parallel_state.get_tensor_model_parallel_group(),
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
            )
        else:
            return tensor_parallel.checkpoint(
                partial(forward_func, packed_seq_params=packed_seq_params),
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb
            )

    if self.config.recompute_method == 'uniform':
        # Uniformly divide the total number of Transformer layers and checkpoint
        # the input activation of each divided chunk.
        # A method to further reduce memory usage reducing checkpoints.
        l = 0
        while l < self.num_layers_per_pipeline_rank:
            hidden_states, context = checkpoint_handler(
                custom(l, l + self.config.recompute_num_layers)
            )

            l += self.config.recompute_num_layers

    elif self.config.recompute_method == 'block':
        # Checkpoint the input activation of only a set number of individual
        # Transformer layers and skip the rest.
        # A method fully use the device memory removing redundant re-computation.
        recompute_skip_num_layers = 0
        for l in range(self.num_layers_per_pipeline_rank):
            # Skip recomputation when input grad computation is not needed.
            # Need to have at least one input tensor with gradient computation
            # for re-enterant autograd engine.
            if self.config.fp8 and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            if (
                    l >= recompute_skip_num_layers
                    and l < self.config.recompute_num_layers + recompute_skip_num_layers
            ):
                hidden_states, context = checkpoint_handler(custom(l, l + 1))
            else:
                hidden_states, context = custom(l, l + 1)(
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    packed_seq_params,
                )
    else:
        raise ValueError("Invalid activation recompute method.")
    return hidden_states