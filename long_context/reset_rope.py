import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.transformer.enums import AttnMaskType
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.custom_layers.transformer_engine import TEDotProductAttention

tokens = torch.randint(2, 9999, (1, 8192)) # 相当于一个长度为 8192 的 sample
print(tokens)
print(len(tokens[0]))
# text = tokenizer.decode(tokens[0])
# print(text)
b, s = tokens.size() # 1，8192
print(b, s)
tokens_flatten = tokens.reshape(-1)  # 展平成一维
print(tokens_flatten.size())
print(len(tokens_flatten))
eos_mask = tokens_flatten == 4 # 遇到4设为true，表示两个样本的分割
print(eos_mask)
for i in eos_mask:
    if i:
        print(i)
eos_mask[s - 1:: s] = True # 等价于eos_mask[s - 1] = True
eos_pos = eos_mask.nonzero(as_tuple=False).reshape(-1)
eos_pos = eos_pos.to(dtype=torch.int32)
zero = torch.tensor([0], device=eos_pos.device, dtype=eos_pos.dtype)
cu_seqlens = torch.cat([zero, eos_pos + 1], dim=0)
print(cu_seqlens)  # 一个token序列中样本分割的位置。

print("ab")
seq_length = 8192
kv_channels = 6
rotary_pos_emb = RotaryEmbedding(
    kv_channels=kv_channels,
    rotary_percent=1.0,
    rotary_interleaved=False,
    seq_len_interpolation_factor=None,
    rotary_base=10000)(seq_length)
print(rotary_pos_emb)
print(rotary_pos_emb.size())

att_mask_batch = 1
attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length))).view(att_mask_batch, 1, seq_length, seq_length)
print(attention_mask)

# tensor([[[[1., 0., 0., 0., 0., 0., 0., 0.],
#           [1., 1., 0., 0., 0., 0., 0., 0.],
#           [1., 1., 1., 0., 0., 0., 0., 0.],
#           [1., 1., 1., 1., 0., 0., 0., 0.],
#           [1., 1., 1., 1., 1., 0., 0., 0.],
#           [1., 1., 1., 1., 1., 1., 0., 0.],
#           [1., 1., 1., 1., 1., 1., 1., 0.],
#           [1., 1., 1., 1., 1., 1., 1., 1.]]]])
print(cu_seqlens)
# tensor([   0,  986, 8192], dtype=torch.int32)
packed_seq_params = PackedSeqParams(cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens)
print(packed_seq_params)

# s：序列长度
# b：batch size
# h：num of heads
# d：每个头的维度

sq = sk = sv = seq_length
bq = bk = bv = 1
hq = 10
hk = hv = 2
dq = dk = dv = kv_channels # 通常是64
Q = torch.rand(sq, bq, hq, dq).to("cuda:0")
K = torch.rand(sk, bk, hk, dk).to("cuda:0")
V = torch.rand(sv, bv, hv, dv).to("cuda:0")

print("Q,K,V shape")
print(Q.shape)
print(K.shape)
print(V.shape)

# torch.Size([8192, 1, 10, 6])
# torch.Size([8192, 1, 2, 6])
# torch.Size([8192, 1, 2, 6])

# print(Q)
# print(K)
# print(V)


def rope_normal(query, key, value, rotary_pos_emb, core_attention,config):
    query = apply_rotary_pos_emb(query, rotary_pos_emb, config)
    key = apply_rotary_pos_emb(key, rotary_pos_emb, config)
    core_attn_out = core_attention(query, key, value, attention_mask, AttnMaskType.causal, packed_seq_params)
    return core_attn_out

def rope_reset(query, key, value, rotary_pos_emb, core_attention,config, cu_seqlens):
    sq, bq, hq, dq = query.shape
    sk, bk, hk, dk = key.shape
    query = query.reshape(sq * bq, hq, dq)
    key = key.reshape(sk * bk, hk, dk)
    query = apply_rotary_pos_emb(query, rotary_pos_emb, config, cu_seqlens=cu_seqlens)
    key = apply_rotary_pos_emb(key, rotary_pos_emb, config, cu_seqlens=cu_seqlens)
    query = query.reshape(sq, bq, hq, dq)
    key = key.reshape(sk, bk, hk, dk)
    core_attn_out = core_attention(query, key, value, attention_mask, AttnMaskType.causal, packed_seq_params)
    return core_attn_out

from TransformerConfig import config as tfconfig
core_attention = TEDotProductAttention(tfconfig, 0, AttnMaskType.causal, "self")

a = rope_reset(Q, K, V, rotary_pos_emb, core_attention, tfconfig, cu_seqlens)
b = rope_normal(Q, K, V, rotary_pos_emb, core_attention, tfconfig)

c = a - b
print(c.size()) # torch.Size([8192, 1, 60])
print(c)