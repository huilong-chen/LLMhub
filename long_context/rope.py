import torch
import logging

from torch import Tensor
from typing import Optional
from megatron.core.transformer.transformer_config import TransformerConfig
from rotary_pos_embedding import RotaryEmbedding

from TransformerConfig import config


from apex.transformer.functional import (
        fused_apply_rotary_pos_emb,
        fused_apply_rotary_pos_emb_thd,
    )

HAVE_APPLY_ROPE_FUSION = True

logger = logging.getLogger(__name__)

# Copy from Megatron-LM

def apply_rotary_pos_emb_thd(
    t: Tensor, cu_seqlens: Tensor, freqs: Tensor, rotary_interleaved: bool = False
) -> Tensor:

    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return torch.cat(
        [
            apply_rotary_pos_emb_bshd(x.unsqueeze(1), freqs[: x.size(0)])
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    # 把 x 从 [even odd] 变成 [-odd, +even]
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)

def apply_rotary_pos_emb_bshd(t: Tensor, freqs: Tensor, rotary_interleaved: bool = False) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    print(freqs.shape) # torch.Size([8, 1, 1, 4])
    rot_dim = freqs.shape[-1]
    print(rot_dim) # 4

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    # 理想情况下，t_pass为空，因此旋转位置嵌入应用于所有张量t。
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    # "第一部分是余弦分量。"
    # "第二部分是正弦分量，需要使用_rotate_half方法改变符号。"
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb(
    t: Tensor, freqs: Tensor, config: TransformerConfig, cu_seqlens: Optional[Tensor] = None,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    根据融合/未融合内核，或bshd（传统）/ thd（打包序列）格式，重新路由到适当的apply_rotary_pos_emb函数。
    """
    if config.apply_rope_fusion and not HAVE_APPLY_ROPE_FUSION:
        # setting apply_rope_fusion in config to False so that subsequent queries to this config also return False
        config.apply_rope_fusion = False
        if not getattr(apply_rotary_pos_emb, "printed_fused_warning", False):
            logger.warning(
                "Setting apply_rope_fusion to false because its implementation"
                " is not included in Apex. Try upgrading to the latest version"
            )
            apply_rotary_pos_emb.printed_fused_warning = True
    if config.apply_rope_fusion:
        if cu_seqlens is None:
            return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
        else:
            return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)
    else:
        if cu_seqlens is None:
            # no fuse, no cu
            return apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved=config.rotary_interleaved)
        else:
            return apply_rotary_pos_emb_thd(
                t, cu_seqlens, freqs, rotary_interleaved=config.rotary_interleaved
            )


def main():

    sq = 8
    bq = 2
    hq = 4
    dq = 4
    query = torch.rand(sq, bq, hq, dq).to("cuda:0")

    kv_channels = dq
    rotary_embedding = RotaryEmbedding(
        kv_channels=kv_channels,
        rotary_percent=1.0,
        rotary_interleaved=False,
        seq_len_interpolation_factor=None,
        rotary_base=10000)
    rotary_pos_emb = rotary_embedding(sq)

    print(f"Q before rope:\n {query}")
    query_rope = apply_rotary_pos_emb(query, rotary_pos_emb, config)
    print(f"Q after rope:\n {query_rope}")
    print(f"Diff of before and after: \n {torch.abs(query - query_rope)}")

if __name__ == "__main__":
    main()