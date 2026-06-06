"""Type stubs for block_sparse_attn_cuda (compiled CUDA extension)."""

from typing import Optional
import torch

def fwd_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_mask_type: torch.Tensor,
    streaming_info: Optional[torch.Tensor],
    row_blockmask: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    p_dropout: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    m_block_dim: int,
    n_block_dim: int,
    exact_streaming: bool,
    return_softmax: bool,
    gen: None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
