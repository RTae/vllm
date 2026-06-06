# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton kernel for paged sparse attention with native GQA support.

Eliminates the gather+expand bottleneck of existing sparse attention backends
(SpargeAttn, XAttention) which require:
  1. index_select: paged [N_pages, page_size, H_kv, D] → dense [N, H_kv, D]
  2. repeat_interleave: [N, H_kv, D] → [N, H_q, D]  (GQA expansion)
  = ~1.3 GB written per sequence across 28 layers at 8K tokens

This kernel reads paged KV directly in the page loop, with:
  - Native GQA:    kv_head = q_head // GQA_RATIO  (no expand)
  - Sparse pages:  check sparse_mask before any KV load
  - Online softmax: Flash-style (m, l, acc) accumulated across pages
  - Causal mask:   applied within each page for prefill

Memory layout (vLLM PagedAttention):
  k_cache: [num_pages, page_size, H_kv, head_dim]
  v_cache: [num_pages, page_size, H_kv, head_dim]
  block_table: [batch, max_pages_per_seq]
  q: [total_tokens, H_q, head_dim]  (varlen packed)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def paged_sparse_attn_kernel(
    # Pointers
    Q_ptr,
    K_cache_ptr,
    V_cache_ptr,
    Out_ptr,
    block_table_ptr,
    sparse_mask_ptr,
    seq_lens_ptr,
    cu_seqlens_ptr,
    # Q strides: [total_tokens, H_q, head_dim]
    stride_qm, stride_qh, stride_qd,
    # K/V cache strides: [num_pages, page_size, H_kv, head_dim]
    stride_kb, stride_kp, stride_kh, stride_kd,
    stride_vb, stride_vp, stride_vh, stride_vd,
    # Output strides: [total_tokens, H_q, head_dim]
    stride_om, stride_oh, stride_od,
    # block_table strides: [batch, max_pages_per_seq]
    stride_bt_b, stride_bt_p,
    # sparse_mask strides: [batch, max_pages_per_seq]
    stride_sm_b, stride_sm_p,
    # Compile-time constants
    H_q: tl.constexpr,
    H_kv: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,   # == PAGE_SIZE
    # Runtime scalars
    softmax_scale,
    max_pages_per_seq,
):
    """
    Grid: [batch_size, H_q, ceil(seq_len_q / BLOCK_M)]

    Each thread block handles BLOCK_M query tokens for one (batch, q_head).
    It iterates over logical KV pages, skipping pages where sparse_mask is False,
    accumulating (m, l, acc) using online softmax (Flash Attention 2 style).
    """
    # ── Grid indices ────────────────────────────────────────────────────────────
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2)

    # GQA: map q head → kv head
    kv_head_idx = q_head_idx // GQA_RATIO

    # ── Sequence info ────────────────────────────────────────────────────────────
    seq_len_q = tl.load(seq_lens_ptr + batch_idx)
    q_token_start = tl.load(cu_seqlens_ptr + batch_idx)

    # Query block token range [q_start, q_end)
    q_start = q_token_start + q_block_idx * BLOCK_M
    q_end = tl.minimum(q_start + BLOCK_M, q_token_start + seq_len_q)
    n_q_tokens = q_end - q_start  # actual tokens in this block (may be < BLOCK_M)

    # ── Load query block [BLOCK_M, HEAD_DIM] ────────────────────────────────────
    q_offsets_m = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    q_offsets_d = tl.arange(0, HEAD_DIM)  # [HEAD_DIM]

    # Absolute token indices for this block
    q_token_idx = q_start + q_offsets_m  # [BLOCK_M]
    q_mask_m = q_offsets_m < n_q_tokens  # [BLOCK_M]

    # Load Q: q_token_idx * stride_qm + q_head_idx * stride_qh + d * stride_qd
    Q_ptrs = (
        Q_ptr
        + q_token_idx[:, None] * stride_qm
        + q_head_idx * stride_qh
        + q_offsets_d[None, :] * stride_qd
    )  # [BLOCK_M, HEAD_DIM]
    q = tl.load(Q_ptrs, mask=q_mask_m[:, None], other=0.0)  # [BLOCK_M, HEAD_DIM]

    # ── Online softmax state ─────────────────────────────────────────────────────
    # m_i: running row-max [BLOCK_M]
    # l_i: running sum-of-exp [BLOCK_M]
    # acc:  running output accumulator [BLOCK_M, HEAD_DIM]
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Absolute position of first query token in this block (for causal mask)
    # q_pos[i] = q_start - q_token_start + i  (position within sequence)
    q_seq_pos = q_start - q_token_start  # start position of query block in sequence

    # ── Page loop ────────────────────────────────────────────────────────────────
    # Iterate over logical KV pages for this batch element.
    # KV length == seq_len_q for full prefill (q_len == kv_len).

    for page_idx in range(0, max_pages_per_seq):
        # Skip pages beyond the valid KV range
        is_valid_page = page_idx < ((seq_len_q + PAGE_SIZE - 1) // PAGE_SIZE)

        # ── Sparse gate: skip page if mask is False ──────────────────────────
        # Load mask BEFORE any KV access — the entire point of this kernel.
        # Also skip invalid (out-of-range) pages.
        do_attend_raw = tl.load(
            sparse_mask_ptr + batch_idx * stride_sm_b + page_idx * stride_sm_p
        )
        do_attend = is_valid_page & (do_attend_raw != 0)

        # ── Physical page index ──────────────────────────────────────────────
        # Load 0 when not attending (dummy index, loads will be masked)
        phys_page = tl.load(
            block_table_ptr + batch_idx * stride_bt_b + page_idx * stride_bt_p,
            mask=is_valid_page, other=0,
        )

        # KV token positions in sequence: [page_idx*PAGE_SIZE, (page_idx+1)*PAGE_SIZE)
        kv_page_start = page_idx * PAGE_SIZE
        kv_offsets_n = tl.arange(0, BLOCK_N)           # [BLOCK_N]
        kv_token_pos = kv_page_start + kv_offsets_n    # absolute pos in seq
        kv_mask_n = (kv_token_pos < seq_len_q) & do_attend  # also gate on do_attend

        # ── Load K page: [BLOCK_N, HEAD_DIM] ────────────────────────────────
        K_ptrs = (
            K_cache_ptr
            + phys_page * stride_kb
            + kv_offsets_n[:, None] * stride_kp
            + kv_head_idx * stride_kh
            + q_offsets_d[None, :] * stride_kd
        )  # [BLOCK_N, HEAD_DIM]
        k = tl.load(K_ptrs, mask=kv_mask_n[:, None], other=0.0)  # [BLOCK_N, HEAD_DIM]

        # ── Attention scores: QK^T  [BLOCK_M, BLOCK_N] ──────────────────────
        s = tl.dot(q, tl.trans(k)) * softmax_scale  # [BLOCK_M, BLOCK_N]

        # ── Causal mask ──────────────────────────────────────────────────────
        q_pos_m = q_seq_pos + q_offsets_m   # [BLOCK_M]
        causal_mask = q_pos_m[:, None] >= kv_token_pos[None, :]  # [BLOCK_M, BLOCK_N]

        combined_mask = causal_mask & kv_mask_n[None, :] & q_mask_m[:, None]
        s = tl.where(combined_mask, s, float("-inf"))

        # ── Online softmax update (Flash-style) ──────────────────────────────
        m_new = tl.maximum(m_i, tl.max(s, axis=1))  # [BLOCK_M]
        # If do_attend is False, s is all -inf, m_new == m_i, no change
        alpha = tl.exp(m_i - m_new)  # [BLOCK_M]
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        p = tl.exp(s - m_new[:, None])  # [BLOCK_M, BLOCK_N]
        p = tl.where(q_mask_m[:, None], p, 0.0)

        l_i = l_i + tl.sum(p, axis=1)  # [BLOCK_M]
        m_i = m_new

        # ── Load V page and accumulate ────────────────────────────────────────
        V_ptrs = (
            V_cache_ptr
            + phys_page * stride_vb
            + kv_offsets_n[:, None] * stride_vp
            + kv_head_idx * stride_vh
            + q_offsets_d[None, :] * stride_vd
        )  # [BLOCK_N, HEAD_DIM]
        v = tl.load(V_ptrs, mask=kv_mask_n[:, None], other=0.0)  # [BLOCK_N, HEAD_DIM]

        acc = acc + tl.dot(p.to(v.dtype), v).to(tl.float32)

    # ── Finalize: divide by normaliser ──────────────────────────────────────────
    # Guard against fully-masked rows (all KV pages skipped for this query pos).
    # When m_i == -inf and l_i == 0, output should be 0 (no attended tokens).
    is_masked = m_i == float("-inf")
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe[:, None]
    # Zero out rows that attended no tokens
    acc = tl.where(is_masked[:, None], tl.zeros_like(acc), acc)

    # ── Write output ────────────────────────────────────────────────────────────
    Out_ptrs = (
        Out_ptr
        + q_token_idx[:, None] * stride_om
        + q_head_idx * stride_oh
        + q_offsets_d[None, :] * stride_od
    )  # [BLOCK_M, HEAD_DIM]
    tl.store(Out_ptrs, acc.to(Q_ptr.dtype.element_ty), mask=q_mask_m[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# Python wrapper
# ─────────────────────────────────────────────────────────────────────────────

def paged_sparse_attention(
    q: torch.Tensor,           # [total_tokens, H_q, head_dim]
    k_cache: torch.Tensor,     # [num_pages, page_size, H_kv, head_dim]
    v_cache: torch.Tensor,     # [num_pages, page_size, H_kv, head_dim]
    block_table: torch.Tensor, # [batch, max_pages_per_seq]  int32
    sparse_mask: torch.Tensor, # [batch, max_pages_per_seq]  bool
    seq_lens: torch.Tensor,    # [batch]  int32
    softmax_scale: float | None = None,
    BLOCK_M: int = 64,
) -> torch.Tensor:
    """
    Paged sparse attention with native GQA — no gather, no repeat_interleave.

    Args:
        q:            [total_tokens, H_q, head_dim]
        k_cache:      [num_pages, page_size, H_kv, head_dim]
        v_cache:      [num_pages, page_size, H_kv, head_dim]
        block_table:  [batch, max_pages_per_seq] — logical→physical page map
        sparse_mask:  [batch, max_pages_per_seq] — True = attend this page
        seq_lens:     [batch] — actual KV length per sequence
        softmax_scale: 1/sqrt(head_dim) if None
        BLOCK_M:      query block size (must be power of 2)

    Returns:
        output: [total_tokens, H_q, head_dim]
    """
    total_tokens, H_q, head_dim = q.shape
    num_pages, page_size, H_kv, _ = k_cache.shape
    batch_size = seq_lens.shape[0]

    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    assert H_q % H_kv == 0, f"GQA: H_q={H_q} must be divisible by H_kv={H_kv}"
    GQA_RATIO = H_q // H_kv
    BLOCK_N = page_size  # one Triton block = one page

    # Build cumulative seq lens for varlen token indexing
    # cu_seqlens[i] = sum(seq_lens[:i])
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
    cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)
    # Use only the first batch_size entries as starting offsets
    cu_seqlens_start = cu_seqlens[:batch_size]

    max_pages = block_table.shape[1]
    max_seq_len = int(seq_lens.max().item())
    num_q_blocks = (max_seq_len + BLOCK_M - 1) // BLOCK_M

    output = torch.empty_like(q)

    # Ensure inputs are contiguous
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    sparse_mask_int = sparse_mask.to(torch.int8).contiguous()

    grid = (batch_size, H_q, num_q_blocks)

    paged_sparse_attn_kernel[grid](
        q, k_cache, v_cache,
        output,
        block_table,
        sparse_mask_int,
        seq_lens,
        cu_seqlens_start,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2),
        # K strides
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        # V strides
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        # Out strides
        output.stride(0), output.stride(1), output.stride(2),
        # block_table strides
        block_table.stride(0), block_table.stride(1),
        # sparse_mask strides
        sparse_mask_int.stride(0), sparse_mask_int.stride(1),
        # constexpr
        H_q=H_q,
        H_kv=H_kv,
        GQA_RATIO=GQA_RATIO,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        # scalar
        softmax_scale=softmax_scale,
        max_pages_per_seq=max_pages,
    )

    return output
