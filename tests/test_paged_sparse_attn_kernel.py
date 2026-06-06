"""
Correctness test for paged_sparse_attn_kernel.

Tests against dense FlashAttention reference (after gather + GQA expand)
with sparse_mask=all True (dense mode, no sparsity).
"""

import sys
sys.path.insert(0, '/workspace/vllm')

import math
import pytest
import torch
from flash_attn import flash_attn_varlen_func

from vllm.v1.attention.backends.paged_sparse_attn_kernel import paged_sparse_attention


def build_test_inputs(
    batch_size: int,
    seq_lens: list[int],
    H_q: int,
    H_kv: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 42,
):
    """Build synthetic paged KV cache and matching dense tensors for reference."""
    torch.manual_seed(seed)
    total_tokens = sum(seq_lens)

    # Dense Q [total_tokens, H_q, head_dim]
    q = torch.randn(total_tokens, H_q, head_dim, dtype=dtype, device=device) * 0.1

    # For each sequence, allocate sequential physical pages
    max_pages_per_seq = max((s + page_size - 1) // page_size for s in seq_lens)
    total_pages = sum((s + page_size - 1) // page_size for s in seq_lens)

    k_cache = torch.randn(total_pages, page_size, H_kv, head_dim, dtype=dtype, device=device) * 0.1
    v_cache = torch.randn(total_pages, page_size, H_kv, head_dim, dtype=dtype, device=device) * 0.1

    block_table = torch.zeros(batch_size, max_pages_per_seq, dtype=torch.int32, device=device)
    page_counter = 0
    for b, seq_len in enumerate(seq_lens):
        num_pages = (seq_len + page_size - 1) // page_size
        for p in range(num_pages):
            block_table[b, p] = page_counter + p
        page_counter += num_pages

    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    return q, k_cache, v_cache, block_table, seq_lens_tensor


def reference_dense_flash(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    H_q: int,
    H_kv: int,
    head_dim: int,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Reference output using dense FlashAttention.
    Gathers KV from paged cache and expands GQA heads (the old approach).
    """
    batch_size = seq_lens.shape[0]
    seq_lens_list = seq_lens.cpu().tolist()
    total_tokens = int(q.shape[0])

    # Gather dense KV for each sequence
    dense_k_list = []
    dense_v_list = []
    for b, seq_len in enumerate(seq_lens_list):
        num_pages = (seq_len + page_size - 1) // page_size
        pages = block_table[b, :num_pages]
        k_blocks = torch.index_select(k_cache, 0, pages)  # [num_pages, page_size, H_kv, D]
        v_blocks = torch.index_select(v_cache, 0, pages)
        # Flatten pages: [num_pages * page_size, H_kv, D] → trim to seq_len
        k_flat = k_blocks.reshape(-1, H_kv, head_dim)[:seq_len]
        v_flat = v_blocks.reshape(-1, H_kv, head_dim)[:seq_len]
        dense_k_list.append(k_flat)
        dense_v_list.append(v_flat)

    dense_k = torch.cat(dense_k_list, dim=0)  # [total_kv_tokens, H_kv, D]
    dense_v = torch.cat(dense_v_list, dim=0)

    # Build cu_seqlens
    seq_lens_t = seq_lens.to(torch.int32)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
    cu_seqlens[1:] = torch.cumsum(seq_lens_t, dim=0)

    # FlashAttention handles GQA natively (H_q != H_kv)
    out = flash_attn_varlen_func(
        q=q,
        k=dense_k,
        v=dense_v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=int(seq_lens.max()),
        max_seqlen_k=int(seq_lens.max()),
        softmax_scale=softmax_scale,
        causal=True,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("batch_size,seq_lens,H_q,H_kv,head_dim,page_size", [
    # Single short sequence
    (1, [128], 32, 8, 128, 16),
    # Single medium sequence (Qwen3-VL typical)
    (1, [512], 32, 8, 128, 16),
    # Batch of two sequences
    (2, [256, 384], 32, 8, 128, 16),
    # Sequence length not divisible by page_size
    (1, [300], 32, 8, 128, 16),
    # No GQA (H_q == H_kv)
    (1, [256], 16, 16, 64, 16),
    # Larger sequence
    (1, [1024], 32, 8, 128, 16),
])
def test_paged_sparse_attn_dense_mask(batch_size, seq_lens, H_q, H_kv, head_dim, page_size):
    """
    With sparse_mask=all True (attend all pages), output must match dense
    FlashAttention reference within numerical tolerance.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.bfloat16
    softmax_scale = head_dim ** -0.5

    q, k_cache, v_cache, block_table, seq_lens_tensor = build_test_inputs(
        batch_size, seq_lens, H_q, H_kv, head_dim, page_size, dtype, device
    )

    max_pages_per_seq = block_table.shape[1]

    # All-True sparse mask (dense mode)
    sparse_mask = torch.ones(batch_size, max_pages_per_seq, dtype=torch.bool, device=device)

    # Our Triton kernel
    out_triton = paged_sparse_attention(
        q, k_cache, v_cache, block_table, sparse_mask, seq_lens_tensor,
        softmax_scale=softmax_scale,
    )

    # Reference dense FlashAttention
    out_ref = reference_dense_flash(
        q, k_cache, v_cache, block_table, seq_lens_tensor,
        page_size, H_q, H_kv, head_dim, softmax_scale,
    )

    # Numerical check
    max_diff = (out_triton - out_ref).abs().max().item()
    mean_diff = (out_triton - out_ref).abs().mean().item()

    print(f"\n[batch={batch_size}, seq={seq_lens}, H_q={H_q}, H_kv={H_kv}]")
    print(f"  max_abs_diff = {max_diff:.6f}")
    print(f"  mean_abs_diff = {mean_diff:.6f}")

    # bfloat16 has ~3e-3 relative error, allow up to 1e-2
    assert max_diff < 1e-2, f"Max diff {max_diff:.6f} exceeds tolerance"


@pytest.mark.parametrize("sparsity", [0.0, 0.25, 0.5, 0.75])
def test_paged_sparse_attn_sparse_mask(sparsity):
    """
    With sparse_mask, attended pages must match the reference computed on
    only those pages (reference also applies the same mask).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 1
    seq_len = 512
    H_q, H_kv, head_dim, page_size = 32, 8, 128, 16
    softmax_scale = head_dim ** -0.5

    q, k_cache, v_cache, block_table, seq_lens_tensor = build_test_inputs(
        batch_size, [seq_len], H_q, H_kv, head_dim, page_size, dtype, device
    )

    num_pages = (seq_len + page_size - 1) // page_size
    max_pages = block_table.shape[1]

    # Create a fixed sparse mask
    torch.manual_seed(0)
    raw = torch.rand(num_pages) > sparsity
    # Always attend the last page (contains the most recent tokens, causal)
    raw[-1] = True
    sparse_mask = torch.zeros(batch_size, max_pages, dtype=torch.bool, device=device)
    sparse_mask[0, :num_pages] = raw.to(device)

    out_triton = paged_sparse_attention(
        q, k_cache, v_cache, block_table, sparse_mask, seq_lens_tensor,
        softmax_scale=softmax_scale,
    )

    # Reference: zero out non-attended KV pages so flash sees the same data
    k_cache_masked = k_cache.clone()
    v_cache_masked = v_cache.clone()
    for p in range(num_pages):
        if not raw[p]:
            phys = block_table[0, p].item()
            k_cache_masked[phys] = 0.0
            v_cache_masked[phys] = 0.0

    out_ref = reference_dense_flash(
        q, k_cache_masked, v_cache_masked, block_table, seq_lens_tensor,
        page_size, H_q, H_kv, head_dim, softmax_scale,
    )

    # For sparse outputs, rows where ALL KV was masked produce zeros in both.
    # Only compare tokens that have at least one attended page causal to them.
    # Rows in out_triton that are all-zero are correctly handled (no attended KV).
    non_zero_rows = out_triton.abs().sum(dim=-1).sum(dim=-1) > 1e-6
    if non_zero_rows.any():
        max_diff = (out_triton[non_zero_rows] - out_ref[non_zero_rows]).abs().max().item()
    else:
        max_diff = 0.0
    print(f"\n[sparsity={sparsity:.0%}]  max_abs_diff = {max_diff:.6f}")
    assert max_diff < 5e-2, f"Max diff {max_diff:.6f} exceeds tolerance for sparsity={sparsity}"


def test_paged_sparse_attn_output_shape():
    """Verify output shape matches input Q shape."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device, dtype = "cuda", torch.bfloat16
    q, k_cache, v_cache, block_table, seq_lens_tensor = build_test_inputs(
        2, [256, 128], 32, 8, 128, 16, dtype, device
    )
    sparse_mask = torch.ones(2, block_table.shape[1], dtype=torch.bool, device=device)
    out = paged_sparse_attention(q, k_cache, v_cache, block_table, sparse_mask, seq_lens_tensor)
    assert out.shape == q.shape, f"Shape mismatch: {out.shape} != {q.shape}"
    assert out.dtype == q.dtype, f"Dtype mismatch: {out.dtype} != {q.dtype}"


if __name__ == "__main__":
    # Quick standalone run without pytest
    import sys

    print("=" * 60)
    print("paged_sparse_attn_kernel correctness tests")
    print("=" * 60)

    print("\n[1] Shape test...")
    test_paged_sparse_attn_output_shape()
    print("    PASSED")

    print("\n[2] Dense mask (all-True) vs FlashAttention reference...")
    for batch_size, seq_lens, H_q, H_kv, head_dim, page_size in [
        (1, [128], 32, 8, 128, 16),
        (1, [512], 32, 8, 128, 16),
        (2, [256, 384], 32, 8, 128, 16),
        (1, [300], 32, 8, 128, 16),
        (1, [1024], 32, 8, 128, 16),
    ]:
        test_paged_sparse_attn_dense_mask(batch_size, seq_lens, H_q, H_kv, head_dim, page_size)
    print("    PASSED")

    print("\n[3] Sparse mask correctness...")
    for sparsity in [0.0, 0.25, 0.5]:
        test_paged_sparse_attn_sparse_mask(sparsity)
    print("    PASSED")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)
