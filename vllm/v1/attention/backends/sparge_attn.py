# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with SpargeAttention for prefill and FlashAttention for
decode.

SpargeAttention (https://arxiv.org/abs/2502.18137) computes sparse attention
during the prefill (prompt-processing) phase by estimating which KV blocks are
important and skipping the rest.  For decode (single-token steps) we fall back
to a standard FlashAttention-2/3 call because SpargeAttention is designed for
long-context prefill and the overhead would not be justified for a 1-token
query.

Usage (offline):
    llm = LLM(model=..., attention_backend="SPARGE_ATTN",
              attention_backend_kwargs={"topk": 0.5})

The ``topk`` parameter (default 0.5) controls the fraction of KV blocks
retained per attention head.  A value of 1.0 is equivalent to dense attention.
"""

import math
import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.paged_sparse_attn_kernel import (
    paged_sparse_attn_kernel,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )


class SpargeAttentionBackend(AttentionBackend):
    """Attention backend that uses SpargeAttention for prefill and
    FlashAttention for decode."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "SPARGE_ATTN"

    @staticmethod
    def get_impl_cls():
        return SpargeAttentionImpl

    @staticmethod
    def get_builder_cls():
        return SpargeAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Same layout as FlashAttention: [num_blocks, 2, block_size, num_kv_heads, head_size]
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False


# Re-use FlashAttention metadata and builder – the metadata structure is
# identical; we only swap the kernel called during forward().
SpargeAttentionMetadata = FlashAttentionMetadata


class SpargeAttentionMetadataBuilder(
    FlashAttentionMetadataBuilder,
):
    """Thin wrapper – metadata construction is identical to FlashAttention."""

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        # Be conservative: SpargeAttn does not support CUDA graphs for prefill.
        return AttentionCGSupport.UNIFORM_BATCH


class SpargeAttentionImpl(AttentionImpl):
    """Attention implementation that dispatches:
    - prefill tokens  →  spas_sage2_attn_meansim_topk_cuda  (sparse)
    - decode  tokens  →  flash_attn_varlen_func              (dense)
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        topk: float = 0.5,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError(
                "SpargeAttentionImpl does not support ALiBi slopes."
            )
        if sliding_window is not None:
            raise NotImplementedError(
                "SpargeAttentionImpl does not support sliding window attention."
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                f"SpargeAttentionImpl does not support kv_cache_dtype={kv_cache_dtype}."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # topk can be overridden via the SPARGE_ATTN_TOPK environment variable.
        import os
        env_topk = os.environ.get("SPARGE_ATTN_TOPK")
        if env_topk is not None:
            topk = float(env_topk)
        self.topk = float(topk)

        self.vllm_flash_attn_version = get_flash_attn_version()

        try:
            from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
            self._sparge_fn = spas_sage2_attn_meansim_topk_cuda
            logger.info_once(
                "SpargeAttention loaded (topk=%.2f). Prefill uses paged sparse "
                "Triton kernel (no gather, native GQA); decode falls back to "
                "FlashAttention.",
                self.topk,
            )
        except ImportError as exc:
            raise ImportError(
                "SpargeAttention backend requires the spas_sage_attn package. "
                "Install it with: uv pip install -e ./spas_sage_attn"
            ) from exc

        # Whether to emit debug diff vs FlashAttention (set SPARGE_DEBUG=1)
        self._debug = bool(os.environ.get("SPARGE_DEBUG"))
        self._debug_first_call = True

    # ------------------------------------------------------------------
    # KV-cache write (identical to FlashAttention)
    # ------------------------------------------------------------------

    def _write_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = kv_cache.unbind(1)
        reshape_and_cache_flash(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            key_cache,
            value_cache,
            slot_mapping[:num_actual_tokens],
            self.kv_cache_dtype,
            # scale tensors are not used for non-quantised caches
            torch.ones(1, device=key.device),
            torch.ones(1, device=key.device),
        )
        return key_cache, value_cache

    # ------------------------------------------------------------------
    # Sparse mask computation (block-level importance scoring)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sparse_mask(
        q_seq: torch.Tensor,   # [N, H_q, D]  dense, no copy
        k_seq: torch.Tensor,   # [N, H_kv, D] dense, no copy
        topk: float,
        page_size: int,
    ) -> torch.Tensor:
        """Compute a per-page boolean sparse mask using a single representative
        query token (the last token — most information-rich under causal attention).

        Scoring: last_q_token [H_kv, D] @ k_page_mean [P, H_kv, D]^T → [P] scores.
        This is O(P × H_kv × D) per layer, ~4× faster than the O(P²) approach.
        GQA heads are averaged before scoring.
        Always keeps first page (attention sink) and last page (recent tokens).

        Args:
            q_seq:     [N, H_q, D]   query tokens (dense)
            k_seq:     [N, H_kv, D]  key tokens (dense)
            topk:      fraction of pages to keep (0.5 = keep 50%)
            page_size: tokens per page (must match vLLM block size)

        Returns:
            sparse_mask: [num_pages] bool, True = attend this page
        """
        N, H_q, D = q_seq.shape
        H_kv = k_seq.shape[1]
        GQA_RATIO = H_q // H_kv
        num_pages = (N + page_size - 1) // page_size

        if num_pages <= 2:
            return torch.ones(num_pages, dtype=torch.bool, device=q_seq.device)

        # Pad K to multiple of page_size
        pad_len = num_pages * page_size - N
        k_f = k_seq.to(torch.float32)
        if pad_len > 0:
            k_f = torch.nn.functional.pad(k_f, (0, 0, 0, 0, 0, pad_len))

        # KV page means: [P, H_kv, D]
        k_means = k_f.reshape(num_pages, page_size, H_kv, D).mean(dim=1)

        # Use last query token as representative — it must attend the most
        # recent context and is the most demanding under causal attention.
        q_rep = q_seq[-1].to(torch.float32)  # [H_q, D]

        # GQA: average query heads within each KV-head group → [H_kv, D]
        if GQA_RATIO > 1:
            q_rep = q_rep.reshape(H_kv, GQA_RATIO, D).mean(dim=1)

        # Scores: [P] — dot product of representative query vs each page mean
        scores = torch.einsum('hd,phd->p', q_rep, k_means) / (D ** 0.5)

        # Always attend first (sink) and last (recency) pages
        scores[0] = float('inf')
        scores[-1] = float('inf')

        # Keep top-k% pages
        num_keep = max(2, int(topk * num_pages))
        threshold = torch.topk(scores, num_keep).values.min()
        return scores >= threshold  # [num_pages] bool

    # ------------------------------------------------------------------
    # Single-sequence paged sparse attention launcher
    # ------------------------------------------------------------------

    def _paged_sparse_attn_single(
        self,
        q_seq: torch.Tensor,       # [N, H_q, D]
        key_cache: torch.Tensor,   # [num_blocks, block_size, H_kv, D]
        value_cache: torch.Tensor,
        block_table_row: torch.Tensor,  # [max_pages] int32
        seq_len: int,
        sparse_mask: torch.Tensor, # [num_pages] bool
    ) -> torch.Tensor:
        """Launch paged_sparse_attn_kernel for a single sequence."""
        N, H_q, D = q_seq.shape
        page_size = key_cache.shape[1]
        H_kv = key_cache.shape[2]
        GQA_RATIO = H_q // H_kv
        BLOCK_M = 64
        BLOCK_N = page_size
        num_pages = sparse_mask.shape[0]
        max_pages = block_table_row.shape[0]

        # Wrap single sequence in batch-1 tensors
        block_table_b1 = block_table_row.unsqueeze(0)           # [1, max_pages]
        sparse_mask_int = sparse_mask.to(torch.int8)
        # Pad sparse_mask_int to max_pages if needed
        if num_pages < max_pages:
            pad = torch.zeros(max_pages - num_pages, dtype=torch.int8,
                              device=q_seq.device)
            sparse_mask_int = torch.cat([sparse_mask_int, pad])
        sparse_mask_b1 = sparse_mask_int.unsqueeze(0)           # [1, max_pages]

        seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device=q_seq.device)
        cu_seqlens_t = torch.tensor([0], dtype=torch.int32, device=q_seq.device)
        num_q_blocks = (N + BLOCK_M - 1) // BLOCK_M

        output = torch.empty_like(q_seq)

        grid = (1, H_q, num_q_blocks)
        paged_sparse_attn_kernel[grid](
            q_seq, key_cache, value_cache,
            output,
            block_table_b1,
            sparse_mask_b1,
            seq_lens_t,
            cu_seqlens_t,
            # Q strides
            q_seq.stride(0), q_seq.stride(1), q_seq.stride(2),
            # K strides
            key_cache.stride(0), key_cache.stride(1),
            key_cache.stride(2), key_cache.stride(3),
            # V strides
            value_cache.stride(0), value_cache.stride(1),
            value_cache.stride(2), value_cache.stride(3),
            # Out strides
            output.stride(0), output.stride(1), output.stride(2),
            # block_table strides
            block_table_b1.stride(0), block_table_b1.stride(1),
            # sparse_mask strides
            sparse_mask_b1.stride(0), sparse_mask_b1.stride(1),
            # constexpr
            H_q=H_q,
            H_kv=H_kv,
            GQA_RATIO=GQA_RATIO,
            HEAD_DIM=D,
            PAGE_SIZE=page_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            # scalar
            softmax_scale=self.scale,
            max_pages_per_seq=max_pages,
        )
        return output

    # ------------------------------------------------------------------
    # Prefill: paged sparse Triton kernel (no gather, native GQA)
    # ------------------------------------------------------------------

    def _forward_prefill_sparge(
        self,
        query: torch.Tensor,          # [total_q, H_q, D]
        key_raw: torch.Tensor,        # [total_q, H_kv, D] — dense, from forward()
        value_raw: torch.Tensor,      # [total_q, H_kv, D] — dense, from forward()
        key_cache: torch.Tensor,      # [num_blocks, block_size, H_kv, D]
        value_cache: torch.Tensor,
        attn_metadata: SpargeAttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """Run sparse prefill using the paged Triton kernel.

        Eliminates both overheads from the original SpargeAttn approach:
          1. index_select (paged → dense gather):   ~3.6ms/layer × 28 = 101ms
          2. repeat_interleave (GQA head expand):   ~1.9ms/layer × 28 =  53ms

        The Triton kernel reads k_cache/v_cache pages directly via block_table,
        applies the sparse mask BEFORE loading any KV, and handles GQA natively
        via  kv_head = q_head // GQA_RATIO  inside the kernel.
        """
        query_start_loc = attn_metadata.query_start_loc.cpu().tolist()
        seq_lens = attn_metadata.seq_lens.cpu().tolist()
        block_table_gpu = attn_metadata.block_table
        num_reqs = len(query_start_loc) - 1

        # All sequences must be full first-prefills (q_len == kv_len, >= 128)
        # for sparse attention to be correct and beneficial.
        all_sparge_eligible = all(
            (query_start_loc[i + 1] - query_start_loc[i]) == int(seq_lens[i])
            and (query_start_loc[i + 1] - query_start_loc[i]) >= 128
            for i in range(num_reqs)
        )

        if not all_sparge_eligible:
            # Mixed batch or chunked prefill — use paged FlashAttention.
            flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                out=output,
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=None,
                window_size=None,
                block_table=block_table_gpu,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
            )
            return

        # All sequences are full first-prefills — use paged sparse Triton kernel.
        page_size = key_cache.shape[1]
        # Minimum sequence length for the paged sparse kernel to beat FlashAttention.
        # Below this threshold FA is faster because page iteration overhead dominates.
        MIN_PAGED_SPARSE_LEN = int(os.environ.get("SPARGE_MIN_LEN", "16384"))

        for i in range(num_reqs):
            q_start = query_start_loc[i]
            q_end = query_start_loc[i + 1]
            kv_len = int(seq_lens[i])

            # Dense slices — no copy, no gather, no GQA expand
            q_seq = query[q_start:q_end]    # [N, H_q, D]
            k_seq = key_raw[q_start:q_end]  # [N, H_kv, D]
            v_seq = value_raw[q_start:q_end]  # noqa: F841 (used via paged kernel)

            # For sequences below the threshold, paged kernel overhead exceeds
            # sparse savings — fall back to dense FlashAttention for this sequence.
            if kv_len < MIN_PAGED_SPARSE_LEN:
                cu_q = torch.tensor([0, kv_len], dtype=torch.int32, device=query.device)
                flash_attn_varlen_func(
                    q=q_seq,
                    k=key_cache,
                    v=value_cache,
                    out=output[q_start:q_end],
                    cu_seqlens_q=cu_q,
                    max_seqlen_q=kv_len,
                    seqused_k=torch.tensor([kv_len], dtype=torch.int32, device=query.device),
                    max_seqlen_k=kv_len,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=block_table_gpu[i:i+1],
                    fa_version=self.vllm_flash_attn_version,
                )
                continue

            # Compute block-level sparse mask from dense K (already available)
            sparse_mask = self.compute_sparse_mask(
                q_seq, k_seq,
                topk=self.topk,
                page_size=page_size,
            )  # [num_pages] bool

            # Launch paged sparse kernel — reads k_cache/v_cache directly,
            # native GQA (no repeat_interleave), skips masked pages entirely.
            out_seq = self._paged_sparse_attn_single(
                q_seq=q_seq,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table_row=block_table_gpu[i],
                seq_len=kv_len,
                sparse_mask=sparse_mask,
            )  # [N, H_q, D]

            output[q_start:q_end] = out_seq

            # Debug: compare against dense FlashAttention on first call
            if self._debug and self._debug_first_call:
                self._debug_first_call = False
                cu_q = torch.tensor([0, q_end - q_start],
                                     dtype=torch.int32, device=query.device)
                ref = flash_attn_varlen_func(
                    q=q_seq,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=cu_q,
                    max_seqlen_q=kv_len,
                    seqused_k=torch.tensor([kv_len], dtype=torch.int32,
                                           device=query.device),
                    max_seqlen_k=kv_len,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=block_table_gpu[i:i+1],
                    fa_version=self.vllm_flash_attn_version,
                )
                diff = (out_seq - ref).abs().max().item()
                logger.debug(
                    "[SPARGE_DEBUG] paged_sparse_attn max diff vs flash: %.6f "
                    "(topk=%.2f, seq_len=%d, num_pages=%d, pages_kept=%d)",
                    diff, self.topk, kv_len, sparse_mask.shape[0],
                    sparse_mask.sum().item(),
                )

    # ------------------------------------------------------------------
    # Decode: fall back to FlashAttention (dense, paged)
    # ------------------------------------------------------------------

    def _forward_decode_flash(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: SpargeAttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=None,
            window_size=None,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
        )

    # ------------------------------------------------------------------
    # forward() – dispatches to prefill or decode
    # ------------------------------------------------------------------

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SpargeAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported in SpargeAttentionImpl."
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = self._write_kv_cache(
            key, value, kv_cache, attn_metadata.slot_mapping, num_actual_tokens
        )

        q = query[:num_actual_tokens]
        out = output[:num_actual_tokens]
        # Keep raw dense KV (before paging) for the no-gather prefill path.
        key_raw = key[:num_actual_tokens]
        value_raw = value[:num_actual_tokens]

        # Detect prefill vs. decode.
        # max_query_len > 1 means at least one sequence is in prefill.
        # For decode-only batches max_query_len == 1.
        if attn_metadata.max_query_len > 1:
            self._forward_prefill_sparge(q, key_raw, value_raw, key_cache, value_cache, attn_metadata, out)
        else:
            self._forward_decode_flash(q, key_cache, value_cache, attn_metadata, out)

        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # KV cache update is performed inside forward() for this backend.
        pass
