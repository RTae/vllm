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
                "SpargeAttention loaded (topk=%.2f). Prefill will use sparse "
                "attention; decode will fall back to FlashAttention.",
                self.topk,
            )
        except ImportError as exc:
            raise ImportError(
                "SpargeAttention backend requires the spas_sage_attn package. "
                "Install it with: uv pip install -e ./spas_sage_attn"
            ) from exc

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
    # Prefill: use SpargeAttention (sparse)
    # ------------------------------------------------------------------

    def _forward_prefill_sparge(
        self,
        query: torch.Tensor,          # [total_q, H, D]
        key_cache: torch.Tensor,       # [num_blocks, block_size, Hkv, D]
        value_cache: torch.Tensor,
        attn_metadata: SpargeAttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """Run SpargeAttention on prefill tokens.

        SpargeAttention expects dense (batched) tensors, so we need to
        expand the paged KV cache into a dense representation first for
        each sequence in the batch.
        """
        # Move index tensors to CPU once before the loop to avoid GPU→CPU
        # synchronization inside a compiled/CUDA-graph execution context.
        query_start_loc = attn_metadata.query_start_loc.cpu().tolist()
        seq_lens = attn_metadata.seq_lens.cpu().tolist()
        block_table_cpu = attn_metadata.block_table.cpu()
        num_reqs = len(query_start_loc) - 1

        for i in range(num_reqs):
            q_start = query_start_loc[i]
            q_end = query_start_loc[i + 1]
            q_len = q_end - q_start
            kv_len = int(seq_lens[i])

            if q_len == 0:
                continue

            # Gather KV pages for this sequence.
            block_table_row = block_table_cpu[i]
            num_blocks_needed = math.ceil(kv_len / key_cache.shape[1])
            block_ids = block_table_row[:num_blocks_needed].tolist()

            # key_cache: [num_blocks, block_size, Hkv, D]  →
            #            flatten to  [num_blocks * block_size, Hkv, D]
            # then slice to kv_len.
            block_size = key_cache.shape[1]
            k_flat = key_cache[block_ids].reshape(-1, self.num_kv_heads, self.head_size)
            v_flat = value_cache[block_ids].reshape(-1, self.num_kv_heads, self.head_size)
            k_seq = k_flat[:kv_len]   # [kv_len, Hkv, D]
            v_seq = v_flat[:kv_len]

            # q_seq: [q_len, H, D]
            q_seq = query[q_start:q_end]

            # SpargeAttention requires at least 128 query tokens.
            # For shorter sequences fall back to FlashAttention exactly
            # like the decode path does, but scoped to this one sequence.
            if q_len < 128:
                cu_seqlens_q = torch.tensor(
                    [0, q_len], dtype=torch.int32, device=query.device
                )
                seqused_k = torch.tensor(
                    [kv_len], dtype=torch.int32, device=query.device
                )
                flash_attn_varlen_func(
                    q=q_seq,
                    k=k_flat,
                    v=v_flat,
                    out=output[q_start:q_end],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=q_len,
                    seqused_k=seqused_k,
                    max_seqlen_k=kv_len,
                    softmax_scale=self.scale,
                    causal=True,
                    fa_version=self.vllm_flash_attn_version,
                )
                continue

            # SpargeAttention API wants layout NHD → already our layout.
            # Add batch dim: [1, T, H, D]
            q_b = q_seq.unsqueeze(0)    # [1, q_len, H,   D]
            k_b = k_seq.unsqueeze(0)    # [1, kv_len, Hkv, D]
            v_b = v_seq.unsqueeze(0)    # [1, kv_len, Hkv, D]

            # GQA: expand KV heads to match Q heads if needed.
            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                k_b = k_b.repeat_interleave(repeat, dim=2)
                v_b = v_b.repeat_interleave(repeat, dim=2)

            out_b = self._sparge_fn(
                q_b, k_b, v_b,
                is_causal=True,
                scale=self.scale,
                topk=self.topk,
                tensor_layout="NHD",
                output_dtype=query.dtype,
            )  # [1, q_len, H, D]

            output[q_start:q_end] = out_b[0]

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

        # Detect prefill vs. decode.
        # max_query_len > 1 means at least one sequence is in prefill.
        # For decode-only batches max_query_len == 1.
        if attn_metadata.max_query_len > 1:
            self._forward_prefill_sparge(q, key_cache, value_cache, attn_metadata, out)
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
