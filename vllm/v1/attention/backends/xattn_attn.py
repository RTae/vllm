# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer using XAttention for sparse prefill and FlashAttention for
decode.

XAttention (https://github.com/mit-han-lab/xattention) accelerates prefill by
estimating which 128-token KV blocks each query block attends to (using a
cheap strided dot-product), then calling block_sparse_attn to compute only
those selected blocks.

Usage:
    llm = LLM(model=..., attention_backend="XATTN",
              enforce_eager=True)

Environment variables:
    XATTN_THRESHOLD   – fraction of blocks to keep (default 0.8, lower = sparser)
    XATTN_STRIDE      – stride for the estimation step (default 8)
    XATTN_BLOCK_SIZE  – sparse block granularity in tokens (default 128, must be 128)
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
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )

# Minimum prefill length to use XAttention (shorter → dense FlashAttention)
_XATTN_MIN_PREFILL_LEN = 256


class XAttentionBackend(AttentionBackend):
    """Attention backend that uses XAttention for long prefill, FlashAttention
    for decode and short prefill sequences."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "XATTN"

    @staticmethod
    def get_impl_cls():
        return XAttentionImpl

    @staticmethod
    def get_builder_cls():
        return XAttentionMetadataBuilder

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
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False


# Re-use FlashAttention metadata – structure is identical.
XAttentionMetadata = FlashAttentionMetadata


class XAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Thin wrapper – metadata construction is identical to FlashAttention."""

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        # XAttn does not support CUDA graphs for prefill.
        return AttentionCGSupport.UNIFORM_BATCH


class XAttentionImpl(AttentionImpl):
    """Attention implementation:
    - prefill (q_len >= _XATTN_MIN_PREFILL_LEN): XAttention sparse attention
    - prefill (q_len < _XATTN_MIN_PREFILL_LEN): FlashAttention dense
    - decode:                                   FlashAttention paged dense
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
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError(
                "XAttentionImpl does not support ALiBi slopes."
            )
        if sliding_window is not None:
            raise NotImplementedError(
                "XAttentionImpl does not support sliding window attention."
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                f"XAttentionImpl does not support kv_cache_dtype={kv_cache_dtype}."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # Configurable via env vars
        self.threshold = float(os.environ.get("XATTN_THRESHOLD", "0.8"))
        self.stride = int(os.environ.get("XATTN_STRIDE", "8"))
        self.block_size = int(os.environ.get("XATTN_BLOCK_SIZE", "128"))

        self.vllm_flash_attn_version = get_flash_attn_version()

        try:
            from xattn.src.Xattention import Xattention_prefill
            self._xattn_prefill = Xattention_prefill
            logger.info_once(
                "XAttention backend loaded (threshold=%.2f, stride=%d, "
                "block_size=%d). Long prefills use sparse attention; "
                "decode falls back to FlashAttention.",
                self.threshold, self.stride, self.block_size,
            )
        except ImportError as exc:
            raise ImportError(
                "XAttention backend requires the xattn package at "
                "/workspace/vllm/xattn. Make sure sys.path includes it."
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
            torch.ones(1, device=key.device),
            torch.ones(1, device=key.device),
        )
        return key_cache, value_cache

    # ------------------------------------------------------------------
    # Prefill helpers
    # ------------------------------------------------------------------

    def _gather_kv_dense(
        self,
        key_cache: torch.Tensor,   # [num_blocks, block_size, Hkv, D]
        value_cache: torch.Tensor,
        block_table_row: torch.Tensor,  # [max_blocks] GPU tensor
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged KV into dense [1, H, N, D] tensors for XAttention."""
        block_size = key_cache.shape[1]
        num_blocks_needed = math.ceil(kv_len / block_size)
        block_ids = block_table_row[:num_blocks_needed]

        k_blocks = torch.index_select(key_cache, 0, block_ids)
        v_blocks = torch.index_select(value_cache, 0, block_ids)
        k_flat = k_blocks.reshape(-1, self.num_kv_heads, self.head_size)
        v_flat = v_blocks.reshape(-1, self.num_kv_heads, self.head_size)
        # Trim to exact kv_len
        k_seq = k_flat[:kv_len].contiguous()
        v_seq = v_flat[:kv_len].contiguous()

        # GQA expansion: repeat KV heads to match Q head count
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_seq = k_seq.repeat_interleave(repeat, dim=1)
            v_seq = v_seq.repeat_interleave(repeat, dim=1)

        # [N, H, D] → [1, H, N, D]
        k_b = k_seq.permute(1, 0, 2).unsqueeze(0).contiguous()
        v_b = v_seq.permute(1, 0, 2).unsqueeze(0).contiguous()
        return k_b, v_b

    def _forward_prefill_xattn(
        self,
        query: torch.Tensor,            # [total_q, H, D]
        key_cache: torch.Tensor,        # [num_blocks, block_size, Hkv, D]
        value_cache: torch.Tensor,
        attn_metadata: XAttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        query_start_loc = attn_metadata.query_start_loc.cpu().tolist()
        seq_lens = attn_metadata.seq_lens.cpu().tolist()
        block_table_gpu = attn_metadata.block_table
        num_reqs = len(query_start_loc) - 1

        for i in range(num_reqs):
            q_start = query_start_loc[i]
            q_end = query_start_loc[i + 1]
            q_len = q_end - q_start
            kv_len = int(seq_lens[i])

            if q_len == 0:
                continue

            q_seq = query[q_start:q_end].contiguous()  # [q_len, H, D]

            # Short sequences: fall back to dense FlashAttention
            if q_len < _XATTN_MIN_PREFILL_LEN:
                k_b, v_b = self._gather_kv_dense(
                    key_cache, value_cache, block_table_gpu[i], kv_len
                )
                # [1, H, N, D] → [N, H, D]
                k_seq = k_b[0].permute(1, 0, 2)
                v_seq = v_b[0].permute(1, 0, 2)
                cu_seqlens_q = torch.tensor(
                    [0, q_len], dtype=torch.int32, device=query.device
                )
                seqused_k = torch.tensor(
                    [kv_len], dtype=torch.int32, device=query.device
                )
                flash_attn_varlen_func(
                    q=q_seq,
                    k=k_seq,
                    v=v_seq,
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

            # Long sequences: use XAttention sparse prefill
            k_b, v_b = self._gather_kv_dense(
                key_cache, value_cache, block_table_gpu[i], kv_len
            )
            # q: [q_len, H, D] → [1, H, q_len, D]
            q_b = q_seq.permute(1, 0, 2).unsqueeze(0).contiguous()

            try:
                out_b = self._xattn_prefill(
                    query_states=q_b,
                    key_states=k_b,
                    value_states=v_b,
                    stride=self.stride,
                    norm=self.scale * math.sqrt(self.head_size),
                    threshold=self.threshold,
                    block_size=self.block_size,
                    use_triton=False,  # A100 is sm80, triton kernel needs sm100
                    causal=True,
                )  # [1, H, q_len, D]  (batch=1)
                # [1, H, q_len, D] → [q_len, H, D]
                output[q_start:q_end] = out_b[0].permute(1, 0, 2)
            except Exception as exc:
                # On any xattn error fall back to dense FlashAttention
                logger.warning(
                    "XAttention prefill failed (seq_idx=%d, q_len=%d, "
                    "kv_len=%d): %s. Falling back to FlashAttention.",
                    i, q_len, kv_len, exc,
                )
                k_seq = k_b[0].permute(1, 0, 2)
                v_seq = v_b[0].permute(1, 0, 2)
                cu_seqlens_q = torch.tensor(
                    [0, q_len], dtype=torch.int32, device=query.device
                )
                seqused_k = torch.tensor(
                    [kv_len], dtype=torch.int32, device=query.device
                )
                flash_attn_varlen_func(
                    q=q_seq,
                    k=k_seq,
                    v=v_seq,
                    out=output[q_start:q_end],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=q_len,
                    seqused_k=seqused_k,
                    max_seqlen_k=kv_len,
                    softmax_scale=self.scale,
                    causal=True,
                    fa_version=self.vllm_flash_attn_version,
                )

    # ------------------------------------------------------------------
    # Decode: standard FlashAttention (paged, dense)
    # ------------------------------------------------------------------

    def _forward_decode_flash(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: XAttentionMetadata,
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
    # forward()
    # ------------------------------------------------------------------

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported in XAttentionImpl."
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = self._write_kv_cache(
            key, value, kv_cache, attn_metadata.slot_mapping, num_actual_tokens
        )

        q = query[:num_actual_tokens]
        out = output[:num_actual_tokens]

        if attn_metadata.max_query_len > 1:
            self._forward_prefill_xattn(
                q, key_cache, value_cache, attn_metadata, out
            )
        else:
            self._forward_decode_flash(
                q, key_cache, value_cache, attn_metadata, out
            )

        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        pass
