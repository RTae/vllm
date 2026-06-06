__version__ = "0.0.2"

# NOTE: We intentionally do NOT preload any torch shared libs here.
# block_sparse_attn_cuda.so is only imported inside XAttentionImpl.__init__,
# which runs in the vLLM worker process (after fork/spawn), so the linker
# can find libc10.so through the worker's already-initialised torch.

from block_sparse_attn.block_sparse_attn_interface import (
    block_sparse_attn_func,
    token_streaming_attn_func,
    block_streaming_attn_func,
)

from . import utils