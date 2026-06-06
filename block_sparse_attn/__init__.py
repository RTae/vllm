__version__ = "0.0.2"

# Pre-load torch shared libs so the CUDA extension can find libc10.so / libtorch.so
import ctypes as _ctypes
import os as _os
import torch as _torch
_torch_lib = _os.path.join(_os.path.dirname(_torch.__file__), "lib")
for _lib in ("libc10.so", "libtorch.so", "libtorch_cuda.so", "libc10_cuda.so"):
    _path = _os.path.join(_torch_lib, _lib)
    if _os.path.exists(_path):
        _ctypes.CDLL(_path, mode=_ctypes.RTLD_GLOBAL)

from block_sparse_attn.block_sparse_attn_interface import (
    block_sparse_attn_func,
    token_streaming_attn_func,
    block_streaming_attn_func,
)

from . import utils