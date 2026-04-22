"""Driver-JIT replacement for the ptxas subprocess path.

`compile_theirs_jit(ptx)` compiles PTX to cubin via libcuda's cuLink
API.  No subprocess, no temp files — ~80x faster than spawning
ptxas.exe per call on Windows.

The driver invokes the same ptxas code internally, so correctness is
equivalent.  Cubin layout differs (subprocess-ptxas embeds debug info
by default; JIT path strips it) but the kernel text + launch semantics
are identical.
"""
import ctypes
from typing import Optional, Tuple

# cuLink JIT option codes (from cuda.h)
_CU_JIT_TARGET                    = 9
_CU_JIT_INFO_LOG_BUFFER           = 3
_CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
_CU_JIT_ERROR_LOG_BUFFER          = 5
_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6

_CU_JIT_INPUT_PTX                 = 1

# Lazy global cuda handle (libcuda is already loaded by the fuzzer's
# existing CudaRunner; we just grab another reference).
_cuda = None

def _libcuda():
    global _cuda
    if _cuda is None:
        import sys
        if sys.platform.startswith('win'):
            _cuda = ctypes.WinDLL('nvcuda')
        else:
            _cuda = ctypes.CDLL('libcuda.so.1')
    return _cuda


# A context pushed by the caller (e.g. the differ daemon) so this
# module can re-install it on the current thread if libcuda has
# dropped the association.
_pushed_ctx: Optional[ctypes.c_void_p] = None


def set_context(ctx: ctypes.c_void_p):
    """Caller registers its CUDA context with the JIT module so the
    JIT call can re-push it if needed before cuLinkCreate."""
    global _pushed_ctx
    _pushed_ctx = ctx


def compile_theirs_jit(ptx: str, sm: int = 120) -> Tuple[Optional[bytes], Optional[str]]:
    """Compile PTX through the CUDA driver's PTX JIT.  Returns (cubin, err).

    Requires an active CUDA context (the differ's CudaRunner already
    creates one).  Thread-safe as long as each caller serializes its
    own cuLink* calls.
    """
    cuda = _libcuda()
    ptx_bytes = ptx.encode() + b'\0'

    # Always push the registered context — the "currently current"
    # context may be a previously-destroyed handle (rc=709 from
    # cuLinkCreate otherwise).  Cheap no-op if already current.
    if _pushed_ctx is None or not _pushed_ctx.value:
        return None, 'no context registered; caller must set_context()'
    cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    cuda.cuCtxSetCurrent.restype = ctypes.c_int
    rc = cuda.cuCtxSetCurrent(_pushed_ctx)
    if rc != 0:
        return None, f'cuCtxSetCurrent rc={rc}'

    # Options: target SM + error log buffer
    err_buf = ctypes.create_string_buffer(4096)
    opts = (ctypes.c_int * 3)(
        _CU_JIT_TARGET,
        _CU_JIT_ERROR_LOG_BUFFER,
        _CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
    vals = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.c_void_p(sm), ctypes.c_void_p),
        ctypes.cast(err_buf, ctypes.c_void_p),
        ctypes.cast(ctypes.c_void_p(len(err_buf)), ctypes.c_void_p))
    state = ctypes.c_void_p()
    rc = cuda.cuLinkCreate_v2(3, opts, vals, ctypes.byref(state))
    if rc != 0:
        return None, f'cuLinkCreate rc={rc}'
    try:
        rc = cuda.cuLinkAddData_v2(state, _CU_JIT_INPUT_PTX,
                                    ctypes.c_char_p(ptx_bytes),
                                    ctypes.c_size_t(len(ptx_bytes)),
                                    ctypes.c_char_p(b'k.ptx'),
                                    0, None, None)
        if rc != 0:
            msg = err_buf.value.decode('utf-8', errors='replace')[:200]
            return None, msg or f'cuLinkAddData rc={rc}'
        cubin_ptr = ctypes.c_void_p()
        cubin_sz = ctypes.c_size_t()
        rc = cuda.cuLinkComplete(state, ctypes.byref(cubin_ptr), ctypes.byref(cubin_sz))
        if rc != 0:
            msg = err_buf.value.decode('utf-8', errors='replace')[:200]
            return None, msg or f'cuLinkComplete rc={rc}'
        return ctypes.string_at(cubin_ptr, cubin_sz.value), None
    finally:
        cuda.cuLinkDestroy(state)
