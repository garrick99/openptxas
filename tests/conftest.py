"""Shared CUDA context for all GPU test files.

Uses the device primary context (cuDevicePrimaryCtxRetain) which is
reference-counted and shared across all callers — avoids the
cuCtxCreate stacking problem that causes ERR715 / context limit
errors when multiple test modules run in a single pytest session.
"""
import ctypes
import struct
import pytest


def _get_cuda():
    try:
        cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        if cuda.cuInit(0) != 0:
            return None
        return cuda
    except Exception:
        return None


_CUDA = _get_cuda()


class CUDAContext:
    """Shared GPU test context using the device primary context.

    The primary context is retained (not created) so multiple CUDAContext
    instances across different test modules all share the same underlying
    GPU context.  This eliminates the context stacking limit that caused
    cuCtxCreate failures when >1 test module ran in a single session.
    """

    def __init__(self):
        self.cuda = _CUDA
        self.ctx = ctypes.c_void_p()
        self.mod = ctypes.c_void_p()
        self.dev = ctypes.c_int()
        self.cuda.cuDeviceGet(ctypes.byref(self.dev), 0)
        rc = self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, self.dev)
        if rc != 0:
            raise RuntimeError(f"cuCtxCreate_v2 failed: {rc}")

    def load(self, cubin):
        if self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
        return self.cuda.cuModuleLoadData(ctypes.byref(self.mod), cubin) == 0

    def get_func(self, name):
        f = ctypes.c_void_p()
        assert self.cuda.cuModuleGetFunction(
            ctypes.byref(f), self.mod, name.encode()) == 0
        return f

    def alloc(self, n):
        p = ctypes.c_uint64()
        assert self.cuda.cuMemAlloc_v2(ctypes.byref(p), n) == 0
        return p.value

    def free(self, p):
        self.cuda.cuMemFree_v2(ctypes.c_uint64(p))

    def fill32(self, ptr, val, count):
        """Fill device memory with count copies of a 32-bit value."""
        data = val.to_bytes(4, 'little') * count
        self.copy_to(ptr, data)

    def copy_to(self, p, d):
        self.cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(p), d, len(d))

    def copy_from(self, p, n):
        b = (ctypes.c_uint8 * n)()
        self.cuda.cuMemcpyDtoH_v2(b, ctypes.c_uint64(p), n)
        return bytes(b)

    def sync(self):
        return self.cuda.cuCtxSynchronize()

    def launch(self, func, grid, block, args, smem=0):
        gx, gy, gz = grid if isinstance(grid, tuple) else (grid, 1, 1)
        bx, by, bz = block if isinstance(block, tuple) else (block, 1, 1)
        holders = []
        ptrs = []
        for a in args:
            if isinstance(a, int) and a > 0xFFFFFFFF:
                h = ctypes.c_uint64(a)
            else:
                h = ctypes.c_int32(a)
            holders.append(h)
            ptrs.append(ctypes.cast(ctypes.byref(h), ctypes.c_void_p))
        aa = (ctypes.c_void_p * len(ptrs))(*ptrs)
        return self.cuda.cuLaunchKernel(
            func, gx, gy, gz, bx, by, bz, smem, None, aa, None)

    def close(self):
        if self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
        if self.ctx.value:
            self.cuda.cuCtxSynchronize()
            self.cuda.cuCtxDestroy_v2(self.ctx)
            self.ctx = ctypes.c_void_p()
            # Pop any residual context from the stack
            dummy = ctypes.c_void_p()
            self.cuda.cuCtxPopCurrent_v2(ctypes.byref(dummy))


# ── session-scoped fixture ──────────────────────────────────────────
# One primary-context retain per pytest session.  All test files that
# request `cuda_ctx` share this single instance.

@pytest.fixture(scope="session")
def cuda_ctx(request):
    """Session-wide CUDA context shared by all GPU test modules.

    Using session scope with a SINGLE context avoids the stacking
    problem that occurs when multiple module-scoped contexts are
    created and destroyed across 9+ test files in one pytest run.
    """
    if _CUDA is None:
        pytest.skip("No CUDA GPU")
    c = CUDAContext()
    yield c
    c.close()
