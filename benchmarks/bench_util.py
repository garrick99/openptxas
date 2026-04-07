"""Shared utilities for OpenPTXas vs NVIDIA ptxas benchmarks."""
import ctypes
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sass.pipeline import compile_ptx_source


def _get_cuda():
    try:
        cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        if cuda.cuInit(0) != 0:
            return None
        return cuda
    except Exception:
        return None


_CUDA = _get_cuda()
if _CUDA is None:
    print("ERROR: nvcuda.dll not found")
    sys.exit(1)


class CUDAContext:
    def __init__(self):
        self.cuda = _CUDA
        self.ctx = ctypes.c_void_p()
        self.mod = ctypes.c_void_p()
        dev = ctypes.c_int()
        assert self.cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
        assert self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev) == 0

    def load(self, cubin_bytes):
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
        err = self.cuda.cuModuleLoadData(ctypes.byref(self.mod), cubin_bytes)
        return err == 0

    def get_func(self, name):
        func = ctypes.c_void_p()
        err = self.cuda.cuModuleGetFunction(ctypes.byref(func), self.mod, name.encode())
        assert err == 0, f"cuModuleGetFunction({name}) failed: {err}"
        return func

    def alloc(self, nbytes):
        ptr = ctypes.c_uint64()
        assert self.cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes) == 0
        return ptr.value

    def memset_d8(self, dev_ptr, val, nbytes):
        self.cuda.cuMemsetD8_v2(ctypes.c_uint64(dev_ptr), ctypes.c_ubyte(val), nbytes)

    def copy_to(self, dev_ptr, host_data):
        assert self.cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(dev_ptr), host_data, len(host_data)) == 0

    def copy_from(self, dev_ptr, nbytes):
        buf = (ctypes.c_uint8 * nbytes)()
        assert self.cuda.cuMemcpyDtoH_v2(buf, ctypes.c_uint64(dev_ptr), nbytes) == 0
        return bytes(buf)

    def free(self, ptr):
        self.cuda.cuMemFree_v2(ctypes.c_uint64(ptr))

    def sync(self):
        return self.cuda.cuCtxSynchronize()

    def close(self):
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
        if self.ctx and self.ctx.value:
            self.cuda.cuCtxSynchronize()
            self.cuda.cuCtxDestroy_v2(self.ctx)

    def event_create(self):
        evt = ctypes.c_void_p()
        assert self.cuda.cuEventCreate(ctypes.byref(evt), 0) == 0
        return evt

    def event_record(self, evt):
        self.cuda.cuEventRecord(evt, None)

    def event_elapsed_ms(self, start, stop):
        ms = ctypes.c_float()
        self.cuda.cuEventSynchronize(stop)
        self.cuda.cuEventElapsedTime(ctypes.byref(ms), start, stop)
        return ms.value

    def launch(self, func, grid, block, args, smem_bytes=0):
        gx, gy, gz = (grid if isinstance(grid, tuple) else (grid, 1, 1))
        bx, by, bz = (block if isinstance(block, tuple) else (block, 1, 1))
        return self.cuda.cuLaunchKernel(
            func, gx, gy, gz, bx, by, bz, smem_bytes, None, args, None)


def compile_openptxas(ptx):
    t0 = time.perf_counter()
    result = compile_ptx_source(ptx)
    cubin = next(iter(result.values())) if isinstance(result, dict) else result
    return cubin, time.perf_counter() - t0


def compile_ptxas(ptx):
    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "k.ptx"
        cf = Path(tmp) / "k.cubin"
        pf.write_text(ptx)
        t0 = time.perf_counter()
        r = subprocess.run(
            ["ptxas", "-arch=sm_120", "-o", str(cf), str(pf)],
            capture_output=True, text=True
        )
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            raise RuntimeError(f"ptxas: {r.stderr}")
        return cf.read_bytes(), dt


def make_args(*values):
    """Build a kernel arg array from a list of ctypes values. Hold refs to prevent GC."""
    ptrs = (ctypes.c_void_p * len(values))(
        *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values]
    )
    return ptrs, values  # return holder to keep values alive


def print_header(title):
    print("=" * 64)
    print(f"  {title}")
    print("  GPU: RTX 5090 (SM_120)")
    print("=" * 64)
    print()


def analyze_cubin(cubin, kernel_name=None):
    """Extract instruction count, register count, and text size from a cubin."""
    import struct as s
    e_shoff = s.unpack_from('<Q', cubin, 40)[0]
    e_shnum = s.unpack_from('<H', cubin, 60)[0]
    e_shstrndx = s.unpack_from('<H', cubin, 62)[0]
    shstrtab_off = s.unpack_from('<Q', cubin, e_shoff + e_shstrndx*64 + 24)[0]
    shstrtab_sz = s.unpack_from('<Q', cubin, e_shoff + e_shstrndx*64 + 32)[0]
    shstrtab = cubin[shstrtab_off:shstrtab_off+shstrtab_sz]
    text_size = 0
    for i in range(e_shnum):
        sh = e_shoff + i*64
        n_off = s.unpack_from('<I', cubin, sh)[0]
        nm = shstrtab[n_off:shstrtab.index(0, n_off)].decode()
        sec_off = s.unpack_from('<Q', cubin, sh+24)[0]
        sec_sz = s.unpack_from('<Q', cubin, sh+32)[0]
        if '.text.' in nm and 'capmerc' not in nm:
            text_data = cubin[sec_off:sec_off+sec_sz]
            text_size = sec_sz
            # Count real instructions (non-NOP, non-padding)
            n_instrs = sec_sz // 16
            n_nops = sum(1 for off in range(0, sec_sz, 16)
                         if (s.unpack_from('<Q', text_data, off)[0] & 0xFFF) == 0x918)
            break
    # Extract register count from nv.info EIATTR
    num_gprs = 0
    for i in range(e_shnum):
        sh = e_shoff + i*64
        n_off = s.unpack_from('<I', cubin, sh)[0]
        nm = shstrtab[n_off:shstrtab.index(0, n_off)].decode()
        sec_off = s.unpack_from('<Q', cubin, sh+24)[0]
        sec_sz = s.unpack_from('<Q', cubin, sh+32)[0]
        if 'capmerc' in nm:
            cm = cubin[sec_off:sec_off+sec_sz]
            if len(cm) > 8:
                num_gprs = cm[8]
            break
    return {
        'text_size': text_size,
        'n_instrs': n_instrs,
        'n_nops': n_nops,
        'n_real': n_instrs - n_nops,
        'num_gprs': num_gprs,
    }


def print_results(t_ours_ms, t_nvid_ms, c_ours, c_nvid,
                  med_ours_us, med_nvid_us, perf_label, perf_ours, perf_nvid,
                  correct, perf_fmt="{:.1f} GB/s"):
    a_ours = analyze_cubin(c_ours)
    a_nvid = analyze_cubin(c_nvid)

    print("=" * 64)
    print("  RESULTS")
    print("=" * 64)
    fmt = "  {:<22} {:>14}  {:>14}  {:>10}"
    print(fmt.format("Metric", "OpenPTXas", "NVIDIA ptxas", "Delta"))
    print("-" * 64)
    print(fmt.format("Instructions (real)",
                     str(a_ours['n_real']),
                     str(a_nvid['n_real']),
                     f"{a_ours['n_real'] - a_nvid['n_real']:+d}"))
    print(fmt.format("Instructions (NOPs)",
                     str(a_ours['n_nops']),
                     str(a_nvid['n_nops']),
                     f"{a_ours['n_nops'] - a_nvid['n_nops']:+d}"))
    print(fmt.format("Text size",
                     f"{a_ours['text_size']} B",
                     f"{a_nvid['text_size']} B",
                     f"{a_ours['text_size'] - a_nvid['text_size']:+d}"))
    print(fmt.format("Registers (GPR)",
                     str(a_ours['num_gprs']),
                     str(a_nvid['num_gprs']),
                     f"{a_ours['num_gprs'] - a_nvid['num_gprs']:+d}"))
    print(fmt.format("Compile time",
                     f"{t_ours_ms:.1f} ms",
                     f"{t_nvid_ms:.1f} ms",
                     f"{t_nvid_ms/max(t_ours_ms,0.001):.1f}x"))
    print(fmt.format("GPU exec (median)",
                     f"{med_ours_us:.1f} us",
                     f"{med_nvid_us:.1f} us",
                     f"{med_nvid_us/max(med_ours_us,0.001):.2f}x"))
    print(fmt.format(perf_label,
                     perf_fmt.format(perf_ours),
                     perf_fmt.format(perf_nvid),
                     f"{perf_ours/max(perf_nvid,0.001):.2f}x"))
    print(fmt.format("Correctness",
                     "PASS" if correct else "FAIL",
                     "(baseline)",
                     ""))
    print("=" * 64)
