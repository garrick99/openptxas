/**
 * gpu_copy_test.cu — Test LDG/STG by copying data through a kernel.
 *
 * Compile: nvcc -o gpu_copy_test gpu_copy_test.cu -lcuda
 * Usage:   gpu_copy_test <cubin_path> <kernel_name>
 */

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CHECK_CU(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *name = nullptr, *str = nullptr; \
        cuGetErrorName(err, &name); \
        cuGetErrorString(err, &str); \
        fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", \
                __FILE__, __LINE__, name ? name : "?", str ? str : "?"); \
        exit(1); \
    } \
} while(0)

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <cubin_path> <kernel_name>\n", argv[0]);
        return 1;
    }

    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    char devname[256];
    cuDeviceGetName(devname, sizeof(devname), dev);
    printf("Device: %s\n", devname);

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    CHECK_CU(cuCtxCreate(&ctx, &ctxParams, 0, dev));

    CUmodule mod;
    CUresult loadErr = cuModuleLoad(&mod, argv[1]);
    if (loadErr != CUDA_SUCCESS) {
        const char *name = nullptr, *str = nullptr;
        cuGetErrorName(loadErr, &name);
        cuGetErrorString(loadErr, &str);
        fprintf(stderr, "Failed to load cubin: %s (%s)\n", name, str);
        return 1;
    }
    printf("Loaded: %s\n", argv[1]);

    CUfunction func;
    CHECK_CU(cuModuleGetFunction(&func, mod, argv[2]));
    printf("Kernel: %s\n", argv[2]);

    // Simple copy test: write known pattern, read it back
    const int N = 1;
    uint64_t h_in = 0xDEADBEEFCAFEBABEULL;
    uint64_t h_out = 0;

    CUdeviceptr d_in, d_out;
    CHECK_CU(cuMemAlloc(&d_in, sizeof(uint64_t)));
    CHECK_CU(cuMemAlloc(&d_out, sizeof(uint64_t)));
    CHECK_CU(cuMemcpyHtoD(d_in, &h_in, sizeof(uint64_t)));

    // Zero out output
    uint64_t zero = 0;
    CHECK_CU(cuMemcpyHtoD(d_out, &zero, sizeof(uint64_t)));

    // Launch: kernel(out_ptr, in_ptr)
    void *args[] = { &d_out, &d_in };
    printf("Launching with d_out=0x%llx, d_in=0x%llx\n",
           (unsigned long long)d_out, (unsigned long long)d_in);
    CHECK_CU(cuLaunchKernel(func,
        1, 1, 1,    // grid
        1, 1, 1,    // block
        0, 0,       // shared mem, stream
        args, nullptr));
    CHECK_CU(cuCtxSynchronize());

    CHECK_CU(cuMemcpyDtoH(&h_out, d_out, sizeof(uint64_t)));

    printf("Input:  0x%016llx\n", (unsigned long long)h_in);
    printf("Output: 0x%016llx\n", (unsigned long long)h_out);

    if (h_out == h_in) {
        printf("\n*** PASS — LDG+STG copy works! ***\n");
    } else {
        printf("\n*** FAIL — expected 0x%016llx, got 0x%016llx ***\n",
               (unsigned long long)h_in, (unsigned long long)h_out);
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return (h_out == h_in) ? 0 : 1;
}
