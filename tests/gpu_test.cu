/**
 * gpu_test.cu — Load an OpenPTXas-generated cubin and run it on the GPU.
 *
 * Compile: nvcc -o gpu_test gpu_test.cu -lcuda
 * Usage:   gpu_test <cubin_path> <kernel_name>
 *
 * Tests the probe_k1 kernel (rotate-left by 1):
 *   output[i] = rotate_left(input[i], 1)
 *
 * Compares against CPU-computed expected results.
 */

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

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

// CPU reference: rotate-left by K
static uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <cubin_path> <kernel_name>\n", argv[0]);
        return 1;
    }
    const char *cubin_path = argv[1];
    const char *kernel_name = argv[2];

    // Initialize CUDA driver API
    CHECK_CU(cuInit(0));

    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));

    char devname[256];
    cuDeviceGetName(devname, sizeof(devname), dev);
    printf("Device: %s\n", devname);

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    CHECK_CU(cuCtxCreate(&ctx, &ctxParams, 0, dev));

    // Load cubin
    CUmodule mod;
    CUresult loadErr = cuModuleLoad(&mod, cubin_path);
    if (loadErr != CUDA_SUCCESS) {
        const char *name = nullptr, *str = nullptr;
        cuGetErrorName(loadErr, &name);
        cuGetErrorString(loadErr, &str);
        fprintf(stderr, "Failed to load cubin '%s': %s (%s)\n",
                cubin_path, name ? name : "?", str ? str : "?");
        return 1;
    }
    printf("Loaded cubin: %s\n", cubin_path);

    CUfunction func;
    CHECK_CU(cuModuleGetFunction(&func, mod, kernel_name));
    printf("Found kernel: %s\n", kernel_name);

    // Test data
    const int N = 16;
    uint64_t h_in[N], h_out[N], h_expected[N];

    for (int i = 0; i < N; i++) {
        h_in[i] = (uint64_t)(i + 1) * 0x0123456789ABCDEFULL;
        h_out[i] = 0xDEADDEADDEADDEADULL;
        h_expected[i] = rotl64(h_in[i], 1);  // probe_k1 rotates by K=1
    }

    // Allocate device memory
    CUdeviceptr d_in, d_out;
    CHECK_CU(cuMemAlloc(&d_in, N * sizeof(uint64_t)));
    CHECK_CU(cuMemAlloc(&d_out, N * sizeof(uint64_t)));
    CHECK_CU(cuMemcpyHtoD(d_in, h_in, N * sizeof(uint64_t)));
    CHECK_CU(cuMemcpyHtoD(d_out, h_out, N * sizeof(uint64_t)));

    // Launch kernel: probe_k1(out_ptr, in_ptr)
    // The kernel processes a single element (index 0) — it's not a loop kernel
    void *args[] = { &d_out, &d_in };
    CHECK_CU(cuLaunchKernel(func,
        1, 1, 1,    // grid (1 block)
        1, 1, 1,    // block (1 thread)
        0, 0,       // shared mem, stream
        args, nullptr));
    CHECK_CU(cuCtxSynchronize());

    // Read back
    CHECK_CU(cuMemcpyDtoH(h_out, d_out, N * sizeof(uint64_t)));

    // Verify first element
    printf("\nInput:    0x%016llx\n", (unsigned long long)h_in[0]);
    printf("Output:   0x%016llx\n", (unsigned long long)h_out[0]);
    printf("Expected: 0x%016llx\n", (unsigned long long)h_expected[0]);

    if (h_out[0] == h_expected[0]) {
        printf("\n*** PASS — OpenPTXas cubin produced correct result! ***\n");
    } else {
        printf("\n*** FAIL — result mismatch ***\n");
        printf("  Diff: output ^ expected = 0x%016llx\n",
               (unsigned long long)(h_out[0] ^ h_expected[0]));
    }

    // Cleanup
    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    return (h_out[0] == h_expected[0]) ? 0 : 1;
}
