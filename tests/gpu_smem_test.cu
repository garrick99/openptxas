/**
 * gpu_smem_test.cu — Verify shared memory store-load cycle.
 *
 * The smem_probe kernel: load from global -> store to shared -> barrier ->
 * load from shared -> store to global. If smem works correctly, output == input.
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
        fprintf(stderr, "Usage: %s <cubin> <kernel_name>\n", argv[0]);
        return 1;
    }

    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CUctxCreateParams p = {};
    CHECK_CU(cuCtxCreate(&ctx, &p, 0, dev));

    CUmodule mod;
    CHECK_CU(cuModuleLoad(&mod, argv[1]));
    CUfunction func;
    CHECK_CU(cuModuleGetFunction(&func, mod, argv[2]));

    // Test: store 0xDEADBEEF to shared, load back, verify
    uint32_t input = 0xDEADBEEFu;
    uint32_t output = 0;

    CUdeviceptr d_in, d_out;
    CHECK_CU(cuMemAlloc(&d_in, sizeof(uint32_t)));
    CHECK_CU(cuMemAlloc(&d_out, sizeof(uint32_t)));
    CHECK_CU(cuMemcpyHtoD(d_in, &input, sizeof(uint32_t)));
    CHECK_CU(cuMemcpyHtoD(d_out, &output, sizeof(uint32_t)));

    void *args[] = { &d_out, &d_in };

    // smem_probe uses shared memory — need shared mem size
    // The kernel statically allocates via .nv.shared section
    CHECK_CU(cuLaunchKernel(func, 1,1,1, 1,1,1, 0, 0, args, nullptr));
    CHECK_CU(cuCtxSynchronize());

    CHECK_CU(cuMemcpyDtoH(&output, d_out, sizeof(uint32_t)));

    printf("Shared Memory Store-Load Test\n");
    printf("  Input:  0x%08x\n", input);
    printf("  Output: 0x%08x\n", output);

    if (output == input) {
        printf("  *** PASS — shared memory store-load cycle works! ***\n");
    } else {
        printf("  *** FAIL — output != input ***\n");
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return (output == input) ? 0 : 1;
}
