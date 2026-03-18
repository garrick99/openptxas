/**
 * gpu_bug_demo.cu — Demonstrate the ptxas rotate-miscompilation bug on GPU.
 *
 * Tests: (a << 8) - (a >> 56)
 * ptxas INCORRECTLY compiles this as rotate_left(a, 8).
 *
 * Compile: nvcc -o gpu_bug_demo gpu_bug_demo.cu -lcuda -allow-unsupported-compiler
 * Usage:   gpu_bug_demo <cubin_path> <kernel_name>
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
    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    CHECK_CU(cuCtxCreate(&ctx, &ctxParams, 0, dev));

    CUmodule mod;
    CHECK_CU(cuModuleLoad(&mod, argv[1]));
    CUfunction func;
    CHECK_CU(cuModuleGetFunction(&func, mod, argv[2]));

    // Test with a = 0xDEADBEEFCAFEBABE
    uint64_t a = 0xDEADBEEFCAFEBABEULL;

    // Correct result: (a << 8) - (a >> 56)
    uint64_t shl_result = a << 8;
    uint64_t shr_result = a >> 56;
    uint64_t correct_sub = shl_result - shr_result;
    uint64_t wrong_rotate = (a << 8) | (a >> 56);  // rotate_left(a, 8)

    printf("Test: (a << 8) - (a >> 56)\n");
    printf("  a             = 0x%016llx\n", (unsigned long long)a);
    printf("  a << 8        = 0x%016llx\n", (unsigned long long)shl_result);
    printf("  a >> 56       = 0x%016llx\n", (unsigned long long)shr_result);
    printf("  CORRECT (sub) = 0x%016llx\n", (unsigned long long)correct_sub);
    printf("  WRONG (rotl)  = 0x%016llx\n", (unsigned long long)wrong_rotate);
    printf("\n");

    // Allocate and run
    CUdeviceptr d_in, d_out;
    CHECK_CU(cuMemAlloc(&d_in, sizeof(uint64_t)));
    CHECK_CU(cuMemAlloc(&d_out, sizeof(uint64_t)));
    CHECK_CU(cuMemcpyHtoD(d_in, &a, sizeof(uint64_t)));
    uint64_t zero = 0;
    CHECK_CU(cuMemcpyHtoD(d_out, &zero, sizeof(uint64_t)));

    void *args[] = { &d_out, &d_in };
    CHECK_CU(cuLaunchKernel(func, 1,1,1, 1,1,1, 0,0, args, nullptr));
    CHECK_CU(cuCtxSynchronize());

    uint64_t result;
    CHECK_CU(cuMemcpyDtoH(&result, d_out, sizeof(uint64_t)));

    printf("GPU result      = 0x%016llx\n", (unsigned long long)result);
    printf("\n");

    if (result == correct_sub) {
        printf("*** CORRECT — GPU computed subtraction properly ***\n");
    } else if (result == wrong_rotate) {
        printf("*** BUG CONFIRMED — GPU computed rotate instead of subtract! ***\n");
        printf("  ptxas miscompiled (a<<8) - (a>>56) as rotate_left(a,8)\n");
        printf("  Diff from correct: 0x%016llx\n",
               (unsigned long long)(result ^ correct_sub));
    } else {
        printf("*** UNEXPECTED result ***\n");
        printf("  Neither correct sub nor wrong rotate\n");
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return (result == correct_sub) ? 0 : 1;
}
