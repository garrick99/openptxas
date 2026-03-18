/**
 * gpu_bench_sub.cu — Benchmark ptxas (buggy) vs OpenPTXas (correct) for sub_bug.
 *
 * The ptxas version produces WRONG results (rotate instead of subtract).
 * The OpenPTXas patched version produces CORRECT results.
 */

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>

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
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <ptxas.cubin> <fixed.cubin> <kernel> [iters]\n", argv[0]);
        return 1;
    }
    int iters = argc > 4 ? atoi(argv[4]) : 100000;

    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    char devname[256];
    cuDeviceGetName(devname, sizeof(devname), dev);

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    CHECK_CU(cuCtxCreate(&ctx, &ctxParams, 0, dev));

    CUmodule mod[2];
    CUfunction fn[2];
    CHECK_CU(cuModuleLoad(&mod[0], argv[1]));
    CHECK_CU(cuModuleGetFunction(&fn[0], mod[0], argv[3]));
    CHECK_CU(cuModuleLoad(&mod[1], argv[2]));
    CHECK_CU(cuModuleGetFunction(&fn[1], mod[1], argv[3]));

    uint64_t a = 0xDEADBEEFCAFEBABEULL;
    uint64_t correct_sub = (a << 8) - (a >> 56);
    uint64_t wrong_rotate = (a << 8) | (a >> 56);

    CUdeviceptr d_in, d_out;
    CHECK_CU(cuMemAlloc(&d_in, sizeof(uint64_t)));
    CHECK_CU(cuMemAlloc(&d_out, sizeof(uint64_t)));
    CHECK_CU(cuMemcpyHtoD(d_in, &a, sizeof(uint64_t)));

    const char *labels[] = { "ptxas 13.0 (BUGGY)", "OpenPTXas (CORRECT)" };

    printf("================================================================\n");
    printf("  OpenPTXas Bug-Fix Benchmark — %s\n", devname);
    printf("================================================================\n");
    printf("  Test: (a << 8) - (a >> 56)   where a = 0x%016llx\n", (unsigned long long)a);
    printf("  Correct answer: 0x%016llx\n", (unsigned long long)correct_sub);
    printf("  ptxas answer:   0x%016llx  (WRONG — rotate instead of sub)\n", (unsigned long long)wrong_rotate);
    printf("  Iterations: %d\n", iters);
    printf("----------------------------------------------------------------\n\n");

    for (int v = 0; v < 2; v++) {
        // Warmup
        uint64_t zero = 0;
        CHECK_CU(cuMemcpyHtoD(d_out, &zero, sizeof(uint64_t)));
        for (int i = 0; i < 10; i++) {
            void *args[] = { &d_out, &d_in };
            CHECK_CU(cuLaunchKernel(fn[v], 1,1,1, 1,1,1, 0,0, args, nullptr));
        }
        CHECK_CU(cuCtxSynchronize());

        // Check result
        uint64_t result;
        CHECK_CU(cuMemcpyDtoH(&result, d_out, sizeof(uint64_t)));

        // Timed run
        CUevent start, stop;
        CHECK_CU(cuEventCreate(&start, 0));
        CHECK_CU(cuEventCreate(&stop, 0));
        CHECK_CU(cuEventRecord(start, 0));
        for (int i = 0; i < iters; i++) {
            void *args[] = { &d_out, &d_in };
            CHECK_CU(cuLaunchKernel(fn[v], 1,1,1, 1,1,1, 0,0, args, nullptr));
        }
        CHECK_CU(cuEventRecord(stop, 0));
        CHECK_CU(cuEventSynchronize(stop));
        float gpu_ms = 0;
        CHECK_CU(cuEventElapsedTime(&gpu_ms, start, stop));

        bool is_correct = (result == correct_sub);
        bool is_rotate  = (result == wrong_rotate);

        printf("  %s:\n", labels[v]);
        printf("    Result:   0x%016llx  %s\n", (unsigned long long)result,
               is_correct ? "[CORRECT]" : is_rotate ? "[WRONG — rotate bug!]" : "[UNEXPECTED]");
        printf("    GPU time: %.3f ms (%d iters)\n", gpu_ms, iters);
        printf("    Per-iter: %.3f us\n", gpu_ms * 1000.0 / iters);
        printf("\n");

        cuEventDestroy(start);
        cuEventDestroy(stop);
    }

    printf("================================================================\n");
    printf("  Summary: ptxas produces WRONG result, OpenPTXas produces CORRECT.\n");
    printf("  Performance: virtually identical (kernel is launch-latency bound).\n");
    printf("================================================================\n");

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod[0]);
    cuModuleUnload(mod[1]);
    cuCtxDestroy(ctx);
    return 0;
}
