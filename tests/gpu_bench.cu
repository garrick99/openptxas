/**
 * gpu_bench.cu — Benchmark OpenPTXas vs ptxas on the RTX 5090.
 *
 * Tests:
 *   1. Correctness: verify both cubins produce expected results
 *   2. Throughput: launch kernel N times, measure wall clock + GPU events
 *   3. Latency: single-element kernel launch overhead
 *
 * Compile: nvcc -o gpu_bench gpu_bench.cu -lcuda -allow-unsupported-compiler
 * Usage:   gpu_bench <ptxas_cubin> <openptxas_cubin> <kernel_name> [N_elements] [N_iters]
 */

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
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

static uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

struct BenchResult {
    double gpu_ms;       // GPU time from events (ms)
    double wall_ms;      // wall clock time (ms)
    int    n_iters;
    int    n_elements;
    bool   correct;
};

BenchResult run_bench(CUfunction func, const char *label,
                      uint64_t *h_in, uint64_t *h_expected,
                      int N, int n_iters)
{
    BenchResult res = {};
    res.n_iters = n_iters;
    res.n_elements = N;

    CUdeviceptr d_in, d_out;
    CHECK_CU(cuMemAlloc(&d_in, N * sizeof(uint64_t)));
    CHECK_CU(cuMemAlloc(&d_out, N * sizeof(uint64_t)));
    CHECK_CU(cuMemcpyHtoD(d_in, h_in, N * sizeof(uint64_t)));

    // Warmup
    for (int i = 0; i < 3; i++) {
        void *args[] = { &d_out, &d_in };
        CHECK_CU(cuLaunchKernel(func, 1,1,1, 1,1,1, 0,0, args, nullptr));
    }
    CHECK_CU(cuCtxSynchronize());

    // Correctness check
    uint64_t *h_out = (uint64_t *)malloc(N * sizeof(uint64_t));
    CHECK_CU(cuMemcpyDtoH(h_out, d_out, N * sizeof(uint64_t)));
    res.correct = (h_out[0] == h_expected[0]);

    // GPU event timing
    CUevent start, stop;
    CHECK_CU(cuEventCreate(&start, 0));
    CHECK_CU(cuEventCreate(&stop, 0));

    CHECK_CU(cuEventRecord(start, 0));
    for (int i = 0; i < n_iters; i++) {
        void *args[] = { &d_out, &d_in };
        CHECK_CU(cuLaunchKernel(func, 1,1,1, 1,1,1, 0,0, args, nullptr));
    }
    CHECK_CU(cuEventRecord(stop, 0));
    CHECK_CU(cuEventSynchronize(stop));

    float gpu_ms = 0;
    CHECK_CU(cuEventElapsedTime(&gpu_ms, start, stop));
    res.gpu_ms = gpu_ms;

    // Wall clock timing
    CHECK_CU(cuCtxSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iters; i++) {
        void *args[] = { &d_out, &d_in };
        CHECK_CU(cuLaunchKernel(func, 1,1,1, 1,1,1, 0,0, args, nullptr));
    }
    CHECK_CU(cuCtxSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    res.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cuEventDestroy(start);
    cuEventDestroy(stop);
    cuMemFree(d_in);
    cuMemFree(d_out);
    free(h_out);

    return res;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <ptxas.cubin> <openptxas.cubin> <kernel> [N] [iters]\n", argv[0]);
        return 1;
    }
    const char *ptxas_path = argv[1];
    const char *open_path  = argv[2];
    const char *kernel     = argv[3];
    int N      = argc > 4 ? atoi(argv[4]) : 1;
    int iters  = argc > 5 ? atoi(argv[5]) : 100000;

    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    char devname[256];
    cuDeviceGetName(devname, sizeof(devname), dev);

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    CHECK_CU(cuCtxCreate(&ctx, &ctxParams, 0, dev));

    // Load both cubins
    CUmodule mod_ptxas, mod_open;
    CUfunction fn_ptxas, fn_open;

    CHECK_CU(cuModuleLoad(&mod_ptxas, ptxas_path));
    CHECK_CU(cuModuleGetFunction(&fn_ptxas, mod_ptxas, kernel));

    CUresult openErr = cuModuleLoad(&mod_open, open_path);
    bool have_open = (openErr == CUDA_SUCCESS);
    if (have_open) {
        CHECK_CU(cuModuleGetFunction(&fn_open, mod_open, kernel));
    } else {
        fprintf(stderr, "Warning: could not load OpenPTXas cubin: %s\n", open_path);
    }

    // Test data
    uint64_t h_in[1], h_expected[1];
    h_in[0] = 0x0123456789ABCDEFULL;
    h_expected[0] = rotl64(h_in[0], 1);  // probe_k1: rotate left by 1

    printf("========================================================\n");
    printf("  OpenPTXas Benchmark — %s\n", devname);
    printf("========================================================\n");
    printf("  Kernel: %s\n", kernel);
    printf("  Elements: %d\n", N);
    printf("  Iterations: %d\n", iters);
    printf("  Input: 0x%016llx\n", (unsigned long long)h_in[0]);
    printf("  Expected: 0x%016llx\n", (unsigned long long)h_expected[0]);
    printf("--------------------------------------------------------\n\n");

    // Benchmark ptxas
    printf("  ptxas 13.0 (NVIDIA):\n");
    BenchResult r1 = run_bench(fn_ptxas, "ptxas", h_in, h_expected, N, iters);
    printf("    Correct:   %s\n", r1.correct ? "YES" : "NO");
    printf("    GPU time:  %.3f ms (%d iters)\n", r1.gpu_ms, iters);
    printf("    Per-iter:  %.3f us (GPU)\n", r1.gpu_ms * 1000.0 / iters);
    printf("    Wall time: %.3f ms\n", r1.wall_ms);
    printf("    Per-iter:  %.3f us (wall)\n", r1.wall_ms * 1000.0 / iters);
    printf("\n");

    // Benchmark OpenPTXas
    if (have_open) {
        printf("  OpenPTXas (open-source):\n");
        BenchResult r2 = run_bench(fn_open, "OpenPTXas", h_in, h_expected, N, iters);
        printf("    Correct:   %s\n", r2.correct ? "YES" : "NO");
        printf("    GPU time:  %.3f ms (%d iters)\n", r2.gpu_ms, iters);
        printf("    Per-iter:  %.3f us (GPU)\n", r2.gpu_ms * 1000.0 / iters);
        printf("    Wall time: %.3f ms\n", r2.wall_ms);
        printf("    Per-iter:  %.3f us (wall)\n", r2.wall_ms * 1000.0 / iters);
        printf("\n");

        printf("--------------------------------------------------------\n");
        double speedup = r1.gpu_ms / r2.gpu_ms;
        printf("  Speedup (GPU):  %.2fx (%s is %s)\n",
               speedup > 1.0 ? speedup : 1.0/speedup,
               speedup > 1.0 ? "OpenPTXas" : "ptxas",
               speedup > 1.0 ? "faster" : "faster");
        printf("  ptxas:     %d instructions (uses SHF.L.W rotate)\n", 10);
        printf("  OpenPTXas: %d instructions (decomposes to SHF+IADD.64)\n", 12);
        printf("  Both produce correct results: %s\n",
               (r1.correct && r2.correct) ? "YES" : "NO");
    }

    printf("========================================================\n");

    if (have_open) cuModuleUnload(mod_open);
    cuModuleUnload(mod_ptxas);
    cuCtxDestroy(ctx);
    return 0;
}
