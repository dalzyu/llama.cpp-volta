#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        std::exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    const cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int) status); \
        std::exit(1); \
    } \
} while (0)

template <typename F>
static float time_ms(F && fn, int repetitions) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repetitions; ++i) {
        fn();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed / repetitions;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s M N K\n", argv[0]);
        return 1;
    }

    const int m = std::atoi(argv[1]);
    const int n = std::atoi(argv[2]);
    const int k = std::atoi(argv[3]);
    constexpr int repetitions = 50;
    constexpr size_t workspace_size = 64ull * 1024 * 1024;

    half * a;
    half * b;
    float * c;
    void * workspace;
    CUDA_CHECK(cudaMalloc(&a, sizeof(half) * (size_t) k * m));
    CUDA_CHECK(cudaMalloc(&b, sizeof(half) * (size_t) k * n));
    CUDA_CHECK(cudaMalloc(&c, sizeof(float) * (size_t) m * n));
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    CUDA_CHECK(cudaMemset(a, 0, sizeof(half) * (size_t) k * m));
    CUDA_CHECK(cudaMemset(b, 0, sizeof(half) * (size_t) k * n));
    CUDA_CHECK(cudaMemset(c, 0, sizeof(float) * (size_t) m * n));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));

    const float legacy_ms = time_ms([&]() {
        CUBLAS_CHECK(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k, &alpha,
            a, CUDA_R_16F, k,
            b, CUDA_R_16F, k,
            &beta, c, CUDA_R_32F, m,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }, repetitions);

    cublasLtHandle_t lt;
    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t a_layout;
    cublasLtMatrixLayout_t b_layout;
    cublasLtMatrixLayout_t c_layout;
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtCreate(&lt));
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16F, k, m, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16F, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32F, m, n, m));
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    std::vector<cublasLtMatmulHeuristicResult_t> results(64);
    int result_count = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, operation,
        a_layout, b_layout, c_layout, c_layout, preference,
        (int) results.size(), results.data(), &result_count));

    std::printf("M\tN\tK\tbackend\talgo\ttile\tsplit_k\tworkspace\twaves\tms\n");
    std::printf("%d\t%d\t%d\tcublas\t99\t0\t1\t0\t0\t%.6f\n", m, n, k, legacy_ms);
    for (int i = 0; i < result_count; ++i) {
        if (results[i].state != CUBLAS_STATUS_SUCCESS) {
            continue;
        }

        int32_t id = -1;
        uint32_t tile = 0;
        int32_t split_k = 1;
        size_t written = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
            &results[i].algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written));
        CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
            &results[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &written));
        CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
            &results[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k, sizeof(split_k), &written));

        const float ms = time_ms([&]() {
            CUBLAS_CHECK(cublasLtMatmul(lt, operation,
                &alpha, a, a_layout, b, b_layout,
                &beta, c, c_layout, c, c_layout,
                &results[i].algo, workspace, workspace_size, nullptr));
        }, repetitions);
        std::printf("%d\t%d\t%d\tcublaslt\t%d\t%u\t%d\t%zu\t%.3f\t%.6f\n",
            m, n, k, id, tile, split_k, results[i].workspaceSize, results[i].wavesCount, ms);
    }

    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_layout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_layout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_layout));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation));
    CUBLAS_CHECK(cublasLtDestroy(lt));
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));
    return 0;
}
