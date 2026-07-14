#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

bool ggml_cuda_mul_mat_vec_q_grouped(ggml_backend_cuda_context & ctx,
    const ggml_tensor * dst0, const ggml_tensor * dst1, const ggml_tensor * dst2);

bool ggml_cuda_mul_mat_vec_q_gdn(ggml_backend_cuda_context & ctx,
    const ggml_tensor * alpha, const ggml_tensor * beta,
    const ggml_tensor * alpha_bias, const ggml_tensor * alpha_scale,
    ggml_tensor * dst_gate, ggml_tensor * dst_beta);

bool ggml_cuda_mul_mat_vec_q_gdn_grouped(ggml_backend_cuda_context & ctx,
    const ggml_tensor * qkv, const ggml_tensor * alpha, const ggml_tensor * beta,
    const ggml_tensor * alpha_bias, const ggml_tensor * alpha_scale,
    ggml_tensor * dst_gate, ggml_tensor * dst_beta);

bool ggml_cuda_mul_mat_vec_q_gdn_conv(ggml_backend_cuda_context & ctx,
    const ggml_tensor * qkv, const ggml_tensor * alpha, const ggml_tensor * beta,
    const ggml_tensor * alpha_bias, const ggml_tensor * alpha_scale,
    const ggml_tensor * conv_states, const ggml_tensor * conv_kernel,
    ggml_tensor * conv_scratch, ggml_tensor * conv_state_update,
    ggml_tensor * q_norm, ggml_tensor * k_norm, ggml_tensor * v_conv,
    ggml_tensor * dst_gate, ggml_tensor * dst_beta);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
