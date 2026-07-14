#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_VOLTA,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
    return MMVQ_PARAMETERS_VOLTA;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA) {
        return MMVQ_PARAMETERS_VOLTA;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(
        ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_VOLTA) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                    return small_k ? 4 : 2;
                case GGML_TYPE_Q8_0:
                    return small_k ? 4 : 2;
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

static constexpr __host__ __device__ int calc_min_blocks_per_sm(
        ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool has_fusion, bool small_k = false) {
    if (table_id == MMVQ_PARAMETERS_VOLTA && type == GGML_TYPE_Q2_K && ncols_dst == 1 && has_fusion && !small_k) {
        return 20;
    }
    if (table_id == MMVQ_PARAMETERS_VOLTA && type == GGML_TYPE_Q3_K && ncols_dst == 1 && has_fusion && !small_k) {
        return 14;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(
        ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1, bool has_fusion = false) {
    if (table_id == MMVQ_PARAMETERS_VOLTA && type == GGML_TYPE_Q8_0 && ncols_dst == 1 && !small_k && !has_fusion) {
        return 2;
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_VOLTA ||
        table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k)*ggml_cuda_get_physical_warp_size(),
                  calc_min_blocks_per_sm(type, ncols_dst, get_device_table_id(), has_fusion, small_k))
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps, has_fusion);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_warp_rows(
        const void * vx, const block_q8_1 * y, float * dst,
        const uint32_t ncols_x, const uint32_t stride_row_x) {
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_warp_iter = vdr*warp_size/qi;

    ggml_cuda_pdl_sync();

    const int row = 2*blockIdx.x + threadIdx.y;
    const int blocks_per_row_x = ncols_x/qk;
    const block_q8_0 * x = (const block_q8_0 *) vx + row*stride_row_x;
    float tmp = 0.0f;

    for (int kbx = threadIdx.x/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_warp_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(threadIdx.x % (qi/vdr));
        tmp += vec_dot_q8_0_q8_1(x, &y[kby], kbx, kqs);
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_warp_rows_silu_mul(
        const void * vx, const block_q8_1 * y, const float * mul, float * dst,
        const uint32_t ncols_x, const uint32_t stride_row_x) {
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_warp_iter = vdr*warp_size/qi;

    ggml_cuda_pdl_sync();

    const int row = 2*blockIdx.x + threadIdx.y;
    const int blocks_per_row_x = ncols_x/qk;
    const block_q8_0 * x = (const block_q8_0 *) vx + row*stride_row_x;
    float tmp = 0.0f;

    for (int kbx = threadIdx.x/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_warp_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(threadIdx.x % (qi/vdr));
        tmp += vec_dot_q8_0_q8_1(x, &y[kby], kbx, kqs);
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if (threadIdx.x == 0) {
        dst[row] = ggml_cuda_op_silu_single(tmp)*mul[row];
    }
}

__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_warp_rows_add(
        const void * vx, const block_q8_1 * y, const float * add, float * dst,
        const uint32_t ncols_x, const uint32_t stride_row_x) {
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_warp_iter = vdr*warp_size/qi;

    ggml_cuda_pdl_sync();

    const int row = 2*blockIdx.x + threadIdx.y;
    const int blocks_per_row_x = ncols_x/qk;
    const block_q8_0 * x = (const block_q8_0 *) vx + row*stride_row_x;
    float tmp = 0.0f;

    for (int kbx = threadIdx.x/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_warp_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(threadIdx.x % (qi/vdr));
        tmp += vec_dot_q8_0_q8_1(x, &y[kby], kbx, kqs);
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if (threadIdx.x == 0) {
        dst[row] = tmp + add[row];
    }
}

__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_bias(
        const void * vx, const block_q8_1 * y, const float * bias, float * dst,
        const uint32_t ncols_x, const uint32_t stride_row_x) {
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = vdr*2*warp_size/qi;

    ggml_cuda_pdl_sync();

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int row = blockIdx.x;
    const int blocks_per_row_x = ncols_x/qk;
    const block_q8_0 * x = (const block_q8_0 *) vx + row*stride_row_x;
    float x_bias = 0.0f;
    if (tid == 0) {
        x_bias = bias[row];
    }
    float tmp = 0.0f;

    for (int kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(tid % (qi/vdr));
        tmp += vec_dot_q8_0_q8_1(x, &y[kby], kbx, kqs);
    }

    __shared__ float tmp_shared[warp_size];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    tmp += tmp_shared[threadIdx.x];
    tmp = warp_reduce_sum<warp_size>(tmp);
    if (tid == 0) {
        float result = tmp;
        result += x_bias;
        dst[row] = result;
    }
}

__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_warp_rows_gate(
        const void * vx, const void * vgate, const block_q8_1 * y, float * dst,
        const uint32_t ncols_x, const uint32_t stride_row_x) {
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_warp_iter = vdr*warp_size/qi;

    ggml_cuda_pdl_sync();

    const int row = 2*blockIdx.x + threadIdx.y;
    const int blocks_per_row_x = ncols_x/qk;
    const block_q8_0 * x = (const block_q8_0 *) vx + row*stride_row_x;
    const block_q8_0 * gate = (const block_q8_0 *) vgate + row*stride_row_x;
    float tmp = 0.0f;
    float tmp_gate = 0.0f;

    for (int kbx = threadIdx.x/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_warp_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(threadIdx.x % (qi/vdr));
        tmp += vec_dot_q8_0_q8_1(x, &y[kby], kbx, kqs);
        tmp_gate += vec_dot_q8_0_q8_1(gate, &y[kby], kbx, kqs);
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    tmp_gate = warp_reduce_sum<warp_size>(tmp_gate);
    if (threadIdx.x == 0) {
        dst[row] = tmp*ggml_cuda_op_silu_single(tmp_gate);
    }
}

__launch_bounds__(512, 2)
static __global__ void mul_mat_vec_q8_0_warp_rows_gate_q8_1(
        const void * vx, const void * vgate, const block_q8_1 * y,
        float * dst, block_q8_1 * dst_q8_1) {
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = 32;
    constexpr int ncols_x = 1024;
    constexpr int blocks_per_row_x = ncols_x/qk;
    constexpr int blocks_per_warp_iter = vdr*warp_size/qi;

    ggml_cuda_pdl_sync();

    const int row0 = QK8_1*blockIdx.x + threadIdx.y;
    const int row1 = row0 + blockDim.y;
    const block_q8_0 * x0 = (const block_q8_0 *) vx + row0*blocks_per_row_x;
    const block_q8_0 * x1 = (const block_q8_0 *) vx + row1*blocks_per_row_x;
    const block_q8_0 * gate0 = (const block_q8_0 *) vgate + row0*blocks_per_row_x;
    const block_q8_0 * gate1 = (const block_q8_0 *) vgate + row1*blocks_per_row_x;
    float tmp0 = 0.0f;
    float tmp1 = 0.0f;
    float tmp_gate0 = 0.0f;
    float tmp_gate1 = 0.0f;

    for (int kbx = threadIdx.x/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_warp_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(threadIdx.x % (qi/vdr));
        tmp0 += vec_dot_q8_0_q8_1(x0, &y[kby], kbx, kqs);
        tmp1 += vec_dot_q8_0_q8_1(x1, &y[kby], kbx, kqs);
        tmp_gate0 += vec_dot_q8_0_q8_1(gate0, &y[kby], kbx, kqs);
        tmp_gate1 += vec_dot_q8_0_q8_1(gate1, &y[kby], kbx, kqs);
    }

    tmp0 = warp_reduce_sum<warp_size>(tmp0);
    tmp1 = warp_reduce_sum<warp_size>(tmp1);
    tmp_gate0 = warp_reduce_sum<warp_size>(tmp_gate0);
    tmp_gate1 = warp_reduce_sum<warp_size>(tmp_gate1);

    __shared__ float results[QK8_1];
    if (threadIdx.x == 0) {
        const float result0 = tmp0*ggml_cuda_op_silu_single(tmp_gate0);
        const float result1 = tmp1*ggml_cuda_op_silu_single(tmp_gate1);
        dst[row0] = result0;
        dst[row1] = result1;
        results[threadIdx.y] = result0;
        results[threadIdx.y + blockDim.y] = result1;
    }
    __syncthreads();

    if (threadIdx.y != 0) {
        return;
    }

    const float value = results[threadIdx.x];
    const float amax = warp_reduce_max<QK8_1>(fabsf(value));
    const float sum = warp_reduce_sum<QK8_1>(value);
    const float d = amax/127.0f;
    const int8_t quant = amax == 0.0f ? 0 : roundf(value/d);

    dst_q8_1[blockIdx.x].qs[threadIdx.x] = quant;
    if (threadIdx.x == 0) {
        dst_q8_1[blockIdx.x].ds = make_half2(d, sum);
    }
}

__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_grouped(
        const void * vx0, const void * vx1, const void * vx2, const block_q8_1 * y,
        float * dst0, float * dst1, float * dst2, const uint32_t ncols_x,
        const uint32_t stride_row_x0, const uint32_t stride_row_x1, const uint32_t stride_row_x2,
        const uint32_t nblocks0, const uint32_t nblocks1) {
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_warp_iter = vdr*warp_size/qi;

    const uint32_t block = blockIdx.x;
    const void * vx;
    float * dst;
    uint32_t row;
    uint32_t stride_row_x;

    if (block < nblocks0) {
        vx = vx0;
        dst = dst0;
        row = 2*block + threadIdx.y;
        stride_row_x = stride_row_x0;
    } else if (block < nblocks0 + nblocks1) {
        vx = vx1;
        dst = dst1;
        row = 2*(block - nblocks0) + threadIdx.y;
        stride_row_x = stride_row_x1;
    } else {
        vx = vx2;
        dst = dst2;
        row = 2*(block - nblocks0 - nblocks1) + threadIdx.y;
        stride_row_x = stride_row_x2;
    }

    const int blocks_per_row_x = ncols_x/qk;
    const block_q8_0 * x = (const block_q8_0 *) vx + row*stride_row_x;
    float tmp = 0.0f;

    for (int kbx = threadIdx.x/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_warp_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(threadIdx.x % (qi/vdr));
        tmp += vec_dot_q8_0_q8_1(x, &y[kby], kbx, kqs);
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

template <ggml_type type>
static __device__ float mul_mat_vec_q_grouped_partial(
        const void * vx, const block_q8_1 * y, const uint32_t ncols_x,
        const uint32_t row, const uint32_t stride_row_x) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = 2;
    constexpr int blocks_per_iter = vdr*nwarps*warp_size/qi;
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int blocks_per_row_x = ncols_x/qk;
    float tmp = 0.0f;
    for (int kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(tid % (qi/vdr));
        tmp += vec_dot_q_cuda(vx, &y[kby], row*stride_row_x + kbx, kqs);
    }
    return tmp;
}

__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_gdn(
        const void * vx_alpha, const void * vx_beta, const block_q8_1 * y,
        const float * alpha_bias, const float * alpha_scale, float * dst_gate, float * dst_beta,
        const uint32_t ncols_x, const uint32_t stride_row_alpha, const uint32_t stride_row_beta,
        const uint32_t nblocks_alpha) {
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_warp_iter = vdr*warp_size/qi;

    const bool is_alpha = blockIdx.x < nblocks_alpha;
    const uint32_t block = is_alpha ? blockIdx.x : blockIdx.x - nblocks_alpha;
    const uint32_t row = 2*block + threadIdx.y;
    const uint32_t stride_row_x = is_alpha ? stride_row_alpha : stride_row_beta;
    const block_q8_0 * x = (const block_q8_0 *) (is_alpha ? vx_alpha : vx_beta) + row*stride_row_x;
    const int blocks_per_row_x = ncols_x/qk;
    float tmp = 0.0f;

    for (int kbx = threadIdx.x/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_warp_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(threadIdx.x % (qi/vdr));
        tmp += vec_dot_q8_0_q8_1(x, &y[kby], kbx, kqs);
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if (threadIdx.x == 0) {
        if (is_alpha) {
            dst_gate[row] = ggml_cuda_op_softplus_single(tmp + alpha_bias[row])*alpha_scale[row];
        } else {
            dst_beta[row] = ggml_cuda_op_sigmoid_single(tmp);
        }
    }
}

__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_gdn_grouped(
        const void * vx_qkv, const void * vx_alpha, const void * vx_beta, const block_q8_1 * y,
        const float * alpha_bias, const float * alpha_scale,
        float * dst_qkv, float * dst_gate, float * dst_beta,
        const uint32_t ncols_x, const uint32_t stride_row_qkv,
        const uint32_t stride_row_alpha, const uint32_t stride_row_beta,
        const uint32_t nblocks_qkv, const uint32_t nblocks_alpha) {
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_warp_iter = vdr*warp_size/qi;

    const uint32_t block = blockIdx.x;
    const bool is_qkv = block < nblocks_qkv;
    const bool is_alpha = !is_qkv && block < nblocks_qkv + nblocks_alpha;
    const uint32_t matrix_block = is_qkv ? block :
        (is_alpha ? block - nblocks_qkv : block - nblocks_qkv - nblocks_alpha);
    const uint32_t row = 2*matrix_block + threadIdx.y;
    const void * vx = is_qkv ? vx_qkv : (is_alpha ? vx_alpha : vx_beta);
    const uint32_t stride_row_x = is_qkv ? stride_row_qkv :
        (is_alpha ? stride_row_alpha : stride_row_beta);
    const block_q8_0 * x = (const block_q8_0 *) vx + row*stride_row_x;
    const int blocks_per_row_x = ncols_x/qk;
    float tmp = 0.0f;

    for (int kbx = threadIdx.x/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_warp_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(threadIdx.x % (qi/vdr));
        tmp += vec_dot_q8_0_q8_1(x, &y[kby], kbx, kqs);
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if (threadIdx.x == 0) {
        if (is_qkv) {
            dst_qkv[row] = tmp;
        } else if (is_alpha) {
            dst_gate[row] = ggml_cuda_op_softplus_single(tmp + alpha_bias[row])*alpha_scale[row];
        } else {
            dst_beta[row] = ggml_cuda_op_sigmoid_single(tmp);
        }
    }
}

__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_gdn_conv(
        const void * vx_qkv, const void * vx_alpha, const void * vx_beta, const block_q8_1 * y,
        const float * alpha_bias, const float * alpha_scale,
        const float * conv_states, const float * conv_kernel,
        float * conv_scratch, float * conv_state_update, float * dst_gate, float * dst_beta,
        const uint32_t ncols_x, const uint32_t stride_row_qkv,
        const uint32_t stride_row_alpha, const uint32_t stride_row_beta,
        const uint32_t stride_row_state, const uint32_t stride_row_conv,
        const uint32_t nblocks_qkv, const uint32_t nblocks_alpha) {
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_warp_iter = vdr*warp_size/qi;

    const uint32_t block = blockIdx.x;
    const bool is_qkv = block < nblocks_qkv;
    const bool is_alpha = !is_qkv && block < nblocks_qkv + nblocks_alpha;
    const uint32_t matrix_block = is_qkv ? block :
        (is_alpha ? block - nblocks_qkv : block - nblocks_qkv - nblocks_alpha);
    const uint32_t row = 2*matrix_block + threadIdx.y;
    const void * vx = is_qkv ? vx_qkv : (is_alpha ? vx_alpha : vx_beta);
    const uint32_t stride_row_x = is_qkv ? stride_row_qkv :
        (is_alpha ? stride_row_alpha : stride_row_beta);
    const block_q8_0 * x = (const block_q8_0 *) vx + row*stride_row_x;
    const int blocks_per_row_x = ncols_x/qk;
    float tmp = 0.0f;

    for (int kbx = threadIdx.x/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_warp_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(threadIdx.x % (qi/vdr));
        tmp += vec_dot_q8_0_q8_1(x, &y[kby], kbx, kqs);
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if (threadIdx.x == 0) {
        if (is_qkv) {
            const float * state_row = conv_states + row*stride_row_state;
            const float * conv_row = conv_kernel + row*stride_row_conv;
            float sumf = 0.0f;
            sumf += state_row[0]*conv_row[0];
            sumf += state_row[1]*conv_row[1];
            sumf += state_row[2]*conv_row[2];
            sumf += tmp*conv_row[3];
            conv_scratch[row] = ggml_cuda_op_silu_single(sumf);
            conv_state_update[3*row + 0] = state_row[1];
            conv_state_update[3*row + 1] = state_row[2];
            conv_state_update[3*row + 2] = tmp;
        } else if (is_alpha) {
            dst_gate[row] = ggml_cuda_op_softplus_single(tmp + alpha_bias[row])*alpha_scale[row];
        } else {
            dst_beta[row] = ggml_cuda_op_sigmoid_single(tmp);
        }
    }
}

template <int block_size>
static __global__ void gdn_conv_finalize_f32(
        const float * conv_scratch, float * q_norm, float * k_norm, float * v_conv,
        const int ncols, const int nrows, const float eps) {
    const int section = blockIdx.x/nrows;
    const int row = blockIdx.x - section*nrows;
    const int tid = threadIdx.x;
    const float * x = conv_scratch + (section*nrows + row)*ncols;
    float * dst = section == 0 ? q_norm + row*ncols :
        (section == 1 ? k_norm + row*ncols : v_conv + row*ncols);

    if (section == 2) {
        for (int col = tid; col < ncols; col += block_size) {
            dst[col] = x[col];
        }
        return;
    }

    float tmp = 0.0f;
    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi*xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();
    const float scale = rsqrtf(fmaxf(tmp, eps*eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale*x[col];
    }
}

__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q4_0_gdn(
        const void * vx_alpha, const void * vx_beta, const block_q8_1 * y,
        const float * alpha_bias, const float * alpha_scale, float * dst_gate, float * dst_beta,
        const uint32_t ncols_x, const uint32_t stride_row_alpha, const uint32_t stride_row_beta,
        const uint32_t nrows_alpha) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const bool is_alpha = blockIdx.x < nrows_alpha;
    const uint32_t row = is_alpha ? blockIdx.x : blockIdx.x - nrows_alpha;
    const void * vx = is_alpha ? vx_alpha : vx_beta;
    const uint32_t stride_row_x = is_alpha ? stride_row_alpha : stride_row_beta;
    float tmp = mul_mat_vec_q_grouped_partial<GGML_TYPE_Q4_0>(vx, y, ncols_x, row, stride_row_x);

    __shared__ float tmp_shared[warp_size];
    if (threadIdx.y == 1) {
        tmp_shared[threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y == 1) {
        return;
    }

    tmp += tmp_shared[threadIdx.x];
    tmp = warp_reduce_sum<warp_size>(tmp);
    if (threadIdx.x == 0) {
        if (is_alpha) {
            dst_gate[row] = ggml_cuda_op_softplus_single(tmp + alpha_bias[row])*alpha_scale[row];
        } else {
            dst_beta[row] = ggml_cuda_op_sigmoid_single(tmp);
        }
    }
}

template <ggml_type type0, ggml_type type1, ggml_type type2>
__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_grouped_mixed(
        const void * vx0, const void * vx1, const void * vx2, const block_q8_1 * y,
        float * dst0, float * dst1, float * dst2, const uint32_t ncols_x,
        const uint32_t stride_row_x0, const uint32_t stride_row_x1, const uint32_t stride_row_x2,
        const uint32_t nrows0, const uint32_t nrows1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const uint32_t block = blockIdx.x;
    float * dst;
    uint32_t row;
    float tmp;

    if (block < nrows0) {
        dst = dst0;
        row = block;
        tmp = mul_mat_vec_q_grouped_partial<type0>(vx0, y, ncols_x, row, stride_row_x0);
    } else if (block < nrows0 + nrows1) {
        dst = dst1;
        row = block - nrows0;
        tmp = mul_mat_vec_q_grouped_partial<type1>(vx1, y, ncols_x, row, stride_row_x1);
    } else {
        dst = dst2;
        row = block - nrows0 - nrows1;
        tmp = mul_mat_vec_q_grouped_partial<type2>(vx2, y, ncols_x, row, stride_row_x2);
    }

    __shared__ float tmp_shared[warp_size];
    if (threadIdx.y == 1) {
        tmp_shared[threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y == 1) {
        return;
    }

    tmp += tmp_shared[threadIdx.x];
    tmp = warp_reduce_sum<warp_size>(tmp);
    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false,
        const bool has_fusion = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps, has_fusion);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y,
        const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (type == GGML_TYPE_Q8_0 && c_ncols_dst == 1 && !small_k) {
        const int device = ggml_cuda_get_device();
        const auto & device_info = ggml_cuda_info().devices[device];
        const bool dense_layout = ids == nullptr && block_nums.y == 1 && block_nums.z == 1 &&
            block_dims.x == uint32_t(device_info.warp_size) && block_dims.y == 2 && block_dims.z == 1 &&
            nrows_x % 2 == 0 && stride_col_dst == nrows_x;
        const bool volta = get_device_table_id(device_info.cc) == MMVQ_PARAMETERS_VOLTA;
        if (!has_fusion && dense_layout && uint64_t(block_nums.x)*2 == nrows_x && volta) {
            const ggml_cuda_kernel_launch_params launch_params =
                ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q8_0_warp_rows, launch_params,
                vx, (const block_q8_1 *) vy, dst, ncols_x, stride_row_x);
            return;
        }
        if (has_fusion && dense_layout && block_nums.x == nrows_x && volta) {
            if (fusion.x_bias != nullptr && fusion.gate == nullptr &&
                fusion.gate_bias == nullptr && fusion.x_scale == nullptr && fusion.gate_scale == nullptr) {
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
                    block_nums, block_dims, 0, stream);
                ggml_cuda_kernel_launch(mul_mat_vec_q8_0_bias, launch_params,
                    vx, (const block_q8_1 *) vy, (const float *) fusion.x_bias, dst, ncols_x, stride_row_x);
                return;
            }
            if (fusion.gate != nullptr && fusion.x_bias == nullptr && fusion.gate_bias == nullptr &&
                fusion.x_scale == nullptr && fusion.gate_scale == nullptr && fusion.glu_op == GGML_GLU_OP_SWIGLU) {
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
                    dim3(nrows_x/2, 1, 1), block_dims, 0, stream);
                ggml_cuda_kernel_launch(mul_mat_vec_q8_0_warp_rows_gate, launch_params,
                    vx, fusion.gate, (const block_q8_1 *) vy, dst, ncols_x, stride_row_x);
                return;
            }
        }
    }

    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block>, launch_params,
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
        constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr                   = get_vdr_mmvq(type);
        const int     blocks_per_row_x      = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id, true);
        bool          use                   = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

            bool use_small_k = should_use_small_k(c_ncols_dst);

            if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, true, has_fusion);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true>(
                    vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, false, has_fusion);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                    vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

bool ggml_cuda_mul_mat_vec_q_silu_mul(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * projection, const ggml_tensor * mul, ggml_tensor * dst) {
    const ggml_tensor * src0 = projection->src[0];
    const ggml_tensor * src1 = projection->src[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    static const bool disable_q8_1_cache = getenv("GGML_CUDA_DISABLE_Q8_1_CACHE") != nullptr;

    if (disable_q8_1_cache || cc != GGML_CUDA_CC_VOLTA || ctx.curr_stream_no != 0 ||
            projection->op != GGML_OP_MUL_MAT || src0->type != GGML_TYPE_Q8_0 ||
            src1->type != GGML_TYPE_F32 || projection->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            src0->ne[0] != 1024 || src0->ne[1] != 2048 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src1->ne[0] != 1024 || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            projection->ne[0] != 2048 || projection->ne[1] != 1 ||
            projection->ne[2] != 1 || projection->ne[3] != 1 ||
            ggml_nelements(mul) != 2048 || ggml_nelements(dst) != 2048 ||
            !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) ||
            !ggml_is_contiguous(mul) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const size_t src1_q8_1_size = 1024*sizeof(block_q8_1)/QK8_1;
    char * src1_q8_1 = ctx.q8_1_cache_get(src1, src1_q8_1_size);
    if (src1_q8_1 == nullptr) {
        return false;
    }

    const dim3 block_nums(2048/2, 1, 1);
    const dim3 block_dims(ggml_cuda_info().devices[ctx.device].warp_size, 2, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, ctx.stream());
    ggml_cuda_kernel_launch(mul_mat_vec_q8_0_warp_rows_silu_mul, launch_params,
        src0->data, (const block_q8_1 *) src1_q8_1, (const float *) mul->data, (float *) dst->data,
        1024, 1024/QK8_0);
    return true;
}

bool ggml_cuda_mul_mat_vec_q_add(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * projection, const ggml_tensor * add, ggml_tensor * dst) {
    const ggml_tensor * src0 = projection->src[0];
    const ggml_tensor * src1 = projection->src[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    static const bool disable_q8_1_cache = getenv("GGML_CUDA_DISABLE_Q8_1_CACHE") != nullptr;

    if (disable_q8_1_cache || cc != GGML_CUDA_CC_VOLTA || ctx.curr_stream_no != 0 ||
            projection->op != GGML_OP_MUL_MAT || src0->type != GGML_TYPE_Q8_0 ||
            src1->type != GGML_TYPE_F32 || projection->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            src0->ne[0] != 2048 || src0->ne[1] != 1024 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src1->ne[0] != 2048 || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            projection->ne[0] != 1024 || projection->ne[1] != 1 ||
            projection->ne[2] != 1 || projection->ne[3] != 1 ||
            ggml_nelements(add) != 1024 || ggml_nelements(dst) != 1024 ||
            !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) ||
            !ggml_is_contiguous(add) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const size_t src1_q8_1_size = 2048*sizeof(block_q8_1)/QK8_1;
    char * src1_q8_1 = ctx.q8_1_cache_get(src1, src1_q8_1_size);
    if (src1_q8_1 == nullptr) {
        src1_q8_1 = ctx.q8_1_cache_alloc(src1, src1_q8_1_size);
        const size_t ts_src1 = ggml_type_size(src1->type);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1, src0->type,
            2048, s11, s12, s13, 2048, 1, 1, 1, ctx.stream());
    }

    const dim3 block_nums(1024/2, 1, 1);
    const dim3 block_dims(ggml_cuda_info().devices[ctx.device].warp_size, 2, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, ctx.stream());
    ggml_cuda_kernel_launch(mul_mat_vec_q8_0_warp_rows_add, launch_params,
        src0->data, (const block_q8_1 *) src1_q8_1, (const float *) add->data, (float *) dst->data,
        2048, 2048/QK8_0);
    return true;
}

bool ggml_cuda_mul_mat_vec_q_grouped(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst0, const ggml_tensor * dst1, const ggml_tensor * dst2) {
    const ggml_tensor * src00 = dst0->src[0];
    const ggml_tensor * src01 = dst1->src[0];
    const ggml_tensor * src02 = dst2->src[0];
    const ggml_tensor * src1  = dst0->src[1];

    const bool same_q4_0 = src00->type == GGML_TYPE_Q4_0 &&
        src01->type == GGML_TYPE_Q4_0 && src02->type == GGML_TYPE_Q4_0;
    const bool same_q8_0 = src00->type == GGML_TYPE_Q8_0 &&
        src01->type == GGML_TYPE_Q8_0 && src02->type == GGML_TYPE_Q8_0;
    const bool mixed_q2 = src00->type == GGML_TYPE_Q2_K &&
        src01->type == GGML_TYPE_Q4_K && src02->type == GGML_TYPE_Q2_K;
    const bool mixed_q6 = src00->type == GGML_TYPE_Q6_K &&
        src01->type == GGML_TYPE_Q4_K && src02->type == GGML_TYPE_Q2_K;

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc != GGML_CUDA_CC_VOLTA || ctx.curr_stream_no != 0 ||
            dst0->op != GGML_OP_MUL_MAT || dst1->op != GGML_OP_MUL_MAT || dst2->op != GGML_OP_MUL_MAT ||
            dst1->src[1] != src1 || dst2->src[1] != src1 ||
            (!same_q4_0 && !same_q8_0 && !mixed_q2 && !mixed_q6) ||
            src1->type != GGML_TYPE_F32 || dst0->type != GGML_TYPE_F32 ||
            dst1->type != GGML_TYPE_F32 || dst2->type != GGML_TYPE_F32 ||
            src00->ne[0] < 1024 ||
            src1->ne[0] != src00->ne[0] ||
            src00->ne[0] != src01->ne[0] || src00->ne[0] != src02->ne[0] ||
            src00->ne[2] != 1 || src00->ne[3] != 1 ||
            src01->ne[2] != 1 || src01->ne[3] != 1 ||
            src02->ne[2] != 1 || src02->ne[3] != 1 ||
            src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            dst0->ne[1] != 1 || dst0->ne[2] != 1 || dst0->ne[3] != 1 ||
            dst1->ne[1] != 1 || dst1->ne[2] != 1 || dst1->ne[3] != 1 ||
            dst2->ne[1] != 1 || dst2->ne[2] != 1 || dst2->ne[3] != 1 ||
            dst0->ne[0] != src00->ne[1] || dst1->ne[0] != src01->ne[1] || dst2->ne[0] != src02->ne[1] ||
            ggml_backend_buffer_get_usage(src00->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            ggml_backend_buffer_get_usage(src01->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            ggml_backend_buffer_get_usage(src02->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }

    if (same_q8_0 &&
            (src00->ne[1] % 2 != 0 || src01->ne[1] % 2 != 0 || src02->ne[1] % 2 != 0)) {
        return false;
    }

    const int64_t ne10 = src1->ne[0];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t src1_q8_1_size = ne10_padded*sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_local(ctx.pool());

    static const bool disable_q8_1_cache = getenv("GGML_CUDA_DISABLE_Q8_1_CACHE") != nullptr;
    char * src1_q8_1 = disable_q8_1_cache ? nullptr : ctx.q8_1_cache_get(src1, src1_q8_1_size);
    if (src1_q8_1 == nullptr) {
        src1_q8_1 = disable_q8_1_cache ? src1_q8_1_local.alloc(src1_q8_1_size) :
                                        ctx.q8_1_cache_alloc(src1, src1_q8_1_size);
        const size_t ts_src1 = ggml_type_size(src1->type);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1, src00->type,
            ne10, s11, s12, s13, ne10_padded, 1, 1, 1, ctx.stream());
    }

    const uint32_t nrows0 = src00->ne[1];
    const uint32_t nrows1 = src01->ne[1];
    const uint32_t nrows2 = src02->ne[1];
    const uint32_t stride_row_x0 = src00->nb[1] / ggml_type_size(src00->type);
    const uint32_t stride_row_x1 = src01->nb[1] / ggml_type_size(src01->type);
    const uint32_t stride_row_x2 = src02->nb[1] / ggml_type_size(src02->type);
    const dim3 block_dims(ggml_cuda_info().devices[ctx.device].warp_size, 2, 1);

    if (same_q8_0) {
        const uint32_t nblocks0 = nrows0/2;
        const uint32_t nblocks1 = nrows1/2;
        const dim3 block_nums(nblocks0 + nblocks1 + nrows2/2, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, ctx.stream());
        ggml_cuda_kernel_launch(mul_mat_vec_q8_0_grouped, launch_params,
            src00->data, src01->data, src02->data, (const block_q8_1 *) src1_q8_1,
            (float *) dst0->data, (float *) dst1->data, (float *) dst2->data, (uint32_t) src00->ne[0],
            stride_row_x0, stride_row_x1, stride_row_x2, nblocks0, nblocks1);
    } else if (same_q4_0) {
        const dim3 block_nums(nrows0 + nrows1 + nrows2, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, ctx.stream());
        ggml_cuda_kernel_launch(mul_mat_vec_q_grouped_mixed<GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>, launch_params,
            src00->data, src01->data, src02->data, (const block_q8_1 *) src1_q8_1,
            (float *) dst0->data, (float *) dst1->data, (float *) dst2->data, (uint32_t) src00->ne[0],
            stride_row_x0, stride_row_x1, stride_row_x2, nrows0, nrows1);
    } else {
        const dim3 block_nums(nrows0 + nrows1 + nrows2, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, ctx.stream());
        if (mixed_q2) {
            ggml_cuda_kernel_launch(mul_mat_vec_q_grouped_mixed<GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q2_K>, launch_params,
                src00->data, src01->data, src02->data, (const block_q8_1 *) src1_q8_1,
                (float *) dst0->data, (float *) dst1->data, (float *) dst2->data, (uint32_t) src00->ne[0],
                stride_row_x0, stride_row_x1, stride_row_x2, nrows0, nrows1);
        } else {
            ggml_cuda_kernel_launch(mul_mat_vec_q_grouped_mixed<GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, GGML_TYPE_Q2_K>, launch_params,
                src00->data, src01->data, src02->data, (const block_q8_1 *) src1_q8_1,
                (float *) dst0->data, (float *) dst1->data, (float *) dst2->data, (uint32_t) src00->ne[0],
                stride_row_x0, stride_row_x1, stride_row_x2, nrows0, nrows1);
        }
    }
    return true;
}

bool ggml_cuda_mul_mat_vec_q_gdn(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * alpha, const ggml_tensor * beta,
        const ggml_tensor * alpha_bias, const ggml_tensor * alpha_scale,
        ggml_tensor * dst_gate, ggml_tensor * dst_beta) {
    const ggml_tensor * src_alpha = alpha->src[0];
    const ggml_tensor * src_beta  = beta->src[0];
    const ggml_tensor * src1      = alpha->src[1];
    const bool q4_0 = src_alpha->type == GGML_TYPE_Q4_0 && src_beta->type == GGML_TYPE_Q4_0;
    const bool q8_0 = src_alpha->type == GGML_TYPE_Q8_0 && src_beta->type == GGML_TYPE_Q8_0;

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc != GGML_CUDA_CC_VOLTA || ctx.curr_stream_no != 0 ||
            alpha->op != GGML_OP_MUL_MAT || beta->op != GGML_OP_MUL_MAT || beta->src[1] != src1 ||
            (!q4_0 && !q8_0) || src1->type != GGML_TYPE_F32 ||
            alpha->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 ||
            alpha_bias->type != GGML_TYPE_F32 || alpha_scale->type != GGML_TYPE_F32 ||
            dst_gate->type != GGML_TYPE_F32 || dst_beta->type != GGML_TYPE_F32 ||
            src_alpha->ne[0] < 1024 || src_alpha->ne[0] != src_beta->ne[0] ||
            src1->ne[0] != src_alpha->ne[0] ||
            src_alpha->ne[2] != 1 || src_alpha->ne[3] != 1 ||
            src_beta->ne[2] != 1 || src_beta->ne[3] != 1 ||
            src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            alpha->ne[1] != 1 || alpha->ne[2] != 1 || alpha->ne[3] != 1 ||
            beta->ne[1] != 1 || beta->ne[2] != 1 || beta->ne[3] != 1 ||
            alpha->ne[0] != src_alpha->ne[1] || beta->ne[0] != src_beta->ne[1] ||
            ggml_nelements(alpha_bias) != alpha->ne[0] || ggml_nelements(alpha_scale) != alpha->ne[0] ||
            ggml_nelements(dst_gate) != alpha->ne[0] || ggml_nelements(dst_beta) != beta->ne[0] ||
            !ggml_is_contiguous(alpha_bias) || !ggml_is_contiguous(alpha_scale) ||
            !ggml_is_contiguous(dst_gate) || !ggml_is_contiguous(dst_beta) ||
            ggml_backend_buffer_get_usage(src_alpha->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            ggml_backend_buffer_get_usage(src_beta->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }

    if (q8_0 && (src_alpha->ne[1] % 2 != 0 || src_beta->ne[1] % 2 != 0)) {
        return false;
    }

    const int64_t ne10 = src1->ne[0];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t src1_q8_1_size = ne10_padded*sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_local(ctx.pool());

    static const bool disable_q8_1_cache = getenv("GGML_CUDA_DISABLE_Q8_1_CACHE") != nullptr;
    char * src1_q8_1 = disable_q8_1_cache ? nullptr : ctx.q8_1_cache_get(src1, src1_q8_1_size);
    if (src1_q8_1 == nullptr) {
        src1_q8_1 = disable_q8_1_cache ? src1_q8_1_local.alloc(src1_q8_1_size) :
                                        ctx.q8_1_cache_alloc(src1, src1_q8_1_size);
        const size_t ts_src1 = ggml_type_size(src1->type);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1, src_alpha->type,
            ne10, s11, s12, s13, ne10_padded, 1, 1, 1, ctx.stream());
    }

    const uint32_t nrows_alpha     = src_alpha->ne[1];
    const uint32_t nrows_beta      = src_beta->ne[1];
    const uint32_t stride_row_alpha = src_alpha->nb[1] / ggml_type_size(src_alpha->type);
    const uint32_t stride_row_beta  = src_beta->nb[1] / ggml_type_size(src_beta->type);
    const dim3 block_dims(ggml_cuda_info().devices[ctx.device].warp_size, 2, 1);

    if (q8_0) {
        const uint32_t nblocks_alpha = nrows_alpha/2;
        const dim3 block_nums(nblocks_alpha + nrows_beta/2, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, ctx.stream());
        ggml_cuda_kernel_launch(mul_mat_vec_q8_0_gdn, launch_params,
            src_alpha->data, src_beta->data, (const block_q8_1 *) src1_q8_1,
            (const float *) alpha_bias->data, (const float *) alpha_scale->data,
            (float *) dst_gate->data, (float *) dst_beta->data, (uint32_t) src_alpha->ne[0],
            stride_row_alpha, stride_row_beta, nblocks_alpha);
    } else {
        const dim3 block_nums(nrows_alpha + nrows_beta, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, ctx.stream());
        ggml_cuda_kernel_launch(mul_mat_vec_q4_0_gdn, launch_params,
            src_alpha->data, src_beta->data, (const block_q8_1 *) src1_q8_1,
            (const float *) alpha_bias->data, (const float *) alpha_scale->data,
            (float *) dst_gate->data, (float *) dst_beta->data, (uint32_t) src_alpha->ne[0],
            stride_row_alpha, stride_row_beta, nrows_alpha);
    }
    return true;
}

bool ggml_cuda_mul_mat_vec_q_gdn_grouped(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * qkv, const ggml_tensor * alpha, const ggml_tensor * beta,
        const ggml_tensor * alpha_bias, const ggml_tensor * alpha_scale,
        ggml_tensor * dst_gate, ggml_tensor * dst_beta) {
    const ggml_tensor * src_qkv   = qkv->src[0];
    const ggml_tensor * src_alpha = alpha->src[0];
    const ggml_tensor * src_beta  = beta->src[0];
    const ggml_tensor * src1      = qkv->src[1];

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc != GGML_CUDA_CC_VOLTA || ctx.curr_stream_no != 0 ||
            qkv->op != GGML_OP_MUL_MAT || alpha->op != GGML_OP_MUL_MAT || beta->op != GGML_OP_MUL_MAT ||
            alpha->src[1] != src1 || beta->src[1] != src1 ||
            src_qkv->type != GGML_TYPE_Q8_0 || src_alpha->type != GGML_TYPE_Q8_0 ||
            src_beta->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 ||
            qkv->type != GGML_TYPE_F32 || alpha->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 ||
            alpha_bias->type != GGML_TYPE_F32 || alpha_scale->type != GGML_TYPE_F32 ||
            dst_gate->type != GGML_TYPE_F32 || dst_beta->type != GGML_TYPE_F32 ||
            src_qkv->ne[0] < 1024 || src_qkv->ne[0] != src_alpha->ne[0] || src_qkv->ne[0] != src_beta->ne[0] ||
            src1->ne[0] != src_qkv->ne[0] ||
            src_qkv->ne[2] != 1 || src_qkv->ne[3] != 1 ||
            src_alpha->ne[2] != 1 || src_alpha->ne[3] != 1 ||
            src_beta->ne[2] != 1 || src_beta->ne[3] != 1 ||
            src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            qkv->ne[1] != 1 || qkv->ne[2] != 1 || qkv->ne[3] != 1 ||
            alpha->ne[1] != 1 || alpha->ne[2] != 1 || alpha->ne[3] != 1 ||
            beta->ne[1] != 1 || beta->ne[2] != 1 || beta->ne[3] != 1 ||
            qkv->ne[0] != src_qkv->ne[1] || alpha->ne[0] != src_alpha->ne[1] || beta->ne[0] != src_beta->ne[1] ||
            ggml_nelements(alpha_bias) != alpha->ne[0] || ggml_nelements(alpha_scale) != alpha->ne[0] ||
            ggml_nelements(dst_gate) != alpha->ne[0] || ggml_nelements(dst_beta) != beta->ne[0] ||
            !ggml_is_contiguous(src_qkv) || !ggml_is_contiguous(src_alpha) || !ggml_is_contiguous(src_beta) ||
            !ggml_is_contiguous(qkv) || !ggml_is_contiguous(alpha_bias) || !ggml_is_contiguous(alpha_scale) ||
            !ggml_is_contiguous(dst_gate) || !ggml_is_contiguous(dst_beta) ||
            src_qkv->ne[1] % 2 != 0 || src_alpha->ne[1] % 2 != 0 || src_beta->ne[1] % 2 != 0 ||
            ggml_backend_buffer_get_usage(src_qkv->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            ggml_backend_buffer_get_usage(src_alpha->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            ggml_backend_buffer_get_usage(src_beta->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }

    const int64_t ne10 = src1->ne[0];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t src1_q8_1_size = ne10_padded*sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_local(ctx.pool());

    static const bool disable_q8_1_cache = getenv("GGML_CUDA_DISABLE_Q8_1_CACHE") != nullptr;
    char * src1_q8_1 = disable_q8_1_cache ? nullptr : ctx.q8_1_cache_get(src1, src1_q8_1_size);
    if (src1_q8_1 == nullptr) {
        src1_q8_1 = disable_q8_1_cache ? src1_q8_1_local.alloc(src1_q8_1_size) :
                                        ctx.q8_1_cache_alloc(src1, src1_q8_1_size);
        const size_t ts_src1 = ggml_type_size(src1->type);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1, src_qkv->type,
            ne10, s11, s12, s13, ne10_padded, 1, 1, 1, ctx.stream());
    }

    const uint32_t nrows_qkv        = src_qkv->ne[1];
    const uint32_t nrows_alpha      = src_alpha->ne[1];
    const uint32_t nrows_beta       = src_beta->ne[1];
    const uint32_t stride_row_qkv   = src_qkv->nb[1] / ggml_type_size(src_qkv->type);
    const uint32_t stride_row_alpha = src_alpha->nb[1] / ggml_type_size(src_alpha->type);
    const uint32_t stride_row_beta  = src_beta->nb[1] / ggml_type_size(src_beta->type);
    const uint32_t nblocks_qkv      = nrows_qkv/2;
    const uint32_t nblocks_alpha    = nrows_alpha/2;
    const dim3 block_nums(nblocks_qkv + nblocks_alpha + nrows_beta/2, 1, 1);
    const dim3 block_dims(ggml_cuda_info().devices[ctx.device].warp_size, 2, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, ctx.stream());
    ggml_cuda_kernel_launch(mul_mat_vec_q8_0_gdn_grouped, launch_params,
        src_qkv->data, src_alpha->data, src_beta->data, (const block_q8_1 *) src1_q8_1,
        (const float *) alpha_bias->data, (const float *) alpha_scale->data,
        (float *) qkv->data, (float *) dst_gate->data, (float *) dst_beta->data,
        (uint32_t) src_qkv->ne[0], stride_row_qkv, stride_row_alpha, stride_row_beta,
        nblocks_qkv, nblocks_alpha);
    return true;
}

bool ggml_cuda_mul_mat_vec_q_gdn_conv(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * qkv, const ggml_tensor * alpha, const ggml_tensor * beta,
        const ggml_tensor * alpha_bias, const ggml_tensor * alpha_scale,
        const ggml_tensor * conv_states, const ggml_tensor * conv_kernel,
        ggml_tensor * conv_scratch, ggml_tensor * conv_state_update,
        ggml_tensor * q_norm, ggml_tensor * k_norm, ggml_tensor * v_conv,
        ggml_tensor * dst_gate, ggml_tensor * dst_beta) {
    const ggml_tensor * src_qkv   = qkv->src[0];
    const ggml_tensor * src_alpha = alpha->src[0];
    const ggml_tensor * src_beta  = beta->src[0];
    const ggml_tensor * src1      = qkv->src[1];
    float eps_q;
    float eps_k;
    memcpy(&eps_q, q_norm->op_params, sizeof(float));
    memcpy(&eps_k, k_norm->op_params, sizeof(float));

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc != GGML_CUDA_CC_VOLTA || ctx.curr_stream_no != 0 ||
            qkv->op != GGML_OP_MUL_MAT || alpha->op != GGML_OP_MUL_MAT || beta->op != GGML_OP_MUL_MAT ||
            q_norm->op != GGML_OP_L2_NORM || k_norm->op != GGML_OP_L2_NORM ||
            alpha->src[1] != src1 || beta->src[1] != src1 ||
            src_qkv->type != GGML_TYPE_Q8_0 || src_alpha->type != GGML_TYPE_Q8_0 ||
            src_beta->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 ||
            qkv->type != GGML_TYPE_F32 || alpha->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 ||
            alpha_bias->type != GGML_TYPE_F32 || alpha_scale->type != GGML_TYPE_F32 ||
            conv_states->type != GGML_TYPE_F32 || conv_kernel->type != GGML_TYPE_F32 ||
            conv_scratch->type != GGML_TYPE_F32 || conv_state_update->type != GGML_TYPE_F32 ||
            q_norm->type != GGML_TYPE_F32 || k_norm->type != GGML_TYPE_F32 || v_conv->type != GGML_TYPE_F32 ||
            dst_gate->type != GGML_TYPE_F32 || dst_beta->type != GGML_TYPE_F32 ||
            src_qkv->ne[0] < 1024 || src_qkv->ne[0] != src_alpha->ne[0] || src_qkv->ne[0] != src_beta->ne[0] ||
            src1->ne[0] != src_qkv->ne[0] ||
            src_qkv->ne[2] != 1 || src_qkv->ne[3] != 1 ||
            src_alpha->ne[2] != 1 || src_alpha->ne[3] != 1 ||
            src_beta->ne[2] != 1 || src_beta->ne[3] != 1 ||
            src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            qkv->ne[1] != 1 || qkv->ne[2] != 1 || qkv->ne[3] != 1 ||
            alpha->ne[1] != 1 || alpha->ne[2] != 1 || alpha->ne[3] != 1 ||
            beta->ne[1] != 1 || beta->ne[2] != 1 || beta->ne[3] != 1 ||
            qkv->ne[0] != src_qkv->ne[1] || alpha->ne[0] != src_alpha->ne[1] || beta->ne[0] != src_beta->ne[1] ||
            conv_states->ne[0] != 3 || conv_states->ne[1] != qkv->ne[0] ||
            conv_states->ne[2] != 1 || conv_states->ne[3] != 1 ||
            conv_kernel->ne[0] != 4 || conv_kernel->ne[1] != qkv->ne[0] ||
            conv_kernel->ne[2] != 1 || conv_kernel->ne[3] != 1 ||
            ggml_nelements(conv_scratch) < qkv->ne[0] ||
            ggml_nelements(conv_state_update) != 3*qkv->ne[0] ||
            q_norm->ne[0] <= 0 || q_norm->ne[0] >= 1024 || q_norm->ne[1] <= 0 ||
            q_norm->ne[2] != 1 || q_norm->ne[3] != 1 ||
            !ggml_are_same_shape(q_norm, k_norm) || !ggml_are_same_shape(q_norm, v_conv) ||
            qkv->ne[0] != 3*ggml_nelements(q_norm) || eps_q < 0.0f || eps_q != eps_k ||
            ggml_nelements(alpha_bias) != alpha->ne[0] || ggml_nelements(alpha_scale) != alpha->ne[0] ||
            ggml_nelements(dst_gate) != alpha->ne[0] || ggml_nelements(dst_beta) != beta->ne[0] ||
            !ggml_is_contiguous(src_qkv) || !ggml_is_contiguous(src_alpha) || !ggml_is_contiguous(src_beta) ||
            !ggml_is_contiguous(conv_states) || !ggml_is_contiguous(conv_kernel) ||
            !ggml_is_contiguous(conv_scratch) || !ggml_is_contiguous(conv_state_update) ||
            !ggml_is_contiguous(q_norm) || !ggml_is_contiguous(k_norm) || !ggml_is_contiguous(v_conv) ||
            !ggml_is_contiguous(alpha_bias) || !ggml_is_contiguous(alpha_scale) ||
            !ggml_is_contiguous(dst_gate) || !ggml_is_contiguous(dst_beta) ||
            src_qkv->ne[1] % 2 != 0 || src_alpha->ne[1] % 2 != 0 || src_beta->ne[1] % 2 != 0 ||
            ggml_backend_buffer_get_usage(src_qkv->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            ggml_backend_buffer_get_usage(src_alpha->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            ggml_backend_buffer_get_usage(src_beta->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }

    const int64_t ne10 = src1->ne[0];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t src1_q8_1_size = ne10_padded*sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_local(ctx.pool());

    static const bool disable_q8_1_cache = getenv("GGML_CUDA_DISABLE_Q8_1_CACHE") != nullptr;
    char * src1_q8_1 = disable_q8_1_cache ? nullptr : ctx.q8_1_cache_get(src1, src1_q8_1_size);
    if (src1_q8_1 == nullptr) {
        src1_q8_1 = disable_q8_1_cache ? src1_q8_1_local.alloc(src1_q8_1_size) :
                                        ctx.q8_1_cache_alloc(src1, src1_q8_1_size);
        const size_t ts_src1 = ggml_type_size(src1->type);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1, src_qkv->type,
            ne10, s11, s12, s13, ne10_padded, 1, 1, 1, ctx.stream());
    }

    const uint32_t nrows_qkv        = src_qkv->ne[1];
    const uint32_t nrows_alpha      = src_alpha->ne[1];
    const uint32_t nrows_beta       = src_beta->ne[1];
    const uint32_t stride_row_qkv   = src_qkv->nb[1] / ggml_type_size(src_qkv->type);
    const uint32_t stride_row_alpha = src_alpha->nb[1] / ggml_type_size(src_alpha->type);
    const uint32_t stride_row_beta  = src_beta->nb[1] / ggml_type_size(src_beta->type);
    const uint32_t stride_row_state = conv_states->nb[1] / sizeof(float);
    const uint32_t stride_row_conv  = conv_kernel->nb[1] / sizeof(float);
    const uint32_t nblocks_qkv      = nrows_qkv/2;
    const uint32_t nblocks_alpha    = nrows_alpha/2;
    const dim3 block_nums(nblocks_qkv + nblocks_alpha + nrows_beta/2, 1, 1);
    const dim3 block_dims(ggml_cuda_info().devices[ctx.device].warp_size, 2, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, ctx.stream());
    // The final convolution output aliases conv_states, so keep it in concat scratch until this launch completes.
    ggml_cuda_kernel_launch(mul_mat_vec_q8_0_gdn_conv, launch_params,
        src_qkv->data, src_alpha->data, src_beta->data, (const block_q8_1 *) src1_q8_1,
        (const float *) alpha_bias->data, (const float *) alpha_scale->data,
        (const float *) conv_states->data, (const float *) conv_kernel->data,
        (float *) conv_scratch->data, (float *) conv_state_update->data,
        (float *) dst_gate->data, (float *) dst_beta->data,
        (uint32_t) src_qkv->ne[0], stride_row_qkv, stride_row_alpha, stride_row_beta,
        stride_row_state, stride_row_conv, nblocks_qkv, nblocks_alpha);

    constexpr int finalize_block_size = WARP_SIZE;
    const dim3 finalize_blocks(3*q_norm->ne[1], 1, 1);
    const dim3 finalize_threads(finalize_block_size, 1, 1);
    const ggml_cuda_kernel_launch_params finalize_params(finalize_blocks, finalize_threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(gdn_conv_finalize_f32<finalize_block_size>, finalize_params,
        (const float *) conv_scratch->data, (float *) q_norm->data, (float *) k_norm->data,
        (float *) v_conv->data, (int) q_norm->ne[0], (int) q_norm->ne[1], eps_q);
    return true;
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t src1_q8_1_size = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_local(ctx.pool());

    static const bool disable_q8_1_cache = getenv("GGML_CUDA_DISABLE_Q8_1_CACHE") != nullptr;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool use_q8_1_cache = !disable_q8_1_cache && cc == GGML_CUDA_CC_VOLTA &&
        ggml_is_quantized(src0->type) && ids == nullptr && ne1 == 1 && ne2 == 1 && ne3 == 1 &&
        ctx.curr_stream_no == 0;

    char * src1_q8_1 = use_q8_1_cache ? ctx.q8_1_cache_get(src1, src1_q8_1_size) : nullptr;
    if (src1_q8_1 == nullptr) {
        src1_q8_1 = use_q8_1_cache ? ctx.q8_1_cache_alloc(src1, src1_q8_1_size) : src1_q8_1_local.alloc(src1_q8_1_size);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    const bool use_gate_q8_1 = fusion != nullptr && fusion->prequantize &&
        use_q8_1_cache && src0->type == GGML_TYPE_Q8_0 && fusion_local.gate != nullptr &&
        fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr &&
        fusion_local.x_scale == nullptr && fusion_local.gate_scale == nullptr &&
        fusion_local.glu_op == GGML_GLU_OP_SWIGLU &&
        ne00 == 1024 && ne01 == 3584 && ne02 == 1 && ne03 == 1 &&
        ne10 == 1024 && ne11 == 1 && ne12 == 1 && ne13 == 1 &&
        ne0 == 3584 && ne1 == 1 && ne2 == 1 && ne3 == 1 &&
        s01 == ne00/QK8_0 && s1 == ne0 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(fusion->gate) &&
        ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    if (use_gate_q8_1) {
        const size_t dst_q8_1_size = ne0*sizeof(block_q8_1)/QK8_1;
        char * dst_q8_1 = ctx.q8_1_cache_get(dst, dst_q8_1_size);
        if (dst_q8_1 == nullptr) {
            dst_q8_1 = ctx.q8_1_cache_alloc(dst, dst_q8_1_size);
        }
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3(ne0/QK8_1, 1, 1), dim3(32, QK8_1/2, 1), 0, stream);
        ggml_cuda_kernel_launch(mul_mat_vec_q8_0_warp_rows_gate_q8_1, launch_params,
            src0->data, fusion_local.gate, (const block_q8_1 *) src1_q8_1, dst_d, (block_q8_1 *) dst_q8_1);
        return;
    }

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
