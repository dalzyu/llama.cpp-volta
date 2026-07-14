#include "concat.cuh"

#include <stdint.h>

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x,
                                                                             const T * y,
                                                                             T *       dst,
                                                                             int64_t   ne00,
                                                                             int64_t   ne01,
                                                                             int64_t   ne02,
                                                                             int64_t   ne0,
                                                                             int64_t   ne1,
                                                                             int64_t   ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

template <typename T>
static void concat_cont_cuda(const T * x,
                             const T * y,
                             T *       dst,
                             int64_t   ne00,
                             int64_t   ne01,
                             int64_t   ne02,
                             int64_t   ne0,
                             int64_t   ne1,
                             int64_t   ne2,
                             int       dim,
                             cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    if (dim == 0) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(concat_cont<T, 0>, launch_params, x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
}

static __device__ __forceinline__ float concat_dim0_value(
        const float * x, const float * y, const int64_t i,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3) {
    const int64_t plane = ne0*ne1;
    const int64_t i3 = i/(plane*ne2);
    const int64_t i2 = (i - i3*plane*ne2)/plane;
    const int64_t i1 = (i - i3*plane*ne2 - i2*plane)/ne0;
    const int64_t i0 = i - i3*plane*ne2 - i2*plane - i1*ne0;

    if (i0 < ne00) {
        return x[i3*ne00*ne01*ne02 + i2*ne00*ne01 + i1*ne00 + i0];
    }

    const int64_t j0 = i0 - ne00;
    return y[i3*ne10*ne11*ne12 + i2*ne10*ne11 + i1*ne10 + j0];
}

static __global__ void concat_cpy_dim0_f32(
        const float * x, const float * y, float * dst, float * copy_dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t view_offs, const int64_t view_ne0, const int64_t view_ne1,
        const int64_t view_ne2, const int64_t view_ne3,
        const int64_t view_nb0, const int64_t view_nb1, const int64_t view_nb2,
        const int64_t view_nb3, const int64_t dst_ne0, const int64_t dst_ne1,
        const int64_t dst_ne2, const int64_t dst_ne3, const int64_t dst_nb0,
        const int64_t dst_nb1, const int64_t dst_nb2, const int64_t dst_nb3,
        const int64_t copy_ne) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    const int64_t total = ne0*ne1*ne2*ne3;

    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
            i < total; i += (int64_t) blockDim.x*gridDim.x) {
        dst[i] = concat_dim0_value(x, y, i,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3);

        if (i < copy_ne) {
            const int64_t view_plane = view_ne0*view_ne1;
            const int64_t i3 = i/(view_plane*view_ne2);
            const int64_t i2 = (i - i3*view_plane*view_ne2)/view_plane;
            const int64_t i1 = (i - i3*view_plane*view_ne2 - i2*view_plane)/view_ne0;
            const int64_t i0 = i - i3*view_plane*view_ne2 - i2*view_plane - i1*view_ne0;
            const int64_t src_offs = view_offs + i0*view_nb0 + i1*view_nb1 + i2*view_nb2 + i3*view_nb3;
            const int64_t dst_plane = dst_ne0*dst_ne1;
            const int64_t d3 = i/(dst_plane*dst_ne2);
            const int64_t d2 = (i - d3*dst_plane*dst_ne2)/dst_plane;
            const int64_t d1 = (i - d3*dst_plane*dst_ne2 - d2*dst_plane)/dst_ne0;
            const int64_t d0 = i - d3*dst_plane*dst_ne2 - d2*dst_plane - d1*dst_ne0;
            const int64_t dst_offs = d0*dst_nb0 + d1*dst_nb1 + d2*dst_nb2 + d3*dst_nb3;

            copy_dst[dst_offs/sizeof(float)] = concat_dim0_value(x, y, src_offs/sizeof(float),
                ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3);
        }
    }

}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {
    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T *       dst_d  = (T *) dst->data;

        if (dim != 3) {
            for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_cont_cuda(
                        src0_d + i3*(src0->nb[3] / sizeof(T)),
                        src1_d + i3*(src1->nb[3] / sizeof(T)),
                        dst_d  + i3*( dst->nb[3] / sizeof(T)),
                        ggml_row_size(src0->type, src0->ne[0])/sizeof(T), src0->ne[1], src0->ne[2],
                        ggml_row_size(dst->type, dst->ne[0])/sizeof(T),  dst->ne[1],  dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            CUDA_CHECK(cudaMemcpyAsync((char *) dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type  == src0->type);

    if (ggml_is_quantized(src0->type)) {
        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ggml_is_contiguous(src1));
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        // if tensors are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
        concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
    } else {
        GGML_ASSERT(ggml_blck_size(src0->type) == 1);

        switch (ggml_type_size(src0->type)) {
            case 1:
                concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
                break;
            case 2:
                concat_cuda<uint16_t>(src0, src1, dst, dim, stream);
                break;
            case 4:
                concat_cuda<uint32_t>(src0, src1, dst, dim, stream);
                break;
            case 8:
                concat_cuda<uint64_t>(src0, src1, dst, dim, stream);
                break;
            default:
                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
                break;
        }
    }
}

void ggml_cuda_op_concat_cpy(ggml_backend_cuda_context & ctx,
                             ggml_tensor *               concat_node,
                             ggml_tensor *               cpy_node) {
    const ggml_tensor * src0      = concat_node->src[0];
    const ggml_tensor * src1      = concat_node->src[1];
    const ggml_tensor * copy_src  = cpy_node->src[0];
    const ggml_tensor * copy_dst  = cpy_node->src[1];

    const int64_t total = ggml_nelements(concat_node);
    const int64_t blocks = (total + CUDA_CONCAT_BLOCK_SIZE - 1)/CUDA_CONCAT_BLOCK_SIZE;
    GGML_ASSERT(blocks <= INT_MAX);

    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params((dim3) blocks, CUDA_CONCAT_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(concat_cpy_dim0_f32, launch_params,
        (const float *) src0->data, (const float *) src1->data, (float *) concat_node->data,
        (float *) copy_dst->data,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        concat_node->ne[0], concat_node->ne[1], concat_node->ne[2], concat_node->ne[3],
        (int64_t) copy_src->view_offs,
        copy_src->ne[0], copy_src->ne[1], copy_src->ne[2], copy_src->ne[3],
        (int64_t) copy_src->nb[0], (int64_t) copy_src->nb[1],
        (int64_t) copy_src->nb[2], (int64_t) copy_src->nb[3],
        copy_dst->ne[0], copy_dst->ne[1], copy_dst->ne[2], copy_dst->ne[3],
        (int64_t) copy_dst->nb[0], (int64_t) copy_dst->nb[1],
        (int64_t) copy_dst->nb[2], (int64_t) copy_dst->nb[3],
        ggml_nelements(copy_src));
}
