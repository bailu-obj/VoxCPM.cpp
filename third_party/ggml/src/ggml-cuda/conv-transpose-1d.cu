#include "conv-transpose-1d.cuh"

template <typename T>
static __device__ __forceinline__ float conv_transpose_1d_load_weight(const T * ptr) {
    return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float conv_transpose_1d_load_weight<half>(const half * ptr) {
    return __half2float(*ptr);
}

template <typename T>
static __global__ void conv_transpose_1d_kernel(
        const int s0,
        const int K,
        const int Cout,
        const int Cin,
        const int L,
        const int KL,
        const int nb01,
        const int nb02,
        const int nb11,
        const int nb12,
        const int nb1,
        const int nb2,
        const T     * __restrict__ src0,
        const float * __restrict__ src1,
              float * __restrict__ dst) {
    extern __shared__ float tmp[];

    const int tid = threadIdx.x;
    const int cout_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int bs = blockDim.x;
    const int tmp_len = bs * s0 + K;
    const int L_blocks = (L + bs - 1) / bs;

    const T * weight = src0;
    const float * input = src1 + batch_idx * nb12;
    float * output = dst + batch_idx * nb2 + cout_idx * nb1;

    for (int idx = tid; idx < tmp_len; idx += bs) {
        tmp[idx] = 0.0f;
    }
    __syncthreads();

    for (int block = 0; block < L_blocks; ++block) {
        if (block > 0) {
            for (int idx = tid; idx < tmp_len; idx += bs) {
                if (idx >= bs * s0) {
                    tmp[idx - bs * s0] = tmp[idx];
                    tmp[idx] = 0.0f;
                } else if (idx >= K) {
                    tmp[idx] = 0.0f;
                }
            }
            __syncthreads();
        }

        const int input_idx = block * bs + tid;
        const bool valid_input = input_idx < L;
        for (int k = 0; k < K; ++k) {
            float acc = 0.0f;
            if (valid_input) {
                for (int cin_idx = 0; cin_idx < Cin; ++cin_idx) {
                    const T * weight_ptr = weight + k + cout_idx * nb01 + cin_idx * nb02;
                    const float input_val = input[input_idx + cin_idx * nb11];
                    acc += conv_transpose_1d_load_weight(weight_ptr) * input_val;
                }
            }
            tmp[tid * s0 + k] += acc;
            __syncthreads();
        }

        const int base_kl = block * bs * s0;
        if (block < L_blocks - 1 && tid < bs) {
            for (int s_idx = 0; s_idx < s0; ++s_idx) {
                const int kl_idx = base_kl + tid * s0 + s_idx;
                if (kl_idx < KL) {
                    output[kl_idx] = tmp[tid * s0 + s_idx];
                }
            }
        }
    }

    const int tail_base = (L_blocks - 1) * bs * s0;
    for (int idx = tid; idx < tmp_len; idx += bs) {
        const int kl_idx = tail_base + idx;
        if (kl_idx < KL) {
            output[kl_idx] = tmp[idx];
        }
    }
}

template <typename T>
static void conv_transpose_1d_cuda(
        const int s0,
        const int K,
        const int Cout,
        const int Cin,
        const int L,
        const int B,
        const int KL,
        const int nb01,
        const int nb02,
        const int nb11,
        const int nb12,
        const int nb1,
        const int nb2,
        const T     * src0,
        const float * src1,
              float * dst,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims(Cout, B, 1);
    const size_t shared_bytes = static_cast<size_t>(block_dims.x * s0 + K) * sizeof(float);

    conv_transpose_1d_kernel<T><<<grid_dims, block_dims, shared_bytes, stream>>>(
        s0, K, Cout, Cin, L, KL, nb01, nb02, nb11, nb12, nb1, nb2, src0, src1, dst);
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int s0 = opts[0];

    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(src0->ne[0] > 0);
    GGML_ASSERT(src0->ne[1] > 0);
    GGML_ASSERT(src0->ne[2] > 0);
    GGML_ASSERT(src1->ne[0] > 0);
    GGML_ASSERT(src1->ne[2] > 0);

    const int K = static_cast<int>(src0->ne[0]);
    const int Cout = static_cast<int>(src0->ne[1]);
    const int Cin = static_cast<int>(src0->ne[2]);
    const int L = static_cast<int>(src1->ne[0]);
    const int B = static_cast<int>(src1->ne[2]);
    const int KL = static_cast<int>(dst->ne[0]);

    cudaStream_t stream = ctx.stream();

    if (src0->type == GGML_TYPE_F16) {
        conv_transpose_1d_cuda<half>(
            s0,
            K,
            Cout,
            Cin,
            L,
            B,
            KL,
            static_cast<int>(src0->nb[1] / sizeof(half)),
            static_cast<int>(src0->nb[2] / sizeof(half)),
            static_cast<int>(src1->nb[1] / sizeof(float)),
            static_cast<int>(src1->nb[2] / sizeof(float)),
            static_cast<int>(dst->nb[1] / sizeof(float)),
            static_cast<int>(dst->nb[2] / sizeof(float)),
            static_cast<const half *>(src0->data),
            static_cast<const float *>(src1->data),
            static_cast<float *>(dst->data),
            stream);
    } else {
        conv_transpose_1d_cuda<float>(
            s0,
            K,
            Cout,
            Cin,
            L,
            B,
            KL,
            static_cast<int>(src0->nb[1] / sizeof(float)),
            static_cast<int>(src0->nb[2] / sizeof(float)),
            static_cast<int>(src1->nb[1] / sizeof(float)),
            static_cast<int>(src1->nb[2] / sizeof(float)),
            static_cast<int>(dst->nb[1] / sizeof(float)),
            static_cast<int>(dst->nb[2] / sizeof(float)),
            static_cast<const float *>(src0->data),
            static_cast<const float *>(src1->data),
            static_cast<float *>(dst->data),
            stream);
    }
}
