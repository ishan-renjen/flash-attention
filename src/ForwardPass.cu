#include "FlashAttention.h"

#define MAX_D 64

#include <cmath>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <torch/types.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define D 64

__global__ void forwardKernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V, 
                                    __nv_bfloat16* __restrict__ O, __nv_bfloat16* __restrict__ LSE, 
                                    const int N, const int Tc, const float scale,
                                    int Br, int Bc) {
    const int b  = blockIdx.x;
    const int h  = blockIdx.y;
    const int qi = blockIdx.z;
    const int tx = threadIdx.x;

    const int q_row = qi * Br + tx;
    const bool valid_q = (q_row < N);

    const int bh = b * gridDim.y + h;
    const int qkv_base = bh * N * D;
    const int lse_base = bh * N;

    extern __shared__ __nv_bfloat16 sram[];

    __nv_bfloat16* sQ = sram;                 // [Br, D]
    __nv_bfloat16* sK = sQ + Br * D;          // [Bc, D]
    __nv_bfloat16* sV = sK + Bc * D;          // [Bc, D]
    __nv_bfloat16* sS = sV + Bc * D;          // [Br, Bc]

    //load Q tile
    for (int idx = tx; idx < Br * D; idx += blockDim.x) {
        const int local_row = idx / D;
        const int local_col = idx % D;
        const int global_row = qi * Br + local_row;

        if (global_row < N) {
            sQ[local_row * D + local_col] =
                Q[qkv_base + global_row * D + local_col];
        } else {
            sQ[local_row * D + local_col] = (__nv_bfloat16)0.0f;
        }
    }
    __syncthreads();

    float m = -INFINITY;
    float l = 0.0f;

    float o_accum[D];
    #pragma unroll
    for (int x = 0; x < D; ++x) {
        o_accum[x] = 0.0f;
    }

    #pragma unroll
    for (int j = 0; j < Tc; ++j) {
        // Load K and V tile
        #pragma unroll
        for (int idx = tx; idx < Bc * D; idx += blockDim.x) {
            const int local_row = idx / D;
            const int local_col = idx % D;
            const int global_row = j * Bc + local_row;

            if (global_row < N) {
                sK[local_row * D + local_col] =
                    K[qkv_base + global_row * D + local_col];
                sV[local_row * D + local_col] =
                    V[qkv_base + global_row * D + local_col];
            } else {
                sK[local_row * D + local_col] = (__nv_bfloat16)0.0f;
                sV[local_row * D + local_col] = (__nv_bfloat16)0.0f;
            }
        }
        __syncthreads();

        if (valid_q) {
            float row_max_local = -INFINITY;

            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                const int k_row = j * Bc + c;

                float score = -INFINITY;
                if (k_row < N) {
                    score = 0.0f;
                    #pragma unroll
                    for (int x = 0; x < D; ++x) {
                        score += __bfloat162float(sQ[tx * D + x]) * __bfloat162float(sK[c * D + x]);
                    }
                    score *= scale;
                }

                sS[tx * Bc + c] = __float2bfloat16(score);
                row_max_local = fmaxf(row_max_local, score);
            }
            
            const float m_new = fmaxf(m, row_max_local);
            const float alpha = __expf(m - m_new);
            float l_new = alpha * l;

            #pragma unroll
            for (int x = 0; x < D; ++x) {
                o_accum[x] *= alpha;
            }

            //softmax
            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                const int k_row = j * Bc + c;
                if (k_row < N) {
                    const float p = __expf(__bfloat162float(sS[tx * Bc + c]) - m_new);
                    l_new += p;

                    #pragma unroll
                    for (int x = 0; x < D; ++x) {
                        o_accum[x] += p * __bfloat162float(sV[c * D + x]);
                    }
                }
            }

            m = m_new;
            l = l_new;
        }

        __syncthreads();
    }

    if (valid_q) {
        const float inv_l = 1.0f / l;

        #pragma unroll
        for (int x = 0; x < D; ++x) {
            O[qkv_base + q_row * D + x] = __float2bfloat16(o_accum[x] * inv_l);
        }

        LSE[lse_base + q_row] = __float2bfloat16(m + __logf(l));
    }
}

void forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, torch::Tensor LSE) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");

    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    TORCH_CHECK(Q.dim() == 4, "Q must have shape [B, H, N, d]");
    TORCH_CHECK(K.dim() == 4, "K must have shape [B, H, N, d]");
    TORCH_CHECK(V.dim() == 4, "V must have shape [B, H, N, d]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q/K/V shape mismatch");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    constexpr int Br = 16;
    constexpr int Bc = 16;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    dim3 grid(B, H, Tr);
    dim3 block(Br);

    auto Q_ptr   = reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>());
    auto K_ptr   = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>());
    auto V_ptr   = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>());
    auto O_ptr   = reinterpret_cast<__nv_bfloat16*>(O.data_ptr<at::BFloat16>());
    auto LSE_ptr = reinterpret_cast<__nv_bfloat16*>(LSE.data_ptr<at::BFloat16>());

    const size_t smem = (Br * 64 + 2 * Bc * 64 + Br * Bc) * sizeof(__nv_bfloat16);
    forwardKernel<<<grid, block, smem>>>(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        LSE_ptr,
        N,
        Tc,
        scale,
        Br,
        Bc);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
