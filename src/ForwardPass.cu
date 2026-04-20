#include "FlashAttention.h"

#define MAX_D 64

#include <cmath>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <torch/types.h>
#include <torch/extension.h>

#define D 64

__global__ void forwardKernelSimple(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, 
                                    float* __restrict__ O, float* __restrict__ LSE, 
                                    const int N, const int Tc, const float scale,
                                    int Br, int Bc) {
    const int b  = blockIdx.x;
    const int h  = blockIdx.y;
    const int qi = blockIdx.z;
    const int tx = threadIdx.x;   // one thread = one query row in the tile

    const int q_row = qi * Br + tx;
    const bool valid_q = (q_row < N);

    const int bh = b * gridDim.y + h;
    const int qkv_base = bh * N * D;
    const int lse_base = bh * N;

    extern __shared__ float sram[];
    float* sQ = sram;                 // [Br, D]
    float* sK = sQ + Br * D;          // [Bc, D]
    float* sV = sK + Bc * D;          // [Bc, D]
    float* sS = sV + Bc * D;          // [Br, Bc]

    // Load Q tile once.
    for (int idx = tx; idx < Br * D; idx += blockDim.x) {
        const int local_row = idx / D;
        const int local_col = idx % D;
        const int global_row = qi * Br + local_row;

        if (global_row < N) {
            sQ[local_row * D + local_col] =
                Q[qkv_base + global_row * D + local_col];
        } else {
            sQ[local_row * D + local_col] = 0.0f;
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

    // Sweep through K/V tiles.
    #pragma unroll
    for (int j = 0; j < Tc; ++j) {
        // Load K and V tile cooperatively.
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
                sK[local_row * D + local_col] = 0.0f;
                sV[local_row * D + local_col] = 0.0f;
            }
        }
        __syncthreads();

        if (valid_q) {
            float row_max_local = -INFINITY;

            // First pass: compute scores and row max.
            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                const int k_row = j * Bc + c;

                float score = -INFINITY;
                if (k_row < N) {
                    score = 0.0f;
                    #pragma unroll
                    for (int x = 0; x < D; ++x) {
                        score += sQ[tx * D + x] * sK[c * D + x];
                    }
                    score *= scale;
                }

                sS[tx * Bc + c] = score;
                row_max_local = fmaxf(row_max_local, score);
            }
            
            const float m_new = fmaxf(m, row_max_local);
            const float alpha = __expf(m - m_new);
            float l_new = alpha * l;

            // Scale old accumulator once.
            #pragma unroll
            for (int x = 0; x < D; ++x) {
                o_accum[x] *= alpha;
            }

            // Second pass: softmax weights and PV accumulation.
            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                const int k_row = j * Bc + c;
                if (k_row < N) {
                    const float p = __expf(sS[tx * Bc + c] - m_new);
                    l_new += p;

                    #pragma unroll
                    for (int x = 0; x < D; ++x) {
                        o_accum[x] += p * sV[c * D + x];
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
            O[qkv_base + q_row * D + x] = o_accum[x] * inv_l;
        }

        LSE[lse_base + q_row] = m + __logf(l);
    }
}

std::vector<torch::Tensor> forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
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
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    constexpr int Br = 32;
    constexpr int Bc = 32;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    auto O = torch::empty_like(Q);
    auto LSE = torch::empty({B, H, N}, Q.options().dtype(torch::kFloat32));

    dim3 grid(B, H, Tr);
    dim3 block(Br);

    const size_t smem = (Br * 64 + 2 * Bc * 64 + Br * Bc) * sizeof(float);
    forwardKernelSimple<<<grid, block, smem>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        LSE.data_ptr<float>(),
        N,
        Tc,
        scale,
        Br,
        Bc);

    return {O, LSE};
}
