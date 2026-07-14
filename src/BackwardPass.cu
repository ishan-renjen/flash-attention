#include <cmath>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <torch/types.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#include "FlashAttention.h"

__global__ void backwardKernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,   //Query, Key, Value
                                const __nv_bfloat16* __restrict__ O, const __nv_bfloat16* __restrict__ dO,                 //output, output gradient
                                const float* __restrict__ L, const __nv_bfloat16* __restrict__ D,                  // logsumexp, normalization terms
                                __nv_bfloat16* __restrict__ dQ, __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV,                 // Query, Key, Value gradients
                                const unsigned int N, const unsigned int d,      //dimensions of QKV matrices [N, d]
                                const unsigned int Br, const unsigned int Bc,    //block sizes [Br, Bc]
                                const unsigned int Tr, const unsigned int Tc,    //number of blocks per row and column
                                const float attentionScalar)                     //scale the Q*K inside of the softmax function with this          
{
    int r = threadIdx.x;   // row index
    int c = threadIdx.y;   // col index

    // One block per K/V col tile (j), head, and batch element
    int j = blockIdx.x;    // column tile index for K/V
    int h = blockIdx.y;    // head
    int b = blockIdx.z;    // batch

    // (b * gridDim.y) + h --> (batchNum * numHeads) + head
    // N x d elements in Q, K, and V per head
    int qkv_offset = ((b * gridDim.y) + h) * N * d;

    // (b * gridDim.y) + h --> (batchNum * numHeads) + head
    // N elements in L and D per head
    int ld_offset  = ((b * gridDim.y) + h) * N;
    
    int col_base = j * Bc;                          // Starting column tile index
    int global_col = col_base + c;                  // starting index + offset of current thread
    int col_offset_qkv = qkv_offset + col_base * d; // Base address of tile 

    extern __shared__ unsigned char sram_raw[];

    int q_tile_size  = Br * d;    // Qi, Oi, dOi, dQi
    int kv_tile_size = Bc * d;    // Kj, Vj, dKj, dVj
    int s_tile_size  = Br * Bc;   // Sij, Pij, dPij, dSij
    int vec_tile_size = Br;       // Li, Di

    float* Li = reinterpret_cast<float*>(sram_raw);

    __nv_bfloat16* Qi  = reinterpret_cast<__nv_bfloat16*>(Li + vec_tile_size);
    __nv_bfloat16* Kj  = &Qi[q_tile_size];
    __nv_bfloat16* Vj  = &Kj[kv_tile_size];
    __nv_bfloat16* dOi = &Vj[q_tile_size];
    __nv_bfloat16* dQi = &dOi[q_tile_size];
    __nv_bfloat16* Di  = &dQi[vec_tile_size];
    __nv_bfloat16* dKj = &Di[vec_tile_size];
    __nv_bfloat16* dVj = &dKj[kv_tile_size];
    __nv_bfloat16* Sij  = &dVj[kv_tile_size];
    __nv_bfloat16* Pij  = &Sij[s_tile_size];
    __nv_bfloat16* dPij = &Pij[s_tile_size];
    __nv_bfloat16* dSij = &dPij[s_tile_size];

    // Kj and Vj are [Bc, d] --> 0 <= r < Bc
    if (r < Bc) 
    {
        // Load Kj and Vj into SRAM & initialize dKj/dVj
        for (int x = c; x < d; x += Bc) 
        {
            // Check for valid sequence position
            if (col_base + r < N) 
            {
                // (r * d) + x = (tile row * head dim) + col offset
                Kj[r * d + x] = K[col_offset_qkv + r * d + x];
                Vj[r * d + x] = V[col_offset_qkv + r * d + x];
            } 
            else 
            {
                Kj[r * d + x] = (__nv_bfloat16)0.0f;
                Vj[r * d + x] = (__nv_bfloat16)0.0f;
            }

            dKj[r * d + x] = (__nv_bfloat16)0.0f;
            dVj[r * d + x] = (__nv_bfloat16)0.0f;
        }
    }

    __syncthreads();

    // Loop over row tiles (Tr row tiles, Br rows per tile)
    #pragma unroll
    for (int i = 0; i < Tr; i++)
    {
        int row_base = i * Br;                           // starting row tile index 
        int global_row = row_base + r;                   // starting row tile index + row offset of thread
        int row_offset_qkv = qkv_offset + row_base * d;  // starting memory address of Q/O/dO per row tile
        int row_offset_ld  = ld_offset + row_base;       // starting memory address of L/D per row tile

        // Check for valid row index
        if (r < Br) 
        {
            // Load Qi, Oi, dOi, Li, Di to SRAM & initialize dQi
            for (int x = c; x < d; x += Bc) 
            {
                // Check for valid sequence position
                if (global_row < N) 
                {
                    Qi[r * d + x]  = Q[row_offset_qkv + r * d + x];
                    // Oi[r * d + x]  = O[row_offset_qkv + r * d + x];
                    // dOi[r * d + x] = dO[row_offset_qkv + r * d + x];
                } 
                else 
                {
                    Qi[r * d + x]  = (__nv_bfloat16)0.0f;
                    // Oi[r * d + x]  = (__nv_bfloat16)0.0f;
                    dOi[r * d + x] = (__nv_bfloat16)0.0f;
                }

                dQi[r * d + x] = (__nv_bfloat16)0.0f;
            }

            // Li and Di have 1 scalar per row --> load using only 1 thread
            if (c == 0) 
            {
                if (global_row < N) 
                {
                    Li[r] = L[row_offset_ld + r];
                    Di[r] = D[row_offset_ld + r];
                } 
                else 
                {
                    Li[r] = 0.0f;
                    Di[r] = (__nv_bfloat16)0.0f;
                }
            }
        }

        __syncthreads();

        // Compute Sij = Qi * Kj^T
        // Check for valid row and col indices
        if (r < Br && c < Bc) 
        {
            // Check for valid sequence positions
            if (global_row < N && global_col < N) 
            {
                float sum = (__nv_bfloat16)0.0f;
                #pragma unroll
                for (int x = 0; x < d; x++) 
                {
                    // Qi(Kj)^T
                    sum += __bfloat162float(Qi[r * d + x]) * __bfloat162float(Kj[c * d + x]);
                }

                // Apply scaling
                Sij[r * Bc + c] = attentionScalar * sum;
            } 
            else
            {
                Sij[r * Bc + c] = -INFINITY;
            }
        }

        __syncthreads();

        // Compute Pij = exp(Sij - Li)
        // Check that the thread row and col indices are within the tile
        if (r < Br && c < Bc) 
        {
            // Check for threads that are out-of-bounds
            if (global_row < N && global_col < N) 
            {
                Pij[r * Bc + c] = __expf(__bfloat162float(Sij[r * Bc + c]) - Li[r]);
            } 
            else 
            {
                Pij[r * Bc + c] = (__nv_bfloat16)0.0f;
            }
        }

        __syncthreads();

        // Compute dVj = dVj + (Pij)^T * dOi
        if (r < Bc) 
        {
            int global_kv_row = col_base + r;

            if (global_kv_row < N) 
            {
                for (int x = c; x < d; x += Bc) 
                {
                    __nv_bfloat16 sum = (__nv_bfloat16)0.0f;

                    for (int rr = 0; rr < Br; rr++) 
                    {
                        int global_q_row = row_base + rr;

                        if (global_q_row < N) 
                        {
                            sum += Pij[rr * Bc + r] * dOi[rr * d + x];
                        }
                    }

                    dVj[r * d + x] += sum;
                }
            }
        }

        __syncthreads();

        // Compute dPij = dOi * Vj^T
        if (r < Br && c < Bc) 
        {
            if (global_row < N && global_col < N)
            {
                float sum = (__nv_bfloat16)0.0f;
                #pragma unroll
                for (int x = 0; x < d; x++)
                {
                    sum += __bfloat162float(dOi[r * d + x]) * __bfloat162float(Vj[c * d + x]);
                }
                dPij[r * Bc + c] = sum;
            }
            else 
            {
                dPij[r * Bc + c] = (__nv_bfloat16)0.0f;
            }
        }

        __syncthreads();

        // Compute dSij = Pij o (dPij - Di)
        if (r < Br && c < Bc) 
        {
            if (global_row < N && global_col < N) 
            {
                dSij[r * Bc + c] = Pij[r * Bc + c] * (dPij[r * Bc + c] - Di[r]);
            }
            else
            {
                dSij[r * Bc + c] = (__nv_bfloat16)0.0f;
            }
        }

        __syncthreads();

        // dQi = dQi + dSij * Kj
        if (r < Br) 
        {
            for (int x = c; x < d; x += Bc) 
            {
                float sum = 0.0f;

                for (int cc = 0; cc < Bc; cc++) 
                {
                    int g_col = col_base + cc;
                    if (g_col < N) 
                    {
                        sum += __bfloat162float(dSij[r * Bc + cc]) * __bfloat162float(Kj[cc * d + x]);
                    }
                }

                dQi[r * d + x] += attentionScalar * sum;
            }
        }

        __syncthreads();

        // Write dQi to HBM
        if (r < Br && global_row < N) 
        {
            for (int x = c; x < d; x += Bc) 
            {
                atomicAdd(&dQ[row_offset_qkv + r * d + x], __bfloat162float(dQi[r * d + x]));
            }
        }

        // Compute dKj = dKj + (dSij)^T * Qi
        if (r < Bc) 
        {
            for (int x = c; x < d; x += Bc) 
            {
                if (col_base + r < N) 
                {
                    float sum = (__nv_bfloat16)0.0f;

                    for (int rr = 0; rr < Br; rr++) 
                    {
                        int g_row = row_base + rr;
                        if (g_row < N) 
                        {
                            sum += __bfloat162float(dSij[rr * Bc + r]) * __bfloat162float(Qi[rr * d + x]);
                        }
                    }

                    dKj[r * d + x] += attentionScalar * sum;
                }
            }
        }
        __syncthreads();
    }

    __syncthreads();

    // Write dKj and dVj to HBM
    if (r < Bc && col_base + r < N) 
    {
        for (int x = c; x < d; x += Bc) 
        {
            dK[col_offset_qkv + r * d + x] = dKj[r * d + x];
            dV[col_offset_qkv + r * d + x] = dVj[r * d + x];
        }
    }

}

void backward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
                                    torch::Tensor O, torch::Tensor dO, torch::Tensor L, 
                                    torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV, 
                                    torch::Tensor D)
{
    //must be on GPU to use
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");
    TORCH_CHECK(O.is_cuda(), "O must be CUDA");
    TORCH_CHECK(dO.is_cuda(), "dO must be CUDA");

    //ideally they are contiguous
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(O.is_contiguous(), "O must be contiguous");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous");

    //check dimensionality of Q, K, V
    TORCH_CHECK(Q.dim() == 4, "Q must have shape [B, H, N, d]");
    TORCH_CHECK(K.dim() == 4, "K must have shape [B, H, N, d]");
    TORCH_CHECK(V.dim() == 4, "V must have shape [B, H, N, d]");
    TORCH_CHECK(O.dim() == 4, "O must have shape [B, H, N, d]");
    TORCH_CHECK(dO.dim() == 4, "dO must have shape [B, H, N, d]");

    //check the dtype of Q, K, V
    TORCH_CHECK(Q.scalar_type() == torch::kBFloat16, "Q must be kBFloat16");
    TORCH_CHECK(K.scalar_type() == torch::kBFloat16, "K must be kBFloat16");
    TORCH_CHECK(V.scalar_type() == torch::kBFloat16, "V must be kBFloat16");
    TORCH_CHECK(O.scalar_type() == torch::kBFloat16, "O must be kBFloat16");
    TORCH_CHECK(dO.scalar_type() == torch::kBFloat16, "dO must be kBFloat16");

    //check that Q, K, V dimensions match
    TORCH_CHECK(K.size(0) == Q.size(0) && V.size(0) == Q.size(0), "(Q,K,V) - Batch size mismatch");
    TORCH_CHECK(K.size(1) == Q.size(1) && V.size(1) == Q.size(1), "(Q,K,V) - Head count mismatch");
    TORCH_CHECK(K.size(2) == Q.size(2) && V.size(2) == Q.size(2), "(Q,K,V) - Sequence length mismatch");
    TORCH_CHECK(K.size(3) == Q.size(3) && V.size(3) == Q.size(3), "(Q,K,V) - Head dimension mismatch");

    // check that dO and O dimensions match
    TORCH_CHECK(O.size(0) == dO.size(0) && O.size(0) == Q.size(0), "(O,dO) - Batch size mismatch");
    TORCH_CHECK(O.size(1) == dO.size(1) && O.size(1) == Q.size(1), "(O,dO) - Head count mismatch");
    TORCH_CHECK(O.size(2) == dO.size(2) && O.size(2) == Q.size(2), "(O,dO) - Sequence length mismatch");
    TORCH_CHECK(O.size(3) == dO.size(3) && O.size(3) == Q.size(3), "(O,dO) - Head dimension mismatch");

    dQ.zero_(); 

    const int batchSize = Q.size(0);
    const int numHeads = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Bc = 16;
    const int Br = 16;

    const int Tr = std::ceil((float)N/Br);  // Divide Q, dQ, D into N/Br blocks
    const int Tc = std::ceil((float)N/Bc);  // Divide K, V, dK, dV into N/Bc blocks

    const float attentionScalar = 1.0f/std::sqrt((float)d);

    // Calculate SRAM size needed per block
    // [Bc x d]  : Kj, Vj, dKj, dVj
    // [Br x d]  : Qi, Oi, dOi, dQi
    // [Br x Bc] : Sij, Pij, dPij, dSij (Temps)
    // [Br]      : Li, Di
    const int sramSize = Br * sizeof(float) + ((3 * Br * d) + (4 * Bc * d) + (4 * Br * Bc) + Br) * sizeof(__nv_bfloat16);
    int maxSramSize;
    int dev = Q.get_device();
    cudaDeviceGetAttribute(&maxSramSize, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    
    if (maxSramSize < sramSize)
    {
        throw std::runtime_error("Requested shared memory exceeds device limit");
    }

    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto dQ_fp32 = torch::zeros({batchSize, numHeads, N, d}, opts_fp32);

    dim3 gridDim(Tc, numHeads, batchSize); // (Tc x numHeads x batchSize) blocks per grid
    dim3 blockDim(Br, Bc);                 // (Br x Bc) threads per block
    
    // Launch kernel
    const auto* Q_ptr = reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>());
    const auto* K_ptr = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>());
    const auto* V_ptr = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>());
    const auto* O_ptr = reinterpret_cast<const __nv_bfloat16*>(O.data_ptr<at::BFloat16>());
    const auto* dO_ptr = reinterpret_cast<const __nv_bfloat16*>(dO.data_ptr<at::BFloat16>());
    const auto* L_ptr = reinterpret_cast<const float*>(L.data_ptr<at::BFloat16>());
    const auto* D_ptr = reinterpret_cast<const __nv_bfloat16*>(D.data_ptr<at::BFloat16>());
    auto* dQ_ptr = reinterpret_cast<__nv_bfloat16*>(dQ.data_ptr<at::BFloat16>());
    auto* dK_ptr = reinterpret_cast<__nv_bfloat16*>(dK.data_ptr<at::BFloat16>());
    auto* dV_ptr = reinterpret_cast<__nv_bfloat16*>(dV.data_ptr<at::BFloat16>());

    backwardKernel<<<gridDim, blockDim, sramSize>>>(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        dO_ptr,
        L_ptr,
        D_ptr,
        dQ_ptr,
        dK_ptr,
        dV_ptr,
        N,
        d,
        Br,
        Bc,
        Tr,
        Tc,
        attentionScalar);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dQ.copy_(dQ_fp32.to(Q.scalar_type()));
}