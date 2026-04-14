#include <cmath>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/types.h>
#include <torch/extension.h>

#include "FlashAttention.h"

__global__ void backwardKernel(const float* Q, const float* K, const float* V,   //Query, Key, Value
                                const float* O, const float* dO,                 //output, output gradient
                                const float* L, const float* D,                  // logsumexp, normalization terms
                                float* dQ, float* dK, float* dV,                 // Query, Key, Value gradients
                                const unsigned int N, const unsigned int d,      //dimensions of QKV matrices [N, d]
                                const unsigned int Br, const unsigned int Bc,    //block sizes [Br, Bc]
                                const unsigned int Tr, const unsigned int Tc,    //number of blocks per row and column
                                const float attentionScalar)                     //scale the Q*K inside of the softmax function with this          
{
    int r = threadIdx.x;   // row in Br tile
    int c = threadIdx.y;   // col in Bc tile

    int j = blockIdx.x;    // column tile index
    int h = blockIdx.y;    // head
    int b = blockIdx.z;    // batch

    int qkv_offset = ((b * gridDim.y) + h) * N * d;
    int ld_offset  = ((b * gridDim.y) + h) * N;

    int col_base = j * Bc;
    int global_col = col_base + c;
    int col_offset_qkv = qkv_offset + col_base * d;

    extern __shared__ float sram[];

    int q_tile_size  = Br * d;    // Qi, Oi, dOi, dQi
    int kv_tile_size = Bc * d;    // Kj, Vj, dKj, dVj
    int s_tile_size  = Br * Bc;   // Sij, Pij, dPij, dSij
    int vec_tile_size = Br;       // Li, Di

    float* Qi  = sram;
    float* Kj  = &Qi[q_tile_size];
    float* Vj  = &Kj[kv_tile_size];
    float* Oi  = &Vj[kv_tile_size];
    float* dOi = &Oi[q_tile_size];
    float* dQi = &dOi[q_tile_size];
    float* Li  = &dQi[q_tile_size];
    float* Di  = &Li[vec_tile_size];
    float* dKj = &Di[vec_tile_size];
    float* dVj = &dKj[kv_tile_size];
    float* Sij  = &dVj[kv_tile_size];
    float* Pij  = &Sij[s_tile_size];
    float* dPij = &Pij[s_tile_size];
    float* dSij = &dPij[s_tile_size];

    __syncthreads();

    // Load Kj and Vj into SRAM
    if (r < Bc && c == 0) 
    {
        if (col_base + r < N) 
        {
            for (int x = 0; x < d; x++) 
            {
                Kj[r * d + x] = K[col_offset_qkv + r * d + x];
                Vj[r * d + x] = V[col_offset_qkv + r * d + x];
            }
        }
        else 
        {
            for (int x = 0; x < d; x++) 
            {
                Kj[r * d + x] = 0.0f;
                Vj[r * d + x] = 0.0f;
            }
        }
    }

    if (r < Bc && c == 0) 
    {
        for (int x = 0; x < d; x++) 
        {
            dKj[r * d + x] = 0.0f;
            dVj[r * d + x] = 0.0f;
        }
    }

    __syncthreads();

    // Loop over row tiles
    for (int i = 0; i < Tr; i++)
    {
        int row_base = i * Br;
        int global_row = row_base + r;
        int row_offset_qkv = qkv_offset + row_base * d;
        int row_offset_ld  = ld_offset + row_base;

        // Initialize dQi
        if (r < Br && c == 0)
        {
            for (int x = 0; x < d; x++)
            {
                dQi[r * d + x] = 0.0f;
            }
        }

        // Load Qi, Oi, dOi, Li, Di to SRAM
        if (r < Br && c == 0) 
        {
            if (global_row < N) 
            {
                for (int x = 0; x < d; x++) 
                {
                    Qi[r * d + x]  = Q[row_offset_qkv + r * d + x];
                    Oi[r * d + x]  = O[row_offset_qkv + r * d + x];
                    dOi[r * d + x] = dO[row_offset_qkv + r * d + x];
                    //dQi[r * d + x] = dQ[row_offset_qkv + r * d + x];
                }

                Li[r] = L[row_offset_ld + r];
                Di[r] = D[row_offset_ld + r];

            }
            else 
            {
                for (int x = 0; x < d; x++) 
                {
                    Qi[r * d + x]  = 0.0f;
                    Oi[r * d + x]  = 0.0f;
                    dOi[r * d + x] = 0.0f;
                    //dQi[r * d + x] = 0.0f;
                }

                Li[r] = 0.0f;
                Di[r] = 0.0f;
            }
        }

        __syncthreads();

        // Compute Sij = Qi * Kj^T
        if (r < Br && c < Bc) 
        {
            if (global_row < N && global_col < N) 
            {
                float sum = 0.0f;
                for (int x = 0; x < d; x++) 
                {
                    sum += Qi[r * d + x] * Kj[c * d + x];
                }

                Sij[r * Bc + c] = attentionScalar * sum;
            } 
            else
            {
                Sij[r * Bc + c] = -INFINITY;
            }
        }

        __syncthreads();

        // Compute Pij = exp(Sij - Li)
        if (r < Br && c < Bc) 
        {
            if (global_row < N && global_col < N) 
            {
                Pij[r * Bc + c] = __expf(Sij[r * Bc + c] - Li[r]);
            } 
            else 
            {
                Pij[r * Bc + c] = 0.0f;
            }
        }

        __syncthreads();

        // Compute dVj = dVj + (Pij)^T * dOi
        if (r < Bc && c == 0) 
        {
            if (col_base + r < N) 
            {
                for (int x = 0; x < d; x++) 
                {
                    float sum = 0.0f;
                    for (int rr = 0; rr < Br; rr++) 
                    {
                        int g_row = row_base + rr;
                        if (g_row < N) 
                        {
                            sum += Pij[rr * Bc + r] * dOi[rr * d + x];
                        }
                    }
                    
                    dVj[r * d + x] += sum;
                }
            }
        }

        // Compute dPij = dOi * Vj^T
        if (r < Br && c < Bc) 
        {
            if (global_row < N && global_col < N)
            {
                float sum = 0.0f;
                for (int x = 0; x < d; x++)
                {
                    sum += dOi[r * d + x] * Vj[c * d + x];
                }
                dPij[r * Bc + c] = sum;
            }
            else 
            {
                dPij[r * Bc + c] = 0.0f;
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
                dSij[r * Bc + c] = 0.0f;
            }
        }

        __syncthreads();

        // dQi = dQi + dSij * Kj
        if (r < Br && c == 0) 
        {
            for (int x = 0; x < d; x++) 
            {
                float sum = 0.0f;
                for (int cc = 0; cc < Bc; cc++) 
                {
                    int g_col = col_base + cc;
                    if (g_col < N)
                    {
                        sum += dSij[r * Bc + cc] * Kj[cc * d + x];
                    }
                }

                dQi[r * d + x] += attentionScalar * sum;
            }
        }

        __syncthreads();

        // Write dQi to HBM
        if (r < Br && c == 0) 
        {
            if (global_row < N) 
            {
                for (int x = 0; x < d; x++) 
                {
                    atomicAdd(&dQ[row_offset_qkv + r * d + x], dQi[r * d + x]);
                }
            }
        }

        // Compute dKj = dKj + (dSij)^T * Qi
        if (r < Bc && c == 0) 
        {
            if (col_base + r < N) 
            {
                for (int x = 0; x < d; x++) 
                {
                    float sum = 0.0f;
                    for (int rr = 0; rr < Br; rr++) 
                    {
                        int g_row = row_base + rr;
                        if (g_row < N) 
                        {
                            sum += dSij[rr * Bc + r] * Qi[rr * d + x];
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
    if (r < Bc && c == 0) 
    {
        if (col_base + r < N) 
        {
            for (int x = 0; x < d; x++) 
            {
                dK[col_offset_qkv + r * d + x] = dKj[r * d + x];
                dV[col_offset_qkv + r * d + x] = dVj[r * d + x];
            }
        }
    }
}

std::vector<torch::Tensor> backward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
                                    torch::Tensor O, torch::Tensor dO, torch::Tensor L)
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
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "V must be float32");
    TORCH_CHECK(O.scalar_type() == torch::kFloat32, "O must be float32");
    TORCH_CHECK(dO.scalar_type() == torch::kFloat32, "dO must be float32");

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

    const int batchSize = Q.size(0);
    const int numHeads = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Bc = 16;
    const int Br = 16;

    const int Tr = std::ceil((float)N/Br);  // Divide Q, dQ, D into Tr blocks
    const int Tc = std::ceil((float)N/Bc);  // Divide K, V, dK, dV into N/Bc blocks

    const float attentionScalar = 1.0f/std::sqrt((float)d);

    // Calculate SRAM size needed per block
    // [Bc x d]  : Kj, Vj, dKj, dVj
    // [Br x d]  : Qi, Oi, dOi, dQi
    // [Br x Bc] : Sij, Pij, dPij, dSij (Temps)
    // [Br]      : Li, Di
    const int sramSize = ((4 * Bc * d) + (4 * Br * d) + (4 * Br * Bc) + (2 * Br)) * sizeof(float);
    int maxSramSize;
    cudaDeviceGetAttribute(&maxSramSize, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    
    if (maxSramSize < sramSize)
    {
        throw std::runtime_error("Requested shared memory exceeds device limit");
    }

    auto opts = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());

    torch::Tensor dQ = torch::zeros_like(Q);
    torch::Tensor dK = torch::zeros_like(K);
    torch::Tensor dV = torch::zeros_like(V);
    torch::Tensor D = torch::sum(O * dO, -1);

    D = torch::mul(O, dO);

    dim3 gridDim(Tc, numHeads, batchSize); // (Tr x numHeads x batchSize) blocks per grid
    dim3 blockDim(Br, Bc);                 // (Br x Bc) threads per block
    
    // Launch kernel
    backwardKernel<<<gridDim, blockDim, sramSize>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        dO.data_ptr<float>(),
        L.data_ptr<float>(),
        D.data_ptr<float>(),
        dQ.data_ptr<float>(),
        dK.data_ptr<float>(),
        dV.data_ptr<float>(),
        N, d, Br, Bc, Tr, Tc,
        attentionScalar
    ) 

    return {dQ, dK, dV};

}