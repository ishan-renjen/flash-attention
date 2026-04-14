#include <cmath>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/types.h>
#include <torch/extension.h>

#include "FlashAttention.h"

#define MAX_D 64

//CUDA kernel for forward pass 
__global__ void forwardKernel(const float* Q, const float* K, const float* V, //Query, Key, Value [B, H, N, d]
                              float* O, float* LSE,                           //output, logsumexp tensors
                              const unsigned int N, const unsigned int d,     //dimensions of QKV matrices [N, d]
                              const unsigned int Br, const unsigned int Bc,   //block sizes [Br, Bc]
                              const unsigned int Tr, const unsigned int Tc,   //number of blocks per row and column
                              const float scale
                        ){    
  /*1.) Divide Q into Tr blocks [Br, d] and K, V into Tc blocks [Bc, d]
    2.) Divide O into Tr blocks, logsumexp into Tr blocks -> each Q has a matching O and logsumexp tile

    BASE CASE: per-row iteration
    index into tile Qi, load into SRAM
    set accum/Output accum = 0, row-wise sum l = 0, row-wise max m = -inf
    for j in range(Tc): (sweep through K/V tiles with fixed Q)
        index into tile Kj, Vj - load them into SRAM
        compute S_i,j = Qi*(Kj^T)
        compute row-wise max = max(prev_max, rowmax(S_i,j))
        compute P_tilde = exp(S_i,j - row-wise max)
        row-wise sum = e^(prev_max - row-wise max)*prev_row-wise sum + accum(P_tilde)
        output update = diag(e^(prev_max - row-wise max)) * prev_outputaccum + (P_tilde)*Vj
    Output tile = 1/(diag(row-wise sum)) * output update
    logsumexp tile = row-wise max + log(row-wise sum)

    write output tile, logsumexp tile to HBM*/

    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z;
    int tx = threadIdx.x;

    //current row for the thread
    int q_row = i * Br  + tx;

    extern __shared__ float sram[];

    float* sQ = sram;
    float* sK = sQ + Br * d; //store above sQ
    float* sV = sK + Bc * d //store above sV
    float* sS = sV + Bc * d;

    float max = -INFINITY;
    float scale_denom = 0.0f;

    //initialize accumulator to 0 to not hit weird issues if accum < d
    float accumulator[MAX_D];
    for(int i = 0; i < d; i++){
        accumulator[i] = 0.0f;
    }

    //iterating through Qi tile
    for(int i = tx; i < Br * d; i+=blockDim.x){
        int local_col = i % d;
        int global_row = i * Br + (i / d);

        //check for max
        if(global_row < N){
            sQ[i + local_col] = Q[((b * gridDim.y + h) * N + global_row) * d + local_col];
        }
        else{
            sQ[local_row * d + local_col] = 0.0f;
        }
    }
    __syncthreads();

    bool valid q_row = (q_row < N);

    //iterate over columns - K/V
    for (int j = 0; j < Tc; ++j) {
        for (int x = tx; x < Bc * d; x += blockDim.x) {
            int local_row = x / d;
            int local_col = x % d;
            int global_row = j * Bc + local_row;

            //check for max
            if (global_row < N) {
                sK[local_row * d + local_col] =
                    K[((b * gridDim.y + h) * N + global_row) * d + local_col];

                sV[local_row * d + local_col] =
                    V[((b * gridDim.y + h) * N + global_row) * d + local_col];
            } else {
                sK[local_row * d + local_col] = 0.0f;
                sV[local_row * d + local_col] = 0.0f;
            }
        }
        __syncthreads();

        //init m
        float row_max_local = -INFINITY;

        //calculate 1 row of unscaled softmax for QK^T
        if (valid_q_row) {
            for (int c = 0; c < Bc; ++c) {
                int k_row = j * Bc + c;

                float score = -INFINITY;
                if (k_row < N) {
                    score = 0.0f;
                    for (int x = 0; x < d; ++x) {
                        score += sQ[tx * d + x] * sK[c * d + x];
                    }
                    score *= scale;
                }

                sS[tx * Bc + c] = score;
                row_max_local = fmaxf(row_max_local, score);
            }
        }
        __syncthreads();

        //update M, P, L - compute new output
        if (valid_q_row) {
            float m_new = fmaxf(m, row_max_local);
            float alpha = __expf(m - m_new);

            float l_new = alpha * l;

            // temporary weighted sum for this row
            float pv_row[MAX_D];
            for (int x = 0; x < d; ++x) {
                pv_row[x] = 0.0f;
            }

            for (int c = 0; c < Bc; ++c) {
                int k_row = j * Bc + c;
                if (k_row < N) {
                    float p_tilde = __expf(sS[tx * Bc + c] - m_new);
                    l_new += p_tilde;

                    for (int x = 0; x < d; ++x) {
                        pv_row[x] += p_tilde * sV[c * d + x];
                    }
                }
            }

            for (int x = 0; x < d; ++x) {
                o_accum[x] = alpha * o_accum[x] + pv_row[x];
            }

            m = m_new;
            l = l_new;
        }

        __syncthreads();
    }

    //normalize, then writeback 
    if (valid_q_row) {
        for (int x = 0; x < d; ++x) {
            O[((b * gridDim.y + h) * N + q_row) * d + x] = o_accum[x] / l;
        }
        LSE[(b * gridDim.y + h) * N + q_row] = m + logf(l);
    }
}

std::vector<torch::Tensor> forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
    //must be on GPU to use
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");

    //ideally they are contiguous
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    //check dimensionality of Q, K, V
    TORCH_CHECK(Q.dim() == 4, "Q must have shape [B, H, N, d]");
    TORCH_CHECK(K.dim() == 4, "K must have shape [B, H, N, d]");
    TORCH_CHECK(V.dim() == 4, "V must have shape [B, H, N, d]");

    //check the dtype of Q, K, V
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "V must be float32");

    //check that Q, K, V dimensions match
    TORCH_CHECK(K.size(0) == Q.size(0) && V.size(0) == Q.size(0), "Batch size mismatch");
    TORCH_CHECK(K.size(1) == Q.size(1) && V.size(1) == Q.size(1), "Head count mismatch");
    TORCH_CHECK(K.size(2) == Q.size(2) && V.size(2) == Q.size(2), "Sequence length mismatch");
    TORCH_CHECK(K.size(3) == Q.size(3) && V.size(3) == Q.size(3), "Head dimension mismatch");

    const int batchSize = Q.size(0);
    const int numHeads  = Q.size(1);
    const int N         = Q.size(2);
    const int d         = Q.size(3);

    const int Bc = 16;
    const int Br = 16; //for RTX 4060 -> tile size of 256 per block, TODO: figure out what the best values for these are

    const int Tr                = std::ceil((float)N/Br);
    const int Tc                = std::ceil((float)N/Bc);
    float attentionScalar       = 1.0f / std::sqrt(d);

    torch::Tensor output    = torch::zeros_like(Q);
    torch::Tensor logsumexp = torch::zeros({batchSize, numHeads, N}, Q.options().dtype(torch::kFloat));

    dim3 gridDim(batchSize, numHeads, Tr);
    dim3 blockDim(Br);

    const int sram_size = (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    forwardKernel<<<gridDim, blockDim, sram_size>>>(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), 
                                         output.data_ptr<float>(), logsumexp.data_ptr<float>(), 
                                         N, d, Br, Bc, Tr, Tc, attentionScalar);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "forward kernel launch failed: ", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "forward kernel execution failed: ", cudaGetErrorString(err));

    return {output, logsumexp};
}