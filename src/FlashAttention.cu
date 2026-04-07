#include <cmath>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/types.h>
#include <torch/extension.h>

#include "FlashAttention.h"

//CUDA kernel for forward pass 
__global__ void forwardKernel(const float* Q, const float* K, const float* V, //Query, Key, Value
                              float* output, float* logsumexp,                //output, logsumexp tensors
                              const unsigned int N, const unsigned int d,     //dimensions of QKV matrices [N, d]
                              const unsigned int Br, const unsigned int Bc,   //block sizes [Br, Bc]
                              const unsigned int Tr, const unsigned int Tc,   //number of blocks per row and column
                              const float attentionScalar                     //scale the Q*K inside of the softmax function with this
                            ){
    /*
    load Qi from HBM -> SRAM
    initialize Oi(0) = (0)dim(Brxd), Li = (0)dim(Br), Mi(0) = (-inf)dim(Br)
    for j in range(1, (Tc)):
        load Kj, Vj from HBM -> SRAM
        1.) Calculate: Si(j) = Qi * Kj_transpose -> [Br, Bc]
        2.) Calculate: Mi(j) = max(Mi(j-1), rowmax(Si(j))) -> [Br]
                        Pi(j) = exp(Si(j) - Mi(j)) -> [Br, Bc] **POINTWISE**
                        Li(j) = e^(Mi(j-1) - Mi(j)) * Li(j-1) + rowsum(Pi(j)) -> [Br]
        3.) Calculate: Oi(j) = 1/diag(e^(Mi(j-1)) - Mi(j)) * Oi(j-1) + Pi(j)*Vj

    write out Oi, Li
    */
}

__global__ void backwardKernel(){

}

std::vector<torch::Tensor> forward_pass(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
    /*
    Computes the forward pass for our Flash Attention implementation.
    inputs:
        torch::Tensor Query, Key, Value: 
            -must be on cuda
            -must have 4 dimensions [Batch Size, Heads(1 if single-head attention), N, d] (Nxd) is the Q, K, V matrix dimensions
    outputs: std::vector<torch::Tensor>
        torch::Tensor attention output
            -same dimensions as Q, K, V
        torch::Tensor logsumexp
            -logsum exponential numerical calculation to be used in backwards pass
    */
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
    const int numHeads = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Bc = 16;
    const int Br = 16; //for RTX 4060 -> tile size of 256 per block, TODO: figure out what the best values for these are

    const int Tr = std::ceil((float)N/Br);
    const int Tc = std::ceil((float)N/Bc);
    const float attentionScalar = 1.0f/std::sqrt((float)d);

    torch::Tensor output = torch::zeros_like(Q);
    torch::Tensor logsumexp = torch::zeros({batchSize, numHeads, N}, Q.options().dtype(torch::kFloat));

    dim3 gridDim(Tr, numHeads, batchSize);
    dim3 blockDim(Br, Bc);
    forwardKernel<<<gridDim, blockDim>>>(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), 
                                         output.data_ptr<float>(), logsumexp.data_ptr<float>(), 
                                         N, d, Br, Bc, Tr, Tc, attentionScalar);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "forwardKernel launch failed: ", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "forwardKernel execution failed: ", cudaGetErrorString(err));

    return {output, logsumexp};
}

torch::Tensor backward_pass(torch::Tensor input){
    printf("backward pass\n");
    return torch::zeros_like(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_custom", &forward_pass, "Custom forward pass (CUDA)");
    m.def("backward_custom", &backward_pass, "Custom backward pass (CUDA)");
}