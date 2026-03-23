#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/extension.h>

#include "FlashAttention.h"

__global__ void forwardKernel(){

}

__global__ void backwardKernel(){

}

torch::Tensor forward_pass(torch::Tensor input){

}

torch::Tensor backward_pass(torch::Tensor input){
    return torch::zeros_like(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_custom", &forward_pass, "Custom forward pass (CUDA)");
    m.def("backward_custom", &backward_pass, "Custom backward pass (CUDA)");
}