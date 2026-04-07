#include <cmath>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/types.h>
#include <torch/extension.h>

#include "FlashAttention.h"

__global__ void backwardKernel(){
}

torch::Tensor backward(torch::Tensor input){
    printf("backward pass\n");
    return torch::zeros_like(input);
}