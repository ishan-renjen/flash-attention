#pragma once

#include <vector>

#include <torch/types.h>

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
std::vector<torch::Tensor> forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

/*
    Computes the backward pass for our Flash Attention implementation.
    inputs:

    outputs:
    
*/
torch::Tensor backward(torch::Tensor input);