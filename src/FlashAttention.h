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
void forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, torch::Tensor LSE);

/*
    Computes the backward pass for our Flash Attention implementation.
    inputs:
        torch::Tensor Q, K, V, O, dO, L
            - dimensions must be [Batch Size, Num. Heads, Sequence Length, Head Dimension]

    outputs:
        std::vector<torch::Tensor>
            torch::Tensor dQ, dK, dV
                - dimensions must be [Batch Size, Num. Heads, Sequence Length, Head Dimension]
    
*/
void backward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
                                    torch::Tensor O, torch::Tensor dO, torch::Tensor L, 
                                    torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV, 
                                    torch::Tensor D);

/*
    Computes the forward pass for our Flash Attention implementation on CPU as a baseline.
    inputs:
        torch::Tensor Query, Key, Value: 
            -must be on CPU
            -must have 4 dimensions [Batch Size, Heads(1 if single-head attention), N, d] (Nxd) is the Q, K, V matrix dimensions
    outputs: std::vector<torch::Tensor>
        torch::Tensor attention output
            -same dimensions as Q, K, V
        torch::Tensor logsumexp
            -logsum exponential numerical calculation to be used in backwards pass
    */
void forwardCPU(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, torch::Tensor LSE);

/*
    Computes the backward pass for our Flash Attention implementation on CPU as a baseline.
    inputs:
        torch::Tensor Q, K, V, O, dO, L
            - dimensions must be [Batch Size, Num. Heads, Sequence Length, Head Dimension]

    outputs:
        std::vector<torch::Tensor>
            torch::Tensor dQ, dK, dV
                - dimensions must be [Batch Size, Num. Heads, Sequence Length, Head Dimension]
    
*/
void backwardCPU(torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
                                       torch::Tensor O, torch::Tensor dO, torch::Tensor L, 
                                       torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV, 
                                       torch::Tensor D);