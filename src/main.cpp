#include <torch/extension.h>

#include "FlashAttention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flashattention_forward", torch::wrap_pybind_function(forward), "custom forward pass");
    m.def("flashattention_backward", torch::wrap_pybind_function(backward), "custom backward pass");
    m.def("flashattention_forward_cpu", torch::wrap_pybind_function(forwardCPU), "CPU forward pass");
    m.def("flashattention_backward_cpu", torch::wrap_pybind_function(backwardCPU), "CPU backward pass");
}