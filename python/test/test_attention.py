from torch.utils.cpp_extension import load #https://docs.pytorch.org/docs/stable/cpp_extension.html
import torch
import math

flash_attn = load(name="flash_attn_ext", sources=['./src/main.cpp', './src/ForwardPass.cu', './src/BackwardPass.cu'], verbose=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dot_product_attention(q, k, v):
    #multiply query and transpose of key
    qk = torch.matmul(q, k.transpose(-2, -1))
    #scale by sqrt(dimension_key) to make sure that softmax gradients dont die with huge values
    qk = qk / math.sqrt(Q.size(-1))

    #softmax of resulting number, then multiply by value
    qk = torch.softmax(qk, dim=-1)
    output = torch.matmul(qk, v)
    logsumexp = torch.logsumexp(output, dim=-1)
    return output, logsumexp

Q = torch.rand(6, 2, 100, 2).to(device)
K = torch.rand(6, 2, 100, 2).to(device)
V = torch.rand(6, 2, 100, 2).to(device)

output, logsumexp         = flash_attn.flashattention_forward(Q, K, V)
output_ref, logsumexp_ref = dot_product_attention(Q, K, V)

print("output allclose:", torch.allclose(output, output_ref, atol=1e-4, rtol=1e-4))
print("logsumexp allclose:", torch.allclose(logsumexp, logsumexp_ref, atol=1e-4, rtol=1e-4))

print("max abs diff output:", (output - output_ref).abs().max().item())
print("max abs diff logsumexp:", (logsumexp - logsumexp_ref).abs().max().item())

#backward = flash_attn.flashattention_backward(torch.tensor(0))