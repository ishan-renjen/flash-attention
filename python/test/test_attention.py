from torch.utils.cpp_extension import load #https://docs.pytorch.org/docs/stable/cpp_extension.html
import torch
import math

flash_attn = load(name="flash_attn_ext", sources=['./src/main.cpp', './src/ForwardPass.cu', './src/BackwardPass.cu'], verbose=True)

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def dot_product_attention(q, k, v):
    # q, k, v: [B, H, N, D]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))  
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)                         
    logsumexp = torch.logsumexp(scores, dim=-1)            
    return output, logsumexp

Q = torch.rand(10, 2, 64, 64).to(device)
K = torch.rand(10, 2, 64, 64).to(device)
V = torch.rand(10, 2, 64, 64).to(device)

output, logsumexp         = flash_attn.flashattention_forward(Q, K, V)
output_ref, logsumexp_ref = dot_product_attention(Q, K, V)

print("output correct?:", torch.allclose(output, output_ref, atol=1e-4, rtol=1e-4))
print("logsumexp correct?:", torch.allclose(logsumexp, logsumexp_ref, atol=1e-4, rtol=1e-4))

print("max abs diff output:", (output - output_ref).abs().max().item())
print("max abs diff logsumexp:", (logsumexp - logsumexp_ref).abs().max().item())