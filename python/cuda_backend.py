from torch.utils.cpp_extension import load #https://docs.pytorch.org/docs/stable/cpp_extension.html
import torch

flash_attn = load(name="flash_attn_ext", sources=['./src/main.cpp', './src/ForwardPass.cu', './src/BackwardPass.cu'], verbose=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Q = torch.zeros(6, 100, 2, 2).to(device)
K = torch.zeros(6, 100, 2, 2).to(device)
V = torch.zeros(6, 100, 2, 2).to(device)

output, logsumexp = flash_attn.flashattention_forward(Q, K, V)
print(output)

backward = flash_attn.flashattention_backward(torch.tensor(0))