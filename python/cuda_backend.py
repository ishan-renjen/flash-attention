from torch.utils.cpp_extension import load #https://docs.pytorch.org/docs/stable/cpp_extension.html
import torch

flash_attn = load(name="flash_attn_ext", sources=["./src/FlashAttention.cu"], verbose=True)

forward = flash_attn.forward_custom(torch.tensor(0))
backward = flash_attn.backward_custom(torch.tensor(0))