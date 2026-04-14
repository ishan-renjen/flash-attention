from torch.utils.cpp_extension import load
import torch
import math

flash_attn = load(
    name="flash_attn_ext",
    sources=['./src/main.cpp', './src/ForwardPass.cu', './src/BackwardPass.cu'],
    verbose=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# --------------------------------------------------
# Reference attention: q, k, v have shape [B, H, N, D]
# --------------------------------------------------
def dot_product_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))   # [B,H,N,N]
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)                                          # [B,H,N,D]
    logsumexp = torch.logsumexp(scores, dim=-1)                              # [B,H,N]
    return output, logsumexp


# --------------------------------------------------
# FLOP estimates (approximate)
# --------------------------------------------------
def attention_forward_flops(B, H, N, D, include_softmax=True):
    # QK^T: 2 * B * H * N * N * D
    # P@V : 2 * B * H * N * N * D
    flops = 4 * H * N * N * D
    # if include_softmax:
    #     flops += 5 * B * H * N * N   # rough softmax estimate
    return flops

def attention_backward_flops(B, H, N, D, include_softmax=True):
    # dV = P^T @ dO         -> 2 * B * H * N * N * D
    # dP = dO @ V^T         -> 2 * B * H * N * N * D
    # dQ = dS @ K           -> 2 * B * H * N * N * D
    # dK = dS^T @ Q         -> 2 * B * H * N * N * D
    flops = 2.5 * (4 * H * N * N * D)
    # if include_softmax:
    #     flops += 5 * B * H * N * N   # rough softmax backward estimate
    return flops


# --------------------------------------------------
# Timing helper
# --------------------------------------------------
def benchmark_cuda(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iters
    return avg_ms


# --------------------------------------------------
# Shapes
# --------------------------------------------------
B, H, N, D = 32, 16, 1024, 64

Q = torch.rand(B, H, N, D, device=device, dtype=torch.float32).contiguous()
K = torch.rand(B, H, N, D, device=device, dtype=torch.float32).contiguous()
V = torch.rand(B, H, N, D, device=device, dtype=torch.float32).contiguous()

# --------------------------------------------------
# Forward correctness
# --------------------------------------------------
output, logsumexp = flash_attn.flashattention_forward(Q, K, V)
output_ref, logsumexp_ref = dot_product_attention(Q, K, V)

print("FORWARD CORRECTNESS")
print("output allclose:", torch.allclose(output, output_ref, atol=1e-4, rtol=1e-4))
print("logsumexp allclose:", torch.allclose(logsumexp, logsumexp_ref, atol=1e-4, rtol=1e-4))
print("max abs diff output:", (output - output_ref).abs().max().item())
print("max abs diff logsumexp:", (logsumexp - logsumexp_ref).abs().max().item())
print("Q shape:", Q.shape, Q.dtype, Q.is_contiguous())
print("K shape:", K.shape, K.dtype, K.is_contiguous())
print("V shape:", V.shape, V.dtype, V.is_contiguous())
print("O shape:", output.shape, output.dtype, output.is_contiguous())
print("L shape:", logsumexp.shape, logsumexp.dtype, logsumexp.is_contiguous())
print()

# --------------------------------------------------
# Backward correctness
# --------------------------------------------------
Q_ref = Q.clone().detach().requires_grad_(True)
K_ref = K.clone().detach().requires_grad_(True)
V_ref = V.clone().detach().requires_grad_(True)

out_ref, lse_ref = dot_product_attention(Q_ref, K_ref, V_ref)
grad_out = torch.randn_like(out_ref)

loss_ref = (out_ref * grad_out).sum()
loss_ref.backward()

dQ_ref = Q_ref.grad.detach()
dK_ref = K_ref.grad.detach()
dV_ref = V_ref.grad.detach()

# print("grad_out shape:", grad_out.shape)
dQ, dK, dV = flash_attn.flashattention_backward(Q, K, V, output, grad_out, logsumexp)

print("BACKWARD CORRECTNESS")
print("dQ allclose:", torch.allclose(dQ, dQ_ref, atol=1e-3, rtol=1e-3))
print("dK allclose:", torch.allclose(dK, dK_ref, atol=1e-3, rtol=1e-3))
print("dV allclose:", torch.allclose(dV, dV_ref, atol=1e-3, rtol=1e-3))
print("max abs diff dQ:", (dQ - dQ_ref).abs().max().item())
print("max abs diff dK:", (dK - dK_ref).abs().max().item())
print("max abs diff dV:", (dV - dV_ref).abs().max().item())
print()

# --------------------------------------------------
# Forward timing
# --------------------------------------------------
def run_ref_forward():
    out, lse = dot_product_attention(Q, K, V)
    return out, lse

def run_flash_forward():
    out, lse = flash_attn.flashattention_forward(Q, K, V)
    return out, lse

ref_fwd_ms = benchmark_cuda(run_ref_forward, warmup=20, iters=100)
flash_fwd_ms = benchmark_cuda(run_flash_forward, warmup=20, iters=100)

print("FORWARD TIMING")
print(f"reference forward: {ref_fwd_ms:.4f} ms")
print(f"flash forward:     {flash_fwd_ms:.4f} ms")
print(f"forward speedup:   {ref_fwd_ms / flash_fwd_ms:.3f}x")
print()

# --------------------------------------------------
# Backward timing
# --------------------------------------------------
def run_ref_backward():
    q = Q.clone().detach().requires_grad_(True)
    k = K.clone().detach().requires_grad_(True)
    v = V.clone().detach().requires_grad_(True)

    out, _ = dot_product_attention(q, k, v)
    loss = (out * grad_out).sum()
    loss.backward()

def run_flash_backward():
    out, lse = flash_attn.flashattention_forward(Q, K, V)

    # Adjust to your actual API
    dQ, dK, dV = flash_attn.flashattention_backward(Q, K, V, out, grad_out, lse)
    return dQ, dK, dV

ref_bwd_ms = benchmark_cuda(run_ref_backward, warmup=20, iters=100)
flash_bwd_ms = benchmark_cuda(run_flash_backward, warmup=20, iters=100)

print("BACKWARD TIMING")
print(f"reference backward: {ref_bwd_ms:.4f} ms")
print(f"flash backward:     {flash_bwd_ms:.4f} ms")
print(f"backward speedup:   {ref_bwd_ms / flash_bwd_ms:.3f}x")
print()

# --------------------------------------------------
# Approximate throughput
# --------------------------------------------------
fwd_flops = attention_forward_flops(B, H, N, D)
bwd_flops = attention_backward_flops(B, H, N, D)

flash_fwd_tflops = fwd_flops / (flash_fwd_ms * 1e-3) / 1e12
ref_fwd_tflops = fwd_flops / (ref_fwd_ms * 1e-3) / 1e12

flash_bwd_tflops = bwd_flops / (flash_bwd_ms * 1e-3) / 1e12
ref_bwd_tflops = bwd_flops / (ref_bwd_ms * 1e-3) / 1e12

print("APPROXIMATE THROUGHPUT")
print(f"reference forward TFLOP/s: {ref_fwd_tflops:.4f}")
print(f"flash forward TFLOP/s:     {flash_fwd_tflops:.4f}")
print(f"reference backward TFLOP/s:{ref_bwd_tflops:.4f}")
print(f"flash backward TFLOP/s:    {flash_bwd_tflops:.4f}")