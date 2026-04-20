from torch.utils.cpp_extension import load
import torch
import math
from torch.profiler import profile, ProfilerActivity, record_function

flash_attn = load(
    name="flash_attn_ext",
    sources=['./src/main.cpp', './src/ForwardPass.cu', './src/BackwardPass.cu'],
    verbose=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")
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
# Shapes
# --------------------------------------------------
B, H, N, D = 32, 16, 1024, 64

Q = torch.rand(B, H, N, D, device=device, dtype=torch.float32).contiguous()
K = torch.rand(B, H, N, D, device=device, dtype=torch.float32).contiguous()
V = torch.rand(B, H, N, D, device=device, dtype=torch.float32).contiguous()

Q_cpu = torch.rand(B, H, N, D, device=device_cpu, dtype=torch.float32).contiguous()
K_cpu = torch.rand(B, H, N, D, device=device_cpu, dtype=torch.float32).contiguous()
V_cpu = torch.rand(B, H, N, D, device=device_cpu, dtype=torch.float32).contiguous()

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
print()


# --------------------------------------------------
# Backward correctness
# --------------------------------------------------
Q_ref = Q.clone().detach().requires_grad_(True)
K_ref = K.clone().detach().requires_grad_(True)
V_ref = V.clone().detach().requires_grad_(True)

Qcpu_ref = Q_cpu.clone().detach().requires_grad_(True)
Kcpu_ref = K_cpu.clone().detach().requires_grad_(True)
Vcpu_ref = V_cpu.clone().detach().requires_grad_(True)

out_ref, lse_ref = dot_product_attention(Q_ref, K_ref, V_ref)
grad_out = torch.randn_like(out_ref)

outcpu_ref, lsecpu_ref = dot_product_attention(Qcpu_ref, Kcpu_ref, Vcpu_ref)
grad_out_cpu = torch.randn_like(outcpu_ref)

loss_ref = (out_ref * grad_out).sum()
loss_ref.backward()

dQ_ref = Q_ref.grad.detach()
dK_ref = K_ref.grad.detach()
dV_ref = V_ref.grad.detach()

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
# Helpers
# --------------------------------------------------
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def profiler_activities():
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    return acts

def warmup(fn, iters=10):
    for _ in range(iters):
        fn()
    cuda_sync()

def print_prof_summary(prof, label, row_limit=25):
    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    print(f"\n{'=' * 100}")
    print(label)
    print(f"{'=' * 100}")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=row_limit))


# --------------------------------------------------
# Workloads
# --------------------------------------------------
def run_ref_forward():
    out, lse = dot_product_attention(Q, K, V)
    return out, lse

def run_flash_forward():
    out, lse = flash_attn.flashattention_forward(Q, K, V)
    return out, lse

def run_cpu_forward():
    output_cpu, logsumexp_cpu = dot_product_attention(Q_cpu, K_cpu, V_cpu)
    return output_cpu, logsumexp_cpu

def run_ref_backward():
    q = Q.clone().detach().requires_grad_(True)
    k = K.clone().detach().requires_grad_(True)
    v = V.clone().detach().requires_grad_(True)

    out, _ = dot_product_attention(q, k, v)

    dQ_ref, dK_ref, dV_ref = torch.autograd.grad(
        outputs=out,
        inputs=(q, k, v),
        grad_outputs=grad_out,
        retain_graph=False, create_graph=False, allow_unused=False)
    return dQ_ref, dK_ref, dV_ref

def run_flash_backward():
    out, lse = flash_attn.flashattention_forward(Q, K, V)
    dQ, dK, dV = flash_attn.flashattention_backward(Q, K, V, out, grad_out, lse)
    return dQ, dK, dV

def run_cpu_backward():
    q = Q_cpu.clone().detach().requires_grad_(True)
    k = K_cpu.clone().detach().requires_grad_(True)
    v = V_cpu.clone().detach().requires_grad_(True)

    out_cpu, _ = dot_product_attention(q, k, v)

    dQ_ref, dK_ref, dV_ref = torch.autograd.grad(
        outputs=out_cpu,
        inputs=(q, k, v),
        grad_outputs=grad_out_cpu,
        retain_graph=False, create_graph=False, allow_unused=False)
    return dQ_ref, dK_ref, dV_ref

# --------------------------------------------------
# Profile simple repeated regions
# --------------------------------------------------
def profile_region(label, fn, warmup_iters=10, active_iters=20, row_limit=25, trace_file=None):
    warmup(fn, warmup_iters)

    with profile(activities=profiler_activities(), record_shapes=True, profile_memory=True, with_stack=False) as prof:
        for _ in range(active_iters):
            with record_function(label):
                fn()

    cuda_sync()
    print_prof_summary(prof, f"{label} ({active_iters} iterations)", row_limit=row_limit)

    if trace_file is not None:
        prof.export_chrome_trace(trace_file)
        print(f"chrome trace saved to: {trace_file}")

    return prof


# --------------------------------------------------
# Run profiler
# --------------------------------------------------
prof_ref_fwd = profile_region(label="reference_forward", fn=run_ref_forward, warmup_iters=10, active_iters=20, row_limit=20, trace_file="reference_forward_trace.json")
prof_ref_fwd = profile_region(label="cpu_forward", fn=run_cpu_forward, warmup_iters=10, active_iters=20, row_limit=20, trace_file="cpu_forward_trace.json")

prof_flash_fwd = profile_region(
    label="flash_forward",
    fn=run_flash_forward,
    warmup_iters=10,
    active_iters=20,
    row_limit=20,
    trace_file="flash_forward_trace.json")

prof_ref_bwd = profile_region(
    label="reference_backward",
    fn=run_ref_backward,
    warmup_iters=10,
    active_iters=20,
    row_limit=20,
    trace_file="reference_backward_trace.json")

prof_ref_bwd = profile_region(
    label="cpu_backward",
    fn=run_cpu_backward,
    warmup_iters=10,
    active_iters=20,
    row_limit=20,
    trace_file="cpu_backward_trace.json")


prof_flash_bwd = profile_region(
    label="flash_manual_backward",
    fn=run_flash_backward,
    warmup_iters=10,
    active_iters=20,
    row_limit=20,
    trace_file="flash_manual_backward_trace.json")