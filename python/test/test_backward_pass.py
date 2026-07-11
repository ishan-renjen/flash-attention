from torch.utils.cpp_extension import load
import torch
import math
import time
import sys

flash_attn = load(name="flash_attn_ext", 
                  sources=["./src/main.cpp", 
                           "./src/ForwardPass.cu", "./src/BackwardPass.cu", 
                           "./src/ForwardPassCPU.cpp", "./src/BackwardPassCPU.cpp"], 
                  extra_cuda_cflags=[
                                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                            ],
                           verbose=True)

def dot_product_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)
    logsumexp = torch.logsumexp(scores, dim=-1)
    return output, logsumexp

def time_fn(fn, use_cuda, warmup_iters=3, active_iters=10):
    for _ in range(warmup_iters):
        fn()

    if use_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(active_iters):
        fn()

    if use_cuda:
        torch.cuda.synchronize()

    end = time.perf_counter()
    return ((end - start) * 1000.0) / active_iters

def check_backward_close(name, dQ, dK, dV, dQ_ref, dK_ref, dV_ref):
    print(name)
    print("dQ allclose:", torch.allclose(dQ, dQ_ref, atol=1e-3, rtol=1e-3))
    print("dK allclose:", torch.allclose(dK, dK_ref, atol=1e-3, rtol=1e-3))
    print("dV allclose:", torch.allclose(dV, dV_ref, atol=1e-3, rtol=1e-3))
    print("max abs diff dQ:", (dQ - dQ_ref).abs().max().item())
    print("max abs diff dK:", (dK - dK_ref).abs().max().item())
    print("max abs diff dV:", (dV - dV_ref).abs().max().item())

def reference_backward(q_in, k_in, v_in, grad_out):
    q = q_in.clone().detach().requires_grad_(True)
    k = k_in.clone().detach().requires_grad_(True)
    v = v_in.clone().detach().requires_grad_(True)

    out, _ = dot_product_attention(q, k, v)
    dQ, dK, dV = torch.autograd.grad(outputs=out, inputs=(q, k, v), grad_outputs=grad_out, retain_graph=False, create_graph=False, allow_unused=False)

    return dQ, dK, dV

def alloc_forward_outputs(Q):
    O = torch.empty_like(Q)
    LSE = torch.empty(Q.shape[:-1], device=Q.device, dtype=torch.float32)
    return O, LSE

def alloc_backward_outputs(Q, K, V, O, dO):
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    D = torch.sum(O * dO, axis=-1)
    return dQ, dK, dV, D

def main():
    if len(sys.argv) != 5:
        print("usage: python3 test_backward_pass.py B H N D")
        sys.exit(1)

    B = int(sys.argv[1])
    H = int(sys.argv[2])
    N = int(sys.argv[3])
    D_dim = int(sys.argv[4])

    print(f"running backward pass with dimensions [{B}, {H}, {N}, {D_dim}]")

    torch.manual_seed(0)

    has_cuda = torch.cuda.is_available()
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda") if has_cuda else None

    Q_cpu = torch.rand(B, H, N, D_dim, device=device_cpu, dtype=torch.float32).contiguous()
    K_cpu = torch.rand(B, H, N, D_dim, device=device_cpu, dtype=torch.float32).contiguous()
    V_cpu = torch.rand(B, H, N, D_dim, device=device_cpu, dtype=torch.float32).contiguous()

    grad_out_cpu = torch.randn(B, H, N, D_dim, device=device_cpu, dtype=torch.float32).contiguous()

    if has_cuda:
        Q_gpu = Q_cpu.to(device_gpu).contiguous()
        K_gpu = K_cpu.to(device_gpu).contiguous()
        V_gpu = V_cpu.to(device_gpu).contiguous()
        grad_out_gpu = grad_out_cpu.to(device_gpu).contiguous()

    dQ_cpu_ref, dK_cpu_ref, dV_cpu_ref = reference_backward(Q_cpu, K_cpu, V_cpu, grad_out_cpu)
    out_cpu, lse_cpu = alloc_forward_outputs(Q_cpu)
    flash_attn.flashattention_forward_cpu(Q_cpu, K_cpu, V_cpu, out_cpu, lse_cpu)
    dQ_cpu, dK_cpu, dV_cpu, D_cpu = alloc_backward_outputs(Q_cpu, K_cpu, V_cpu, out_cpu, grad_out_cpu)
    flash_attn.flashattention_backward_cpu(Q_cpu, K_cpu, V_cpu, out_cpu, grad_out_cpu, lse_cpu, dQ_cpu, dK_cpu, dV_cpu, D_cpu)

    check_backward_close("BACKWARD CPU CORRECTNESS", dQ_cpu, dK_cpu, dV_cpu, dQ_cpu_ref, dK_cpu_ref, dV_cpu_ref)

    if has_cuda:
        dQ_gpu_ref, dK_gpu_ref, dV_gpu_ref = reference_backward(Q_gpu, K_gpu, V_gpu, grad_out_gpu)
        out_gpu, lse_gpu = alloc_forward_outputs(Q_gpu)
        flash_attn.flashattention_forward(Q_gpu, K_gpu, V_gpu, out_gpu, lse_gpu)
        dQ_gpu, dK_gpu, dV_gpu, D_gpu = alloc_backward_outputs(Q_gpu, K_gpu, V_gpu, out_gpu, grad_out_gpu)
        flash_attn.flashattention_backward(Q_gpu, K_gpu, V_gpu, out_gpu, grad_out_gpu, lse_gpu, dQ_gpu, dK_gpu, dV_gpu, D_gpu)
        torch.cuda.synchronize()

        check_backward_close("BACKWARD GPU CORRECTNESS", dQ_gpu, dK_gpu, dV_gpu, dQ_gpu_ref, dK_gpu_ref, dV_gpu_ref)

    def run_ref_cpu_backward():
        return reference_backward(Q_cpu, K_cpu, V_cpu, grad_out_cpu)

    def run_flash_cpu_backward():
        out, lse = alloc_forward_outputs(Q_cpu)
        flash_attn.flashattention_forward_cpu(Q_cpu, K_cpu, V_cpu, out, lse)

        dQ, dK, dV, D_tmp = alloc_backward_outputs(Q_cpu, K_cpu, V_cpu, out, grad_out_cpu)
        flash_attn.flashattention_backward_cpu(Q_cpu, K_cpu, V_cpu, out, grad_out_cpu, lse, dQ, dK, dV, D_tmp)

        return dQ, dK, dV

    if has_cuda:
        def run_ref_gpu_backward():
            return reference_backward(Q_gpu, K_gpu, V_gpu, grad_out_gpu)

        def run_flash_gpu_backward():
            out, lse = alloc_forward_outputs(Q_gpu)
            flash_attn.flashattention_forward(Q_gpu, K_gpu, V_gpu, out, lse)

            dQ, dK, dV, D_tmp = alloc_backward_outputs(Q_gpu, K_gpu, V_gpu, out, grad_out_gpu)
            flash_attn.flashattention_backward(Q_gpu, K_gpu, V_gpu, out, grad_out_gpu, lse, dQ, dK, dV, D_tmp)

            return dQ, dK, dV

    print("\n" + "=" * 80)
    print("TIMING RESULTS")
    print("=" * 80)

    if has_cuda:
        ref_gpu_ms = time_fn(run_ref_gpu_backward, use_cuda=True, warmup_iters=3,active_iters=10)
        flash_gpu_ms = time_fn(run_flash_gpu_backward, use_cuda=True, warmup_iters=3, active_iters=10)

        print(f"GPU reference backward: {ref_gpu_ms:.4f} ms / iter")
        print(f"GPU flash backward:     {flash_gpu_ms:.4f} ms / iter")

    ref_cpu_ms = time_fn(run_ref_cpu_backward, use_cuda=False, warmup_iters=1, active_iters=3)
    flash_cpu_ms = time_fn(run_flash_cpu_backward, use_cuda=False, warmup_iters=1, active_iters=3)

    print(f"CPU reference backward: {ref_cpu_ms:.4f} ms / iter")
    print(f"CPU flash backward:     {flash_cpu_ms:.4f} ms / iter")

if __name__ == "__main__":
    main()