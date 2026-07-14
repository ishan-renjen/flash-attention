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

def time_cpu(fn, warmup_iters=1, active_iters=3):
    for _ in range(warmup_iters):
        fn()

    start = time.perf_counter()
    for _ in range(active_iters):
        fn()
    end = time.perf_counter()

    total_ms = (end - start) * 1000.0
    return total_ms / active_iters

def time_cuda(fn, warmup_iters=3, active_iters=10):
    for _ in range(warmup_iters):
        fn()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(active_iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_ms = (end - start) * 1000.0
    return total_ms / active_iters

def check_close(name, out, out_ref, lse, lse_ref):
    print(name)
    out_f      = out.float()
    out_ref_f  = out_ref.float()
    lse_f      = lse.float()
    lse_ref_f  = lse_ref.float()
    atol, rtol = 2e-2, 2e-2

    print("output allclose:", torch.allclose(out_f, out_ref_f, atol=atol, rtol=rtol))
    print("logsumexp allclose:", torch.allclose(lse_f, lse_ref_f, atol=atol, rtol=rtol))
    print("max abs diff output:", (out_f - out_ref_f).abs().max().item())
    print("max abs diff logsumexp:", (lse_f - lse_ref_f).abs().max().item())

def alloc_forward_outputs(Q):
    out = torch.empty_like(Q)
    lse = torch.empty(Q.shape[:-1], device=Q.device, dtype=torch.float32)
    return out, lse

def main():
    if len(sys.argv) != 5:
        print("usage: python3 test_forward_pass.py B H N D")
        sys.exit(1)

    B = int(sys.argv[1])
    H = int(sys.argv[2])
    N = int(sys.argv[3])
    D = int(sys.argv[4])

    print(f"running forward pass with dimensions [{B}, {H}, {N}, {D}]")

    torch.manual_seed(0)

    has_cuda = torch.cuda.is_available()
    device_gpu = torch.device("cuda") if has_cuda else None
    device_cpu = torch.device("cpu")

    Q_cpu = torch.rand(B, H, N, D, device=device_cpu, dtype=torch.bfloat16).contiguous()
    K_cpu = torch.rand(B, H, N, D, device=device_cpu, dtype=torch.bfloat16).contiguous()
    V_cpu = torch.rand(B, H, N, D, device=device_cpu, dtype=torch.bfloat16).contiguous()

    if has_cuda:
        Q_gpu = Q_cpu.to(device_gpu).contiguous()
        K_gpu = K_cpu.to(device_gpu).contiguous()
        V_gpu = V_cpu.to(device_gpu).contiguous()

    if has_cuda:
        out_gpu, lse_gpu = alloc_forward_outputs(Q_gpu)
        flash_attn.flashattention_forward(Q_gpu, K_gpu, V_gpu, out_gpu, lse_gpu)
        out_gpu_ref, lse_gpu_ref = dot_product_attention(Q_gpu, K_gpu, V_gpu)
        torch.cuda.synchronize()

        check_close("FORWARD GPU CORRECTNESS", out_gpu, out_gpu_ref, lse_gpu, lse_gpu_ref)

    out_cpu, lse_cpu = alloc_forward_outputs(Q_cpu)
    flash_attn.flashattention_forward_cpu(Q_cpu, K_cpu, V_cpu, out_cpu, lse_cpu)
    out_cpu_ref, lse_cpu_ref = dot_product_attention(Q_cpu, K_cpu, V_cpu)

    check_close("FORWARD CPU CORRECTNESS", out_cpu, out_cpu_ref, lse_cpu, lse_cpu_ref)

    def run_ref_cpu():
        return dot_product_attention(Q_cpu, K_cpu, V_cpu)

    def run_flash_cpu():
        out, lse = alloc_forward_outputs(Q_cpu)
        flash_attn.flashattention_forward_cpu(Q_cpu, K_cpu, V_cpu, out, lse)

        return out, lse

    if has_cuda:
        def run_ref_gpu():
            return dot_product_attention(Q_gpu, K_gpu, V_gpu)

        def run_flash_gpu():
            out, lse = alloc_forward_outputs(Q_gpu)
            flash_attn.flashattention_forward(Q_gpu, K_gpu, V_gpu, out, lse)

            return out, lse

    print("\n" + "=" * 80)
    print("TIMING RESULTS")
    print("=" * 80)

    if has_cuda:
        ref_gpu_ms = time_cuda(run_ref_gpu, warmup_iters=3, active_iters=10)
        flash_gpu_ms = time_cuda(run_flash_gpu, warmup_iters=3, active_iters=10)

        print(f"GPU reference forward: {ref_gpu_ms:.4f} ms / iter")
        print(f"GPU flash forward:     {flash_gpu_ms:.4f} ms / iter")

    ref_cpu_ms = time_cpu(run_ref_cpu, warmup_iters=1, active_iters=3)
    flash_cpu_ms = time_cpu(run_flash_cpu, warmup_iters=1, active_iters=3)

    print(f"CPU reference forward: {ref_cpu_ms:.4f} ms / iter")
    print(f"CPU flash forward:     {flash_cpu_ms:.4f} ms / iter")

if __name__ == "__main__":
    main()