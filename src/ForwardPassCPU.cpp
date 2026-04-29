#include <cmath>
#include <vector>

#include <torch/torch.h>
#include <torch/types.h>
#include <torch/extension.h>

#include "FlashAttention.h"

std::vector<torch::Tensor> forwardCPU(torch::Tensor Q, 
                                      torch::Tensor K, 
                                      torch::Tensor V){
    TORCH_CHECK(Q.device().is_cpu(), "Q must be on CPU");
    TORCH_CHECK(K.device().is_cpu(), "K must be on CPU");
    TORCH_CHECK(V.device().is_cpu(), "V must be on CPU");

    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");

    TORCH_CHECK(Q.dim() == 4, "Q must have shape [B, H, N, D]");
    TORCH_CHECK(K.dim() == 4, "K must have shape [B, H, N, D]");
    TORCH_CHECK(V.dim() == 4, "V must have shape [B, H, N, D]");
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have same shape");

    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    const int batch_dim = Q.size(0);
    const int head_dim  = Q.size(1);
    const int n         = Q.size(2);
    const int d         = Q.size(3);

    const int Bc = 32;
    const int Br = 32;

    const int Tr = (n + Br - 1) / Br;
    const int Tc = (n + Bc - 1) / Bc;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    torch::Tensor O = torch::empty_like(Q);
    torch::Tensor LSE = torch::empty({batch_dim, head_dim, n}, Q.options().dtype(torch::kFloat32));

    /*loops to calculate out and LSE*/
    for(int batch=0; batch<batch_dim; batch++){
        for(int head=0; head<head_dim; head++){

            torch::Tensor Q_item = Q[batch][head];
            torch::Tensor K_item = K[batch][head];
            torch::Tensor V_item = V[batch][head];

            for(int q_idx=0; q_idx<Tr; q_idx++){
                //index into Q
                int q_start = q_idx * Br;
                int q_end = std::min(q_start + Br, n);
                int q_rows = q_end - q_start;

                torch::Tensor Q_tile = Q_item.slice(0, q_start, q_end);

                torch::Tensor m_i = torch::full({q_rows}, -std::numeric_limits<float>::infinity(), Q.options().dtype(torch::kFloat32));
                torch::Tensor l_i = torch::zeros({q_rows}, Q.options().dtype(torch::kFloat32));
                torch::Tensor O_i = torch::zeros({q_rows, d}, Q.options().dtype(torch::kFloat32));

                torch::Tensor m_new = torch::full({q_rows}, -std::numeric_limits<float>::infinity(), Q.options().dtype(torch::kFloat32));
                torch::Tensor l_new = torch::zeros({q_rows}, Q.options().dtype(torch::kFloat32));

                for(int kv_idx=0; kv_idx<Tc; kv_idx++){
                    //index into K, V
                    int kv_start = kv_idx * Bc;
                    int kv_end = std::min(kv_start + Bc, n);
                    int kv_rows = kv_end - kv_start;

                    torch::Tensor K_tile = K_item.slice(0, kv_start, kv_end);
                    torch::Tensor V_tile = V_item.slice(0, kv_start, kv_end);

                    //calculate S = Q_tile * K_tile^T
                    torch::Tensor S = torch::matmul(Q_tile, K_tile.transpose(0, 1)) * scale;

                    //calculate Mi, Pi, li, Oi
                    torch::Tensor m_tilde = std::get<0>(S.max(1));
                    m_new = torch::maximum(m_i, m_tilde);
                    torch::Tensor alpha = torch::exp(m_i - m_new);
                    alpha = torch::where(torch::isinf(m_i), torch::zeros_like(alpha), alpha);

                    torch::Tensor P = torch::exp(S - m_new.unsqueeze(1));
                    l_new = alpha * l_i + P.sum(1);

                    O_i = O_i * alpha.unsqueeze(1) + torch::matmul(P, V_tile);

                    m_i = m_new;
                    l_i = l_new;
                }
                //calculate O, Li

                O_i = O_i / l_i.unsqueeze(1);
                O[batch][head].slice(0, q_start, q_end).copy_(O_i);
                LSE[batch][head].slice(0, q_start, q_end).copy_(m_i + torch::log(l_i));
            }
        }
    }

    return {O, LSE};
}