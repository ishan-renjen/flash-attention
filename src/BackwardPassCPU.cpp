#include <cmath>
#include <vector>

#include <torch/types.h>
#include <torch/extension.h>

#include "FlashAttention.h"


std::vector<torch::Tensor> backwardCPU(torch::Tensor Q, torch::Tensor K, torch::Tensor V,  torch::Tensor O, torch::Tensor dO, torch::Tensor L){
    //must be on GPU to use
    TORCH_CHECK(Q.is_cpu(), "Q must be CPU");
    TORCH_CHECK(K.is_cpu(), "K must be CPU");
    TORCH_CHECK(V.is_cpu(), "V must be CPU");
    TORCH_CHECK(O.is_cpu(), "O must be CPU");
    TORCH_CHECK(dO.is_cpu(), "dO must be CPU");
    TORCH_CHECK(L.is_cpu(), "dO must be CPU");

    //ideally they are contiguous
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(O.is_contiguous(), "O must be contiguous");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous");
    TORCH_CHECK(L.is_contiguous(), "L must be contiguous");

    //check dimensionality of Q, K, V
    TORCH_CHECK(Q.dim() == 4, "Q must have shape [B, H, N, d]");
    TORCH_CHECK(K.dim() == 4, "K must have shape [B, H, N, d]");
    TORCH_CHECK(V.dim() == 4, "V must have shape [B, H, N, d]");
    TORCH_CHECK(O.dim() == 4, "O must have shape [B, H, N, d]");
    TORCH_CHECK(dO.dim() == 4, "dO must have shape [B, H, N, d]");
    TORCH_CHECK(L.dim() == 3, "L must have shape [B, H, N]");

    //check the dtype of Q, K, V
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "V must be float32");
    TORCH_CHECK(O.scalar_type() == torch::kFloat32, "O must be float32");
    TORCH_CHECK(dO.scalar_type() == torch::kFloat32, "dO must be float32");
    TORCH_CHECK(L.scalar_type() == torch::kFloat32, "L must be float32");

    //check that Q, K, V dimensions match
    TORCH_CHECK(K.size(0) == Q.size(0) && V.size(0) == Q.size(0), "(Q,K,V) - Batch size mismatch");
    TORCH_CHECK(K.size(1) == Q.size(1) && V.size(1) == Q.size(1), "(Q,K,V) - Head count mismatch");
    TORCH_CHECK(K.size(2) == Q.size(2) && V.size(2) == Q.size(2), "(Q,K,V) - Sequence length mismatch");
    TORCH_CHECK(K.size(3) == Q.size(3) && V.size(3) == Q.size(3), "(Q,K,V) - Head dimension mismatch");

    // check that dO and O dimensions match
    TORCH_CHECK(O.size(0) == dO.size(0) && O.size(0) == Q.size(0), "(O,dO) - Batch size mismatch");
    TORCH_CHECK(O.size(1) == dO.size(1) && O.size(1) == Q.size(1), "(O,dO) - Head count mismatch");
    TORCH_CHECK(O.size(2) == dO.size(2) && O.size(2) == Q.size(2), "(O,dO) - Sequence length mismatch");
    TORCH_CHECK(O.size(3) == dO.size(3) && O.size(3) == Q.size(3), "(O,dO) - Head dimension mismatch");

    const int batchSize = Q.size(0);
    const int numHeads = Q.size(1);
    const int n = Q.size(2);
    const int d = Q.size(3);

    const int Bc = 16;
    const int Br = 16;

    const int Tr = std::ceil((float)n/Br);  // Divide Q, dQ, D into Tr blocks
    const int Tc = std::ceil((float)n/Bc);  // Divide K, V, dK, dV into N/Bc blocks

    torch::Tensor dQ = torch::zeros_like(Q);
    torch::Tensor dK = torch::zeros_like(K);
    torch::Tensor dV = torch::zeros_like(V);
    torch::Tensor D = (dO * O).sum(-1);

    const float scale = 1.0f/std::sqrt((float)d);

    for(int batch=0; batch<batchSize; batch++){
        for(int head=0; head<numHeads; head++){
            //get specific tiles for [batch, head]
            torch::Tensor Q_item  = Q.index({batch, head});
            torch::Tensor K_item  = K.index({batch, head});
            torch::Tensor V_item  = V.index({batch, head});
            torch::Tensor O_item  = O.index({batch, head});
            torch::Tensor L_item  = L.index({batch, head});
            torch::Tensor dO_item = dO.index({batch, head});
            torch::Tensor D_item  = D.index({batch, head});

            torch::Tensor dQ_item = dQ.index({batch, head});
            torch::Tensor dK_item = dK.index({batch, head});
            torch::Tensor dV_item = dV.index({batch, head});

            for(int kv_idx = 0; kv_idx < Tc; kv_idx++){
                //load K, V column tiles
                int kv_start = kv_idx * Bc;
                int kv_end = std::min(kv_start + Bc, n);
                int kv_rows = kv_end - kv_start;

                torch::Tensor K_tile = K_item.slice(0, kv_start, kv_end);
                torch::Tensor V_tile = V_item.slice(0, kv_start, kv_end);

                //initialize dK, dV
                torch::Tensor dK_tile = torch::zeros_like(K_tile);
                torch::Tensor dV_tile = torch::zeros_like(V_tile);

                for(int q_idx = 0; q_idx < Tr; q_idx++){
                    //load Qi, Oi, dOi, dQi, Li, Di
                    int q_start = q_idx * Br;
                    int q_end = std::min(q_start + Br, n);
                    int q_rows = q_end - q_start;

                    torch::Tensor Q_tile = Q_item.slice(0, q_start, q_end);
                    torch::Tensor O_tile = O_item.slice(0, q_start, q_end);
                    torch::Tensor L_tile = L_item.slice(0, q_start, q_end);
                    torch::Tensor D_tile = D_item.slice(0, q_start, q_end);
                    torch::Tensor dO_tile = dO_item.slice(0, q_start, q_end);
                    torch::Tensor dQ_tile = torch::zeros_like(Q_tile);
                    
                    //compute scores
                    torch::Tensor S_tile = torch::matmul(Q_tile, K_tile.transpose(0, 1)) * scale;
                    //compute P
                    torch::Tensor P = torch::exp(S_tile - L_tile.unsqueeze(1));
                    //compute dV
                    torch::Tensor dV_contrib = torch::matmul(P.transpose(0, 1), dO_tile);
                    //compute dP
                    torch::Tensor dP = torch::matmul(dO_tile, V_tile.transpose(0, 1));
                    //compute dS
                    torch::Tensor dS = P * (dP - D_tile.unsqueeze(1));
                    //update dQ_tile
                    torch::Tensor dQ_contrib = torch::matmul(dS, K_tile) * scale;
                    //update dK_tile
                    torch::Tensor dK_contrib = torch::matmul(dS.transpose(0, 1), Q_tile) * scale;

                    dQ_item.slice(0, q_start, q_end).add_(dQ_contrib);
                    dK_tile += dK_contrib;
                    dV_tile += dV_contrib;
                }
                //write dK, dV to memory
                dK_item.slice(0, kv_start, kv_end).copy_(dK_tile);
                dV_item.slice(0, kv_start, kv_end).copy_(dV_tile);
            }
        }
    }

    return {dQ, dK, dV};
}