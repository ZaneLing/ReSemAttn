"""Stage 3: relation-semantic attention layer."""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationConditionedKV(nn.Module):
    """Relation-conditioned key/value construction.

    Implements:
      W_K(R_ik) = W_K^0 + sum_b omega_b(R_ik) U_K,b V_K,b^T
      W_V(R_ik) = W_V^0 + sum_b omega_b(R_ik) U_V,b V_V,b^T
      k_ik = W_K(R_ik) z_ik
      v_ik = W_V(R_ik) z_ik
    """
    def __init__(self, hidden_dim: int, num_bases: int = 4, rank: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_bases = num_bases
        self.rank = rank
        self.base_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.base_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Uk = nn.Parameter(torch.randn(num_bases, hidden_dim, rank) * 0.02)
        self.Vk = nn.Parameter(torch.randn(num_bases, hidden_dim, rank) * 0.02)
        self.Uv = nn.Parameter(torch.randn(num_bases, hidden_dim, rank) * 0.02)
        self.Vv = nn.Parameter(torch.randn(num_bases, hidden_dim, rank) * 0.02)
        self.mix = nn.Linear(hidden_dim, num_bases)

    def forward(self, z: torch.Tensor, phi_R: torch.Tensor) -> Dict[str, torch.Tensor]:
        # z, phi_R: [N,K,d]
        omega = torch.softmax(self.mix(phi_R), dim=-1)  # [N,K,B]
        base_k = self.base_k(z)
        base_v = self.base_v(z)
        # low-rank transformations applied as U(V^T z)
        # einsum z [N,K,d], V [B,d,r] -> [N,K,B,r] -> U [B,d,r] -> [N,K,B,d]
        zk = torch.einsum('nkd,bdr->nkbr', z, self.Vk)
        kv_delta = torch.einsum('nkbr,bdr->nkbd', zk, self.Uk)
        zv = torch.einsum('nkd,bdr->nkbr', z, self.Vv)
        vv_delta = torch.einsum('nkbr,bdr->nkbd', zv, self.Uv)
        delta_k = torch.sum(omega.unsqueeze(-1) * kv_delta, dim=2)
        delta_v = torch.sum(omega.unsqueeze(-1) * vv_delta, dim=2)
        return {"k": base_k + delta_k, "v": base_v + delta_v, "omega": omega}


class RelationSemanticAttentionLayer(nn.Module):
    """Evidence branch injected into a Transformer hidden layer.

    Implements:
      alpha_tk^l = softmax_k(q_t^T k_ik / sqrt(d) + lambda_l log(p_K + eps))
      c_t^l = sum_k alpha_tk^l v_ik
      g_t^l = sigmoid(W_g[h_t;c_t;h_schema;h_ai])
      h_tilde_t^l = h_t^l + g_t^l \odot c_t^l
    """
    def __init__(self, hidden_dim: int, prior_init: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(4 * hidden_dim, hidden_dim)
        self.lambda_raw = nn.Parameter(torch.tensor(float(prior_init)))
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, h: torch.Tensor, key: torch.Tensor, value: torch.Tensor, prior: torch.Tensor, path_mask: torch.Tensor, schema_repr: torch.Tensor, cand_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        # h [N,n,d], key/value [N,K,d], prior/path_mask [N,K], schema_repr/cand_repr [N,d]
        d = h.shape[-1]
        q = self.q_proj(h)
        logits = torch.einsum('nld,nkd->nlk', q, key) / (d ** 0.5)
        lam = F.softplus(self.lambda_raw)
        logits = logits + lam * torch.log(prior.clamp_min(1e-8)).unsqueeze(1)
        logits = logits.masked_fill(~path_mask.unsqueeze(1), -1e9)
        alpha = torch.softmax(logits, dim=-1)
        context = torch.einsum('nlk,nkd->nld', alpha, value)
        schema_exp = schema_repr.unsqueeze(1).expand_as(h)
        cand_exp = cand_repr.unsqueeze(1).expand_as(h)
        gate = torch.sigmoid(self.gate(torch.cat([h, context, schema_exp, cand_exp], dim=-1)))
        h_tilde = h + gate * context
        h_tilde = self.norm(h_tilde + self.dropout(self.ffn(h_tilde)))
        return {"hidden": h_tilde, "alpha": alpha, "context": context, "gate": gate, "lambda": lam}
