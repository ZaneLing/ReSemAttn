"""Stage 2: candidate-specific path memory and relation-semantic prior."""
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [..., L, d]
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        w = torch.softmax(logits, dim=-1)
        return torch.sum(w.unsqueeze(-1) * x, dim=-2)


class PathMemoryEncoder(nn.Module):
    """Encodes m_ik=(E_ik, R_ik, P_ik, T_ik) into z_ik.

    Implements:
      z_ik = LN(W_z[phi_E(E_ik); phi_R(R_ik); phi_P(P_ik); phi_T(T_ik)])
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        h = cfg["hidden_dim"]
        self.hidden_dim = h
        self.max_hops = int(cfg.get("max_hops", 4))
        self.entity_emb = nn.Embedding(cfg.get("entity_vocab_size", 50000), h)
        self.type_emb = nn.Embedding(cfg.get("type_vocab_size", 256), h)
        self.relation_emb = nn.Embedding(cfg.get("relation_vocab_size", 4096), h)
        self.direction_emb = nn.Embedding(cfg.get("direction_vocab_size", 8), h)
        self.pos_emb = nn.Embedding(self.max_hops + 1, h)
        self.entity_pool = AttnPool(h)
        self.rel_ffn = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h))
        self.rel_pool = AttnPool(h)
        # path structure: [L, directions L, types L, endpoint indicator] => fixed approximate size
        p_in = 1 + self.max_hops + self.max_hops + 1
        self.path_mlp = nn.Sequential(nn.Linear(p_in, h), nn.GELU(), nn.Linear(h, h))
        self.topo_mlp = nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, h))
        self.W_z = nn.Linear(4 * h, h)
        self.ln = nn.LayerNorm(h)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_ids = batch["path_node_ids"]      # [N,K,L+1]
        type_ids = batch["path_type_ids"]
        rel_ids = batch["path_rel_ids"]        # [N,K,L]
        dir_ids = batch["path_dir_ids"]
        topo = batch["path_topology"].float()  # [N,K,4]
        pmask = batch["path_mask"]             # [N,K]
        N, K, Lp1 = node_ids.shape
        L = rel_ids.shape[-1]
        device = node_ids.device
        pos_nodes = torch.arange(Lp1, device=device).view(1, 1, Lp1)
        e_repr = self.entity_emb(node_ids) + self.type_emb(type_ids) + self.pos_emb(pos_nodes)
        phi_E = self.entity_pool(e_repr)  # [N,K,d]
        pos_rels = torch.arange(L, device=device).view(1, 1, L)
        r_repr = self.relation_emb(rel_ids) + self.direction_emb(dir_ids) + self.pos_emb(pos_rels)
        r_enc = self.rel_ffn(r_repr)
        phi_R = self.rel_pool(r_enc)
        # Path features: normalized hop length, directions, type ids, endpoint match placeholder=1.
        max_h = self.max_hops
        pfeat = torch.zeros(N, K, 1 + max_h + max_h + 1, device=device)
        pfeat[..., 0] = float(L) / max(max_h, 1)
        d = dir_ids.float() / 2.0
        t = type_ids[..., 1:].float() / max(self.type_emb.num_embeddings - 1, 1)
        pfeat[..., 1:1+L] = d[..., :max_h]
        pfeat[..., 1+max_h:1+max_h+min(max_h, t.shape[-1])] = t[..., :max_h]
        pfeat[..., -1] = 1.0
        phi_P = self.path_mlp(pfeat)
        phi_T = self.topo_mlp(topo)
        z = self.ln(self.W_z(torch.cat([phi_E, phi_R, phi_P, phi_T], dim=-1)))
        z = z * pmask.unsqueeze(-1).float()
        return {"z": z, "phi_E": phi_E, "phi_R": phi_R, "phi_P": phi_P, "phi_T": phi_T, "relation_embeds": self.relation_emb.weight}


class RelationSemanticPrior(nn.Module):
    """Computes d_ik, E_K, and p_K.

    Implements:
      d_ik = [D_ent, D_rel, D_path, D_topo]^T
      E_K(q,a_i,m_ik) = beta^T d_ik, beta = softplus(gamma)
      p_K(m_ik|q,a_i) = softmax_k(-E_K)
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        self.max_hops = int(cfg.get("max_hops", 4))
        self.type_vocab_size = int(cfg.get("type_vocab_size", 256))
        self.relation_vocab_size = int(cfg.get("relation_vocab_size", 4096))
        self.gamma = nn.Parameter(torch.zeros(4))
        self.topo_linear = nn.Linear(4, 1)
        self.use_relaxed = bool(cfg.get("use_relaxed_relation_similarity", True))

    @staticmethod
    def _exact_match_dist(observed: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
        # observed [N,K,L], expected [L]
        exp = expected.view(*([1] * (observed.dim() - 1)), -1)
        return (observed != exp).float().mean(dim=-1)

    def entity_distance(self, path_type_ids: torch.Tensor, schema_type_ids: torch.Tensor) -> torch.Tensor:
        Lp1 = path_type_ids.shape[-1]
        exp = schema_type_ids[:Lp1].view(1, 1, Lp1)
        return (path_type_ids != exp).float().mean(dim=-1)

    def relation_distance(self, path_rel_ids: torch.Tensor, schema_rel_ids: torch.Tensor, schema_rel_mask: torch.Tensor, relation_embeds: torch.Tensor | None = None) -> torch.Tensor:
        # path_rel_ids [N,K,L], schema_rel_ids [L,Rmax]
        N, K, L = path_rel_ids.shape
        Rmax = schema_rel_ids.shape[-1]
        obs = path_rel_ids.unsqueeze(-1).expand(N, K, L, Rmax)
        exp = schema_rel_ids[:L].view(1, 1, L, Rmax)
        mask = schema_rel_mask[:L].view(1, 1, L, Rmax)
        exact = (obs == exp).float().masked_fill(~mask, 0.0)
        max_sim = exact.max(dim=-1).values
        if self.use_relaxed and relation_embeds is not None:
            # relaxed embedding cosine similarity to admissible relation family
            obs_e = relation_embeds[path_rel_ids]  # [N,K,L,d]
            exp_e = relation_embeds[schema_rel_ids[:L].clamp_min(0)]  # [L,Rmax,d]
            sim = F.cosine_similarity(obs_e.unsqueeze(-2), exp_e.view(1,1,L,Rmax,-1), dim=-1)
            sim = ((sim + 1) / 2.0).masked_fill(~mask, 0.0)
            max_sim = torch.maximum(max_sim, 0.25 * sim.max(dim=-1).values + 0.75 * max_sim)
        return (1.0 - max_sim).mean(dim=-1)

    def path_distance(self, path_type_ids: torch.Tensor, schema_type_ids: torch.Tensor, path_mask: torch.Tensor) -> torch.Tensor:
        # Simple schema shape mismatch plus endpoint consistency.
        shape = self.entity_distance(path_type_ids, schema_type_ids)
        endpoint_mismatch = (path_type_ids[..., -1] != schema_type_ids[min(path_type_ids.shape[-1]-1, len(schema_type_ids)-1)]).float()
        return 0.7 * shape + 0.3 * endpoint_mismatch

    def topology_distance(self, path_topology: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.topo_linear(path_topology)).squeeze(-1)

    def forward(self, batch: Dict[str, torch.Tensor], relation_embeds: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        D_ent = self.entity_distance(batch["path_type_ids"], batch["schema_type_ids"])
        D_rel = self.relation_distance(batch["path_rel_ids"], batch["schema_rel_ids"], batch["schema_rel_mask"], relation_embeds)
        D_path = self.path_distance(batch["path_type_ids"], batch["schema_type_ids"], batch["path_mask"])
        D_topo = self.topology_distance(batch["path_topology"].float())
        d = torch.stack([D_ent, D_rel, D_path, D_topo], dim=-1)
        beta = F.softplus(self.gamma)
        energy = torch.matmul(d, beta)
        energy = energy.masked_fill(~batch["path_mask"], 1e9)
        prior = torch.softmax(-energy, dim=-1)
        prior = prior * batch["path_mask"].float()
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)
        return {"d": d, "beta": beta, "energy": energy, "prior": prior}
