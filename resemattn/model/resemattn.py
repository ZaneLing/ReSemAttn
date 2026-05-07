"""Full four-stage ReSemAttn model."""
from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn

from resemattn.model.memory import PathMemoryEncoder, RelationSemanticPrior
from resemattn.model.attention import RelationConditionedKV, RelationSemanticAttentionLayer


class LightTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~attn_mask.bool()
        y, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + self.drop(y))
        x = self.ln2(x + self.drop(self.ffn(x)))
        return x


class TextEncoder(nn.Module):
    """Compact Transformer-style encoder used as a runnable reference backbone.

    It uses lightweight multi-head self-attention blocks so the toy example runs on CPU.
    Replace this class with Qwen3-8B + LoRA for production paper runs.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        h = cfg["hidden_dim"]
        self.emb = nn.Embedding(cfg.get("vocab_size", 50000), h, padding_idx=0)
        self.pos = nn.Embedding(cfg.get("max_seq_len", 256), h)
        self.blocks = nn.ModuleList([
            LightTransformerBlock(h, cfg.get("n_heads", 4), cfg.get("dropout", 0.1))
            for _ in range(cfg.get("n_layers", 1))
        ])
        self.ln = nn.LayerNorm(h)

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        N, n = token_ids.shape
        pos = torch.arange(n, device=token_ids.device).view(1, n)
        h = self.emb(token_ids) + self.pos(pos)
        for block in self.blocks:
            h = block(h, attn_mask)
        return self.ln(h)


def span_mean(hidden: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
    # hidden [N,n,d], spans [N,2] inclusive start, exclusive-ish end
    outs = []
    for i in range(hidden.shape[0]):
        s = int(spans[i, 0].item())
        e = int(spans[i, 1].item())
        e = max(e, s + 1)
        e = min(e, hidden.shape[1])
        outs.append(hidden[i, s:e].mean(dim=0))
    return torch.stack(outs, dim=0)


class ReSemAttnModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        m = cfg["model"] if "model" in cfg else cfg
        h = m["hidden_dim"]
        self.text_encoder = TextEncoder(m)
        self.memory_encoder = PathMemoryEncoder(m)
        self.prior = RelationSemanticPrior(m)
        self.kv = RelationConditionedKV(h, m.get("relation_kv_bases", 4), m.get("relation_kv_rank", 32))
        self.resem_layer = RelationSemanticAttentionLayer(h)
        self.pool = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.LayerNorm(h))
        self.scorer = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, batch: Dict[str, torch.Tensor], save_attention: bool = False) -> Dict[str, torch.Tensor]:
        # Stage 2: path memory + prior.
        mem = self.memory_encoder(batch)
        prior_out = self.prior(batch, mem["relation_embeds"])
        kv = self.kv(mem["z"], mem["phi_R"])
        # Stage 3: relation-semantic attention.
        h = self.text_encoder(batch["token_ids"], batch["attn_mask"])
        schema_repr = span_mean(h, batch["schema_span"])
        cand_repr = span_mean(h, batch["candidate_span"])
        attn_out = self.resem_layer(h, kv["k"], kv["v"], prior_out["prior"], batch["path_mask"], schema_repr, cand_repr)
        h2 = attn_out["hidden"]
        # Stage 4: candidate scoring.
        seq_repr = h2[:, -1, :]
        cand_repr2 = span_mean(h2, batch["candidate_span"])
        schema_repr2 = span_mean(h2, batch["schema_span"])
        pooled = self.pool(torch.cat([seq_repr, cand_repr2, schema_repr2], dim=-1))
        scores = self.scorer(pooled).squeeze(-1)
        out = {"scores": scores, "prior": prior_out["prior"], "energy": prior_out["energy"], "d": prior_out["d"], "beta": prior_out["beta"]}
        if save_attention:
            out.update({"alpha": attn_out["alpha"], "gate": attn_out["gate"], "omega": kv["omega"]})
        return out
