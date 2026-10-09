from __future__ import annotations

import math

import torch
from torch import nn


class RelationConditionedProjection(nn.Module):
    def __init__(self, hidden_dim: int, bases: int = 4, rank: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bases = bases
        self.rank = rank
        self.base = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mixture = nn.Linear(hidden_dim, bases)
        self.u = nn.Parameter(torch.empty(bases, hidden_dim, rank))
        self.v = nn.Parameter(torch.empty(bases, hidden_dim, rank))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, memory: torch.Tensor, relation_vector: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.mixture(relation_vector), dim=-1)
        low_rank = []
        for basis in range(self.bases):
            projected = (memory @ self.v[basis]) @ self.u[basis].transpose(0, 1)
            low_rank.append(projected)
        stacked = torch.stack(low_rank, dim=1)
        return self.base(memory) + (stacked * weights.unsqueeze(-1)).sum(dim=1)


class ReSemAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        relation_bases: int = 4,
        relation_rank: int = 16,
        prior_strength: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = RelationConditionedProjection(hidden_dim, relation_bases, relation_rank)
        self.value = RelationConditionedProjection(hidden_dim, relation_bases, relation_rank)
        self.gate = nn.Linear(hidden_dim * 4, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.prior_strength = nn.Parameter(torch.tensor(float(prior_strength)))

    def forward(
        self,
        token_states: torch.Tensor,
        path_memory: torch.Tensor,
        relation_vectors: torch.Tensor,
        path_prior: torch.Tensor,
        schema_vector: torch.Tensor,
        candidate_vector: torch.Tensor,
        schema_confidence: float | torch.Tensor,
        relation_conditioned: bool = True,
        use_prior: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_states.dim() != 2 or path_memory.dim() != 2:
            raise ValueError("token_states and path_memory must be rank-2 tensors.")
        queries = self.query(token_states)
        keys = (
            self.key(path_memory, relation_vectors)
            if relation_conditioned
            else self.key.base(path_memory)
        )
        values = (
            self.value(path_memory, relation_vectors)
            if relation_conditioned
            else self.value.base(path_memory)
        )
        logits = queries @ keys.transpose(0, 1) / math.sqrt(self.hidden_dim)
        confidence = torch.as_tensor(schema_confidence, dtype=logits.dtype, device=logits.device)
        if use_prior:
            logits = logits + self.prior_strength * confidence * torch.log(path_prior + 1e-8)
        attention = torch.softmax(logits, dim=-1)
        context = attention @ values
        schema_expand = schema_vector.unsqueeze(0).expand_as(token_states)
        candidate_expand = candidate_vector.unsqueeze(0).expand_as(token_states)
        gate = torch.sigmoid(
            self.gate(torch.cat([token_states, context, schema_expand, candidate_expand], dim=-1))
        )
        updated = self.output_norm(token_states + self.dropout(gate * context))
        return updated, attention
