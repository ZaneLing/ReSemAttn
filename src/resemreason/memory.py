from __future__ import annotations

import math

import torch
from torch import nn

from .graph import BiomedicalKG
from .types import KGPath, RelationSchema


class PathMemoryEncoder(nn.Module):
    def __init__(
        self,
        kg: BiomedicalKG,
        hidden_dim: int,
        type_dim: int = 32,
        relation_dim: int = 64,
        topology_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kg = kg
        self.hidden_dim = hidden_dim
        self.type_embedding = nn.Embedding(max(1, len(kg.entity_types)), type_dim)
        self.relation_embedding = nn.Embedding(max(1, len(kg.relations)), relation_dim)
        self.direction_embedding = nn.Embedding(2, relation_dim)
        self.path_mlp = nn.Sequential(
            nn.Linear(4, topology_dim), nn.GELU(), nn.Linear(topology_dim, topology_dim)
        )
        self.topology_mlp = nn.Sequential(
            nn.Linear(4, topology_dim), nn.GELU(), nn.Linear(topology_dim, topology_dim)
        )
        input_dim = type_dim + relation_dim + topology_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.relation_projection = nn.Linear(relation_dim, hidden_dim)

    def _path_features(self, path: KGPath, device: torch.device | str) -> torch.Tensor:
        unique_ratio = len(set(path.nodes)) / max(1, len(path.nodes))
        return torch.tensor(
            [
                float(path.hops),
                unique_ratio,
                float(path.nodes[0] == path.nodes[-1]),
                float(path.search_score),
            ],
            dtype=torch.float32,
            device=device,
        )

    def _topology_features(self, path: KGPath, device: torch.device | str) -> torch.Tensor:
        degrees = [math.log1p(self.kg.degree(node)) for node in path.nodes]
        return torch.tensor(
            [
                sum(degrees) / len(degrees),
                max(degrees),
                degrees[-1],
                sum(1 for degree in degrees if degree > math.log(10)) / len(degrees),
            ],
            dtype=torch.float32,
            device=device,
        )

    def encode_paths(
        self,
        paths: list[KGPath],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not paths:
            raise ValueError("At least one path is required.")
        memories = []
        relation_vectors = []
        topology_penalties = []
        for path in paths:
            if path.hops < 1:
                raise ValueError("Evidence memory requires at least one edge per path.")
            type_ids = torch.tensor(
                [self.kg.type_to_id[self.kg.entity(node).entity_type] for node in path.nodes],
                device=device,
            )
            relation_ids = torch.tensor(
                [self.kg.relation_to_id[relation] for relation in path.relations],
                device=device,
            )
            direction_ids = torch.tensor(
                [1 if direction == 1 else 0 for direction in path.directions],
                device=device,
            )
            entity_component = self.type_embedding(type_ids).mean(dim=0)
            relation_component = (
                self.relation_embedding(relation_ids) + self.direction_embedding(direction_ids)
            ).mean(dim=0)
            path_component = self.path_mlp(self._path_features(path, device))
            topology_raw = self._topology_features(path, device)
            topology_component = self.topology_mlp(topology_raw)
            memory = self.projection(
                torch.cat(
                    [entity_component, relation_component, path_component, topology_component]
                )
            )
            memories.append(memory)
            relation_vectors.append(self.relation_projection(relation_component))
            topology_penalties.append(topology_raw[0])
        return (
            torch.stack(memories),
            torch.stack(relation_vectors),
            torch.stack(topology_penalties),
        )

    def mismatch(
        self, schema: RelationSchema, path: KGPath, topology_penalty: torch.Tensor
    ) -> torch.Tensor:
        entity_mismatch = 0.0
        for idx, node in enumerate(path.nodes[: len(schema.entity_types)]):
            entity_mismatch += float(self.kg.entity(node).entity_type != schema.entity_types[idx])
        entity_mismatch /= max(1, min(len(path.nodes), len(schema.entity_types)))
        relation_mismatch = 0.0
        for idx, relation in enumerate(path.relations[: schema.hops]):
            relation_mismatch += float(relation != schema.relations[idx])
            relation_mismatch += 0.75 * float(path.directions[idx] != schema.directions[idx])
        relation_mismatch /= max(1, min(path.hops, schema.hops))
        path_mismatch = abs(path.hops - schema.hops) + float(path.nodes[0] == path.nodes[-1])
        return torch.stack(
            [
                topology_penalty.new_tensor(entity_mismatch),
                topology_penalty.new_tensor(relation_mismatch),
                topology_penalty.new_tensor(path_mismatch),
                topology_penalty,
            ]
        )


class RelationSemanticPrior(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        paths: list[KGPath],
        schemas: list[RelationSchema],
        topology_penalties: torch.Tensor,
        memory_encoder: PathMemoryEncoder,
        schema_probabilities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not schemas or not paths:
            raise ValueError("Paths and schemas must be nonempty.")
        rho = (
            topology_penalties.new_tensor([s.probability for s in schemas])
            if schema_probabilities is None
            else schema_probabilities
        )
        if (
            rho.shape != (len(schemas),)
            or not torch.isfinite(rho).all()
            or (rho < 0).any()
            or rho.sum() <= 0
        ):
            raise ValueError("Invalid schema probabilities.")
        rho = rho / rho.sum()
        beta = torch.nn.functional.softplus(self.gamma)
        path_scores = []
        for path_idx, path in enumerate(paths):
            schema_scores = []
            for schema_idx, schema in enumerate(schemas):
                mismatch = memory_encoder.mismatch(schema, path, topology_penalties[path_idx])
                energy = beta @ mismatch
                schema_scores.append(
                    rho[schema_idx].clamp_min(torch.finfo(rho.dtype).tiny).log() - energy
                )
            path_scores.append(torch.logsumexp(torch.stack(schema_scores), dim=0))
        absolute = torch.stack(path_scores)
        prior = torch.softmax(absolute, dim=0)
        return absolute, prior
