from __future__ import annotations

import heapq
import math
from itertools import product

import torch
from torch import nn

from .graph import BiomedicalKG
from .text_encoder import BaseTextEncoder
from .types import RelationSchema


class SoftSchemaPredictor(nn.Module):
    """Factorized question-only schema predictor with top-M hypothesis decoding."""

    def __init__(self, text_encoder: BaseTextEncoder, kg: BiomedicalKG, max_hops: int = 3):
        super().__init__()
        self.text_encoder = text_encoder
        self.kg = kg
        self.max_hops = max_hops
        hidden = text_encoder.hidden_dim
        self.hop_head = nn.Linear(hidden, max_hops)
        self.start_type_head = nn.Linear(hidden, len(kg.entity_types))
        self.target_type_head = nn.Linear(hidden, len(kg.entity_types))
        self.relation_heads = nn.ModuleList(
            nn.Linear(hidden, len(kg.relations)) for _ in range(max_hops)
        )
        self.direction_heads = nn.ModuleList(nn.Linear(hidden, 2) for _ in range(max_hops))
        self.bridge_type_heads = nn.ModuleList(
            nn.Linear(hidden, len(kg.entity_types)) for _ in range(max_hops - 1)
        )

    def forward(
        self, question: str, device: torch.device | str = "cpu"
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        vector = self.text_encoder.encode([question], device=device).pooled[0].float()
        return {
            "hop": self.hop_head(vector),
            "start_type": self.start_type_head(vector),
            "target_type": self.target_type_head(vector),
            "relations": [head(vector) for head in self.relation_heads],
            "directions": [head(vector) for head in self.direction_heads],
            "bridge_types": [head(vector) for head in self.bridge_type_heads],
        }

    @torch.no_grad()
    def predict(
        self,
        question: str,
        top_m: int = 4,
        beam_size: int = 12,
        device: torch.device | str = "cpu",
    ) -> list[RelationSchema]:
        if top_m < 1 or beam_size < top_m:
            raise ValueError("Require 1 <= top_m <= beam_size.")
        logits = self.forward(question, device=device)
        hop_logp = torch.log_softmax(logits["hop"], dim=-1)
        start_logp = torch.log_softmax(logits["start_type"], dim=-1)
        target_logp = torch.log_softmax(logits["target_type"], dim=-1)
        rel_logps = [torch.log_softmax(item, dim=-1) for item in logits["relations"]]
        dir_logps = [torch.log_softmax(item, dim=-1) for item in logits["directions"]]
        bridge_logps = [torch.log_softmax(item, dim=-1) for item in logits["bridge_types"]]

        type_top = min(2, len(self.kg.entity_types))
        rel_top = min(3, len(self.kg.relations))
        heap = []
        serial = 0

        for hops in range(1, self.max_hops + 1):
            start_vals, start_ids = start_logp.topk(type_top)
            target_vals, target_ids = target_logp.topk(type_top)
            rel_options = [rel_logps[idx].topk(rel_top) for idx in range(hops)]
            dir_options = [dir_logps[idx].topk(2) for idx in range(hops)]
            bridge_options = [bridge_logps[idx].topk(type_top) for idx in range(max(0, hops - 1))]

            for s_pos, t_pos in product(range(type_top), repeat=2):
                rel_ranges = [range(len(values)) for values, _ in rel_options]
                dir_ranges = [range(len(values)) for values, _ in dir_options]
                bridge_ranges = [range(len(values)) for values, _ in bridge_options]
                bridge_products = list(product(*bridge_ranges)) if bridge_ranges else [()]
                for rel_choice in product(*rel_ranges):
                    for dir_choice in product(*dir_ranges):
                        for bridge_choice in bridge_products:
                            score = float(
                                hop_logp[hops - 1] + start_vals[s_pos] + target_vals[t_pos]
                            )
                            relations = []
                            directions = []
                            entity_types = [self.kg.entity_types[start_ids[s_pos].item()]]
                            for idx in range(hops):
                                rel_values, rel_ids = rel_options[idx]
                                dir_values, dir_ids = dir_options[idx]
                                score += float(
                                    rel_values[rel_choice[idx]] + dir_values[dir_choice[idx]]
                                )
                                relations.append(self.kg.relations[rel_ids[rel_choice[idx]].item()])
                                directions.append(1 if dir_ids[dir_choice[idx]].item() == 1 else -1)
                                if idx < hops - 1:
                                    bridge_values, bridge_ids = bridge_options[idx]
                                    score += float(bridge_values[bridge_choice[idx]])
                                    entity_types.append(
                                        self.kg.entity_types[bridge_ids[bridge_choice[idx]].item()]
                                    )
                            entity_types.append(self.kg.entity_types[target_ids[t_pos].item()])
                            schema = RelationSchema(
                                hops=hops,
                                entity_types=tuple(entity_types),
                                relations=tuple(relations),
                                directions=tuple(directions),
                                probability=math.exp(score),
                            )
                            key = (score, serial, schema)
                            serial += 1
                            if len(heap) < beam_size:
                                heapq.heappush(heap, key)
                            elif score > heap[0][0]:
                                heapq.heapreplace(heap, key)

        selected = sorted(heap, key=lambda row: (-row[0], row[1]))[:top_m]
        ranked = [item[2] for item in selected]
        probabilities = torch.softmax(
            torch.tensor([item[0] for item in selected], dtype=torch.float64), 0
        ).tolist()
        return [
            RelationSchema(
                hops=schema.hops,
                entity_types=schema.entity_types,
                relations=schema.relations,
                directions=schema.directions,
                probability=probability,
            )
            for schema, probability in zip(ranked, probabilities)
        ]

    def log_probabilities(self, logits: dict, schemas: list[RelationSchema]) -> torch.Tensor:
        """Factorized log likelihood, before the retained top-M renormalization."""
        outputs = []
        for schema in schemas:
            if schema.hops > self.max_hops:
                raise ValueError("Gold schema exceeds configured maximum hops.")
            lp = torch.log_softmax(logits["hop"], -1)[schema.hops - 1]
            for name, entity_type in [
                ("start_type", schema.entity_types[0]),
                ("target_type", schema.entity_types[-1]),
            ]:
                lp = lp + torch.log_softmax(logits[name], -1)[self.kg.type_to_id[entity_type]]
            for j, relation in enumerate(schema.relations):
                lp = (
                    lp
                    + torch.log_softmax(logits["relations"][j], -1)[
                        self.kg.relation_to_id[relation]
                    ]
                )
                lp = (
                    lp
                    + torch.log_softmax(logits["directions"][j], -1)[int(schema.directions[j] == 1)]
                )
                if j < schema.hops - 1:
                    lp = (
                        lp
                        + torch.log_softmax(logits["bridge_types"][j], -1)[
                            self.kg.type_to_id[schema.entity_types[j + 1]]
                        ]
                    )
            outputs.append(lp)
        return torch.stack(outputs)

    def supervision_loss(self, logits: dict, gold_schemas) -> torch.Tensor:
        # Marginal likelihood of the unique acceptable schema set; never inject gold at inference.
        unique = {(s.hops, s.entity_types, s.relations, s.directions): s for s in gold_schemas}
        if not unique:
            return logits["hop"].sum() * 0
        return -torch.logsumexp(self.log_probabilities(logits, list(unique.values())), 0)

    @staticmethod
    def confidence_tensor(probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.numel() == 1:
            return probabilities.new_ones(())
        rho = probabilities / probabilities.sum()
        entropy = -(rho * rho.clamp_min(torch.finfo(rho.dtype).tiny).log()).sum()
        return (1 - entropy / math.log(len(rho))).clamp(0, 1)

    @staticmethod
    def confidence(schemas: list[RelationSchema]) -> float:
        if not schemas:
            return 0.0
        return float(
            SoftSchemaPredictor.confidence_tensor(
                torch.tensor([s.probability for s in schemas], dtype=torch.float64)
            )
        )
