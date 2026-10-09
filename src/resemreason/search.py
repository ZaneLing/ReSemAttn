from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from .graph import BiomedicalKG
from .text_encoder import BaseTextEncoder
from .types import GroundingCandidate, KGPath, RelationSchema, path_key


class RelationGuidedSearcher:
    def __init__(
        self,
        kg: BiomedicalKG,
        text_encoder: BaseTextEncoder,
        config: dict,
    ) -> None:
        self.strategy = config.get("strategy", "guided")
        if self.strategy not in ("guided", "bfs"):
            raise ValueError("Search strategy must be guided or bfs.")
        self.kg = kg
        self.text_encoder = text_encoder
        self.beam_width = int(config.get("beam_width", 24))
        self.max_hops = int(config.get("max_hops", 3))
        self.max_expansions = int(config.get("max_expansions", 1000))
        self.max_paths_per_candidate = int(config.get("max_paths_per_candidate", 8))
        self.lambda_ground = float(config.get("lambda_ground", 1.0))
        self.lambda_schema = float(config.get("lambda_schema", 2.0))
        self.lambda_semantic = float(config.get("lambda_semantic", 0.5))
        self.lambda_hub = float(config.get("lambda_hub", 0.15))
        self.relation_mismatch = float(config.get("relation_mismatch", 1.0))
        self.direction_mismatch = float(config.get("direction_mismatch", 0.75))
        self.type_mismatch = float(config.get("type_mismatch", 0.5))
        self.length_mismatch = float(config.get("length_mismatch", 0.5))

    def _distance(self, schema: RelationSchema, path: KGPath, partial: bool = True) -> float:
        distance = 0.0
        limit = min(path.hops, schema.hops)
        for idx in range(limit):
            if path.relations[idx] != schema.relations[idx]:
                distance += self.relation_mismatch
            if path.directions[idx] != schema.directions[idx]:
                distance += self.direction_mismatch
            node_type = self.kg.entity(path.nodes[idx + 1]).entity_type
            if node_type != schema.entity_types[idx + 1]:
                distance += self.type_mismatch
        if not partial:
            distance += self.length_mismatch * abs(path.hops - schema.hops)
        return distance

    def _compatibility(
        self, schemas: list[RelationSchema], path: KGPath, partial: bool = True
    ) -> float:
        values = [
            math.log(max(schema.probability, 1e-12)) - self._distance(schema, path, partial=partial)
            for schema in schemas
        ]
        tensor = torch.tensor(values, dtype=torch.float32)
        return float(torch.logsumexp(tensor, dim=0).item())

    def _endpoint_semantics(
        self,
        question_vector: torch.Tensor,
        endpoint_id: str,
        device: torch.device | str,
    ) -> float:
        entity_vector = self.text_encoder.encode(
            [self.kg.entity(endpoint_id).text], device=device
        ).pooled[0]
        return float(F.cosine_similarity(question_vector, entity_vector, dim=0).item())

    @torch.no_grad()
    def search(
        self,
        question: str,
        groundings: list[GroundingCandidate],
        schemas: list[RelationSchema],
        device: torch.device | str = "cpu",
    ) -> dict[str, list[KGPath]]:
        self.last_stats = {"edge_attempts": 0, "unique_nodes": 0}
        if not groundings or not schemas:
            return {}
        question_vector = (
            self.text_encoder.encode([question], device=device).pooled[0]
            if self.strategy == "guided"
            else None
        )
        beams: list[tuple[KGPath, float]] = []
        grounding_probability = {item.entity_id: item.probability for item in groundings}
        for grounding in groundings:
            path = KGPath(
                nodes=(grounding.entity_id,), relations=(), directions=(), search_score=0.0
            )
            beams.append((path, self.lambda_ground * math.log(max(grounding.probability, 1e-12))))

        completed: list[KGPath] = []
        expansions = 0
        visited = set()
        for _hop in range(1, self.max_hops + 1):
            next_beam: list[tuple[KGPath, float]] = []
            for path, _ in beams:
                for edge in sorted(
                    self.kg.neighbors(path.end), key=lambda e: (e.target, e.relation, e.direction)
                ):
                    if expansions >= self.max_expansions:
                        break
                    expansions += 1
                    visited.add(edge.target)
                    if edge.target in path.nodes:
                        continue
                    provisional = path.extend(edge, score=0.0)
                    if self.strategy == "bfs":
                        score = -float(provisional.hops)
                    else:
                        compatibility = self._compatibility(schemas, provisional, partial=True)
                        semantic = self._endpoint_semantics(question_vector, edge.target, device)
                        hub = math.log1p(self.kg.degree(edge.target))
                        anchor_prob = grounding_probability.get(provisional.start, 1e-12)
                        score = (
                            self.lambda_ground * math.log(max(anchor_prob, 1e-12))
                            + self.lambda_schema * compatibility
                            + self.lambda_semantic * semantic
                            - self.lambda_hub * hub
                        )
                    extended = KGPath(
                        nodes=provisional.nodes,
                        relations=provisional.relations,
                        directions=provisional.directions,
                        search_score=score,
                    )
                    next_beam.append((extended, score))
                    if any(schema.hops == extended.hops for schema in schemas):
                        completed.append(extended)
                if expansions >= self.max_expansions:
                    break
            next_beam.sort(key=lambda item: (-item[1], path_key(item[0])))
            beams = next_beam[: self.beam_width]
            if not beams or expansions >= self.max_expansions:
                break

        grouped: dict[str, list[KGPath]] = defaultdict(list)
        for path in sorted(completed, key=lambda item: (-item.search_score, path_key(item))):
            if len(grouped[path.end]) < self.max_paths_per_candidate:
                grouped[path.end].append(path)
        self.last_stats = {"edge_attempts": expansions, "unique_nodes": len(visited)}
        return dict(grouped)
