from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .types import Edge, Entity


class BiomedicalKG:
    """Directed biomedical KG with explicit inverse traversal records."""

    def __init__(self, entities: Iterable[Entity], edges: Iterable[Edge], add_inverse: bool = True):
        self.entities = {entity.entity_id: entity for entity in entities}
        self._adjacency: dict[str, list[Edge]] = defaultdict(list)
        self._stored_edges: list[Edge] = []

        for edge in edges:
            if edge.source not in self.entities or edge.target not in self.entities:
                raise ValueError(f"Edge references unknown entity: {edge}")
            self._stored_edges.append(edge)
            self._adjacency[edge.source].append(edge)
            if add_inverse:
                self._adjacency[edge.target].append(
                    Edge(
                        source=edge.target,
                        relation=edge.relation,
                        target=edge.source,
                        direction=-edge.direction,
                        provenance=edge.provenance,
                        metadata={**edge.metadata, "inverse_of": (edge.source, edge.target)},
                    )
                )

        self.entity_types = sorted({entity.entity_type for entity in self.entities.values()})
        self.relations = sorted({edge.relation for edge in self._stored_edges})
        self.type_to_id = {name: idx for idx, name in enumerate(self.entity_types)}
        self.relation_to_id = {name: idx for idx, name in enumerate(self.relations)}

    @classmethod
    def from_jsonl(
        cls,
        entity_path: str | Path,
        edge_path: str | Path,
        add_inverse: bool = True,
    ) -> "BiomedicalKG":
        entities = []
        with Path(entity_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                entities.append(
                    Entity(
                        entity_id=row["id"],
                        name=row["name"],
                        entity_type=row["type"],
                        aliases=tuple(row.get("aliases", [])),
                        metadata=row.get("metadata", {}),
                    )
                )

        edges = []
        with Path(edge_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                edges.append(
                    Edge(
                        source=row["source"],
                        relation=row["relation"],
                        target=row["target"],
                        direction=int(row.get("direction", 1)),
                        provenance=row.get("provenance"),
                        metadata=row.get("metadata", {}),
                    )
                )
        return cls(entities, edges, add_inverse=add_inverse)

    def neighbors(self, entity_id: str) -> list[Edge]:
        return self._adjacency.get(entity_id, [])

    def degree(self, entity_id: str) -> int:
        return len(self._adjacency.get(entity_id, []))

    def entity(self, entity_id: str) -> Entity:
        try:
            return self.entities[entity_id]
        except KeyError as exc:
            raise KeyError(f"Unknown entity: {entity_id}") from exc

    def describe_path(self, nodes: tuple[str, ...], relations: tuple[str, ...], directions: tuple[int, ...]) -> str:
        chunks = [self.entity(nodes[0]).name]
        for idx, relation in enumerate(relations):
            arrow = "->" if directions[idx] == 1 else "<-"
            chunks.extend([f"-{relation}{arrow}", self.entity(nodes[idx + 1]).name])
        return " ".join(chunks)
