from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    entity_type: str
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def text(self) -> str:
        aliases = "; ".join(self.aliases)
        return f"{self.name} [{self.entity_type}]" + (f" aliases: {aliases}" if aliases else "")


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str
    direction: int = 1
    provenance: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class GroundingCandidate:
    entity_id: str
    probability: float
    score: float


@dataclass(frozen=True)
class RelationSchema:
    hops: int
    entity_types: tuple[str, ...]
    relations: tuple[str, ...]
    directions: tuple[int, ...]
    probability: float = 1.0

    def __post_init__(self) -> None:
        if self.hops < 1:
            raise ValueError("Schema must contain at least one hop.")
        if len(self.entity_types) != self.hops + 1:
            raise ValueError("entity_types must have hops + 1 entries.")
        if len(self.relations) != self.hops or len(self.directions) != self.hops:
            raise ValueError("relations and directions must have exactly hops entries.")


@dataclass(frozen=True)
class KGPath:
    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    directions: tuple[int, ...]
    search_score: float = 0.0

    def __post_init__(self) -> None:
        if len(self.nodes) != len(self.relations) + 1:
            raise ValueError("A path must contain one more node than relations.")
        if len(self.relations) != len(self.directions):
            raise ValueError("relations and directions must have equal length.")

    @property
    def start(self) -> str:
        return self.nodes[0]

    @property
    def end(self) -> str:
        return self.nodes[-1]

    @property
    def hops(self) -> int:
        return len(self.relations)

    def extend(self, edge: Edge, score: float) -> "KGPath":
        return KGPath(
            nodes=self.nodes + (edge.target,),
            relations=self.relations + (edge.relation,),
            directions=self.directions + (edge.direction,),
            search_score=score,
        )


@dataclass
class CandidatePrediction:
    entity_id: str
    score: float
    probability: float
    path_scores: list[float]
    supporting_paths: list[KGPath]
    attentions: list[Any] = field(default_factory=list)


@dataclass
class InferenceResult:
    question: str
    groundings: list[GroundingCandidate]
    schemas: list[RelationSchema]
    predictions: list[CandidatePrediction]
    answer_ids: list[str]
