from __future__ import annotations

from collections.abc import Sequence

from .graph import BiomedicalKG
from .types import KGPath, RelationSchema


def hits_at_k(ranked_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    return float(any(entity_id in gold_ids for entity_id in ranked_ids[:k]))


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: set[str]) -> float:
    for rank, entity_id in enumerate(ranked_ids, start=1):
        if entity_id in gold_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranked_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    top = ranked_ids[:k]
    return sum(entity_id in gold_ids for entity_id in top) / max(1, k)


def recall_at_k(ranked_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return sum(entity_id in gold_ids for entity_id in ranked_ids[:k]) / len(gold_ids)


def coverage(candidate_ids: Sequence[str], gold_ids: set[str]) -> float:
    return float(any(entity_id in gold_ids for entity_id in candidate_ids))


def schema_path_validity(kg: BiomedicalKG, path: KGPath, schemas: Sequence[RelationSchema]) -> bool:
    for schema in schemas:
        if path.hops != schema.hops:
            continue
        if tuple(path.relations) != tuple(schema.relations):
            continue
        if tuple(path.directions) != tuple(schema.directions):
            continue
        path_types = tuple(kg.entity(node).entity_type for node in path.nodes)
        if path_types == tuple(schema.entity_types):
            return True
    return False


def joint_at_1(
    top_answer: str | None,
    top_path: KGPath | None,
    gold_ids: set[str],
    kg: BiomedicalKG,
    gold_schemas: Sequence[RelationSchema],
) -> float:
    if top_answer is None or top_path is None:
        return 0.0
    return float(
        top_answer in gold_ids
        and top_path.end == top_answer
        and schema_path_validity(kg, top_path, gold_schemas)
    )
