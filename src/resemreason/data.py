from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .types import KGPath, RelationSchema


@dataclass
class QuestionExample:
    question_id: str
    question: str
    gold_answers: tuple[str, ...]
    gold_anchors: tuple[str, ...] = ()
    gold_schemas: tuple[RelationSchema, ...] = ()
    # Pairs must be matched and labeled upstream; gold endpoint alone is not path validity.
    candidate_pairs: tuple[tuple[str, str], ...] = ()
    path_pairs: tuple[tuple[KGPath, KGPath], ...] = ()
    path_pair_types: tuple[str, ...] = ()
    source_id: str | None = None
    split: str | None = None
    dataset: str = "biohopr"
    options: dict[str, tuple[str, ...]] = field(default_factory=dict)
    gold_option: str | None = None
    gold_texts: tuple[str, ...] = ()
    gold_relations: tuple[tuple[str, int], ...] = ()


def parse_path(row: dict) -> KGPath:
    return KGPath(
        tuple(row["nodes"]),
        tuple(row["relations"]),
        tuple(int(d) for d in row["directions"]),
        float(row.get("search_score", 0)),
    )


def load_questions(path: str | Path) -> list[QuestionExample]:
    examples, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["id"])
            if qid in seen:
                raise ValueError(f"Duplicate question ID: {qid}")
            seen.add(qid)
            schemas = tuple(
                RelationSchema(
                    hops=int(s["hops"]),
                    entity_types=tuple(s["entity_types"]),
                    relations=tuple(s["relations"]),
                    directions=tuple(s["directions"]),
                    probability=float(s.get("probability", 1.0)),
                )
                for s in row.get("gold_schemas", [])
            )
            path_pairs = []
            for pair in row.get("path_pairs", []):
                if pair.get("valid_label") != 1 or pair.get("invalid_label") != 0:
                    raise ValueError(
                        "Path pairs require explicit independent valid_label=1/invalid_label=0."
                    )
                valid, invalid = parse_path(pair["valid"]), parse_path(pair["invalid"])
                if valid.end != invalid.end:
                    raise ValueError("Same-answer path hinges require equal endpoints.")
                if valid.hops != invalid.hops:
                    raise ValueError("Matched path pairs require equal hop counts.")
                path_pairs.append((valid, invalid))
            examples.append(
                QuestionExample(
                    question_id=qid,
                    question=row["question"],
                    gold_answers=tuple(row.get("gold_answers", [])),
                    gold_anchors=tuple(row.get("gold_anchors", [])),
                    gold_schemas=schemas,
                    candidate_pairs=tuple(tuple(v) for v in row.get("candidate_pairs", [])),
                    path_pairs=tuple(path_pairs),
                    path_pair_types=tuple(
                        pair.get("negative_type", "unspecified")
                        for pair in row.get("path_pairs", [])
                    ),
                    source_id=row.get("source_id"),
                    split=row.get("split"),
                    dataset=row.get("dataset", "biohopr"),
                    options={str(k): tuple(v) for k, v in row.get("options", {}).items()},
                    gold_option=str(row["gold_option"])
                    if row.get("gold_option") is not None
                    else None,
                    gold_texts=tuple(row.get("gold_texts", [])),
                    gold_relations=tuple((r, int(d)) for r, d in row.get("gold_relations", [])),
                )
            )
    return examples
