from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .types import RelationSchema


@dataclass
class QuestionExample:
    question_id: str
    question: str
    gold_answers: tuple[str, ...]
    gold_anchors: tuple[str, ...] = ()
    gold_schemas: tuple[RelationSchema, ...] = ()


def load_questions(path: str | Path) -> list[QuestionExample]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            schemas = []
            for item in row.get("gold_schemas", []):
                schemas.append(
                    RelationSchema(
                        hops=int(item["hops"]),
                        entity_types=tuple(item["entity_types"]),
                        relations=tuple(item["relations"]),
                        directions=tuple(int(value) for value in item["directions"]),
                        probability=float(item.get("probability", 1.0)),
                    )
                )
            examples.append(
                QuestionExample(
                    question_id=str(row["id"]),
                    question=row["question"],
                    gold_answers=tuple(row.get("gold_answers", [])),
                    gold_anchors=tuple(row.get("gold_anchors", [])),
                    gold_schemas=tuple(schemas),
                )
            )
    return examples
