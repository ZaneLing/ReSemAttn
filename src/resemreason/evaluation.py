from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from . import metrics


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_validity_labels(path):
    labels = {}
    if not path:
        return labels
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = metrics.canonical_path_key(
            row["question_id"], row["nodes"], row["relations"], row["directions"]
        )
        label = row["valid"]
        if label not in (0, 1, None):
            raise ValueError("Validity must be 0, 1, or null (unresolved).")
        if key in labels and labels[key] != label:
            raise ValueError("Conflicting independent labels require adjudication.")
        labels[key] = label
    return labels


def build_row(example, result, kg, labels):
    gold = set(example.gold_answers)
    ranked = [p.entity_id for p in result.predictions]
    row = {
        "question_id": example.question_id,
        "source_id": example.source_id,
        "gold_answers": list(example.gold_answers),
        "answer_ids": result.answer_ids,
        "ranked_ids": ranked,
        "predictions": [
            dict(
                entity_id=p.entity_id,
                score=p.score,
                probability=p.probability,
                paths=[asdict(path) for path in p.supporting_paths],
                path_scores=p.path_scores,
            )
            for p in result.predictions
        ],
        **metrics.answer_row(ranked, result.answer_ids, gold),
        **metrics.schema_metrics(result.schemas, example.gold_schemas),
    }
    top = result.predictions[0] if result.predictions else None
    path = top.supporting_paths[0] if top and top.supporting_paths else None
    if path is None or path.end != top.entity_id:
        row.update(top_path_valid=0, validity_source="missing_path")
    else:
        kg.validate_path(path)
        key = metrics.canonical_path_key(
            example.question_id, path.nodes, path.relations, path.directions
        )
        validity = labels.get(key)
        row.update(
            top_path_key=key,
            top_path_valid=validity,
            validity_source="independent" if validity is not None else "unresolved",
        )
    from .schema import SoftSchemaPredictor

    row["SchemaConfidence"] = SoftSchemaPredictor.confidence(result.schemas)
    pool_labels = [
        labels.get(
            metrics.canonical_path_key(example.question_id, p.nodes, p.relations, p.directions)
        )
        for prediction in result.predictions
        for p in prediction.supporting_paths
    ]
    row["AcceptedAny"] = float(bool(result.answer_ids))
    row["pool_paths"] = len(pool_labels)
    row["all_invalid_pool"] = (
        all(v == 0 for v in pool_labels)
        if pool_labels and all(v is not None for v in pool_labels)
        else None
    )
    return row


def select_threshold(dev_rows):
    if not dev_rows:
        raise ValueError("Threshold fitting requires nonempty development predictions.")
    choices = []
    for i in range(1, 20):
        threshold = i / 20
        f1 = sum(
            metrics.set_scores(
                [p["entity_id"] for p in r["predictions"] if p["probability"] >= threshold],
                r["gold_answers"],
            )["SetF1"]
            for r in dev_rows
        ) / len(dev_rows)
        choices.append((f1, -threshold))
    score, negative_threshold = max(choices)
    return -negative_threshold, score


def checkpoint_for_evaluation(pipeline, path, allow_untrained_toy=False):
    if path:
        pipeline.load(path)
        return sha256(path)
    if allow_untrained_toy and pipeline.config.get("data", {}).get("kind") == "toy":
        return "untrained_toy"
    raise ValueError(
        "Provide --checkpoint; only explicit data.kind=toy can use --allow-untrained-toy."
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
