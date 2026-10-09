#!/usr/bin/env python
"""E1/E2/E5: freeze retrieval once, rescore exactly those paths with real checkpoints."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from resemreason.data import load_questions, parse_path
from resemreason.evaluation import (
    build_row,
    checkpoint_for_evaluation,
    load_validity_labels,
    write_json,
)
from resemreason.metrics import summarize_rows
from resemreason.pipeline import ReSemReasonPipeline
from resemreason.types import GroundingCandidate, RelationSchema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["export", "score"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-untrained-toy", action="store_true")
    parser.add_argument("--questions")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--validity-labels")
    parser.add_argument("--output", default="outputs/frozen")
    parser.add_argument("--variant")
    args = parser.parse_args()
    p = ReSemReasonPipeline.from_yaml(args.config, variant=args.variant)
    ckpt = checkpoint_for_evaluation(p, args.checkpoint, args.allow_untrained_toy)
    examples = load_questions(p._resolve(args.questions or p.config["data"]["questions"]))
    if args.mode == "export":
        records = []
        for e in examples:
            anchors, schemas, pools = p.retrieve(e.question)
            records.append(
                dict(
                    question_id=e.question_id,
                    question=e.question,
                    groundings=[asdict(a) for a in anchors],
                    schemas=[asdict(s) for s in schemas],
                    paths={c: [asdict(path) for path in paths] for c, paths in pools.items()},
                    retriever_checkpoint=ckpt,
                    graph_checksums=p.graph_checksums(),
                    seed=p.config["seed"],
                )
            )
        out = Path(args.pool)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
        print(f"Frozen {len(records)} pools: {out}")
        return
    payload = Path(args.pool).read_bytes()
    pool_hash = hashlib.sha256(payload).hexdigest()
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    lookup = {e.question_id: e for e in examples}
    if len(records) != len(lookup) or {r["question_id"] for r in records} != set(lookup):
        raise ValueError("Frozen pool and evaluation question sets must agree exactly.")
    labels = load_validity_labels(args.validity_labels)
    rows = []
    for r in records:
        e = lookup[r["question_id"]]
        if r["question"] != e.question or r["graph_checksums"] != p.graph_checksums():
            raise ValueError("Frozen question/graph mismatch.")
        anchors = [GroundingCandidate(**a) for a in r["groundings"]]
        schemas = [
            RelationSchema(
                s["hops"],
                tuple(s["entity_types"]),
                tuple(s["relations"]),
                tuple(s["directions"]),
                s["probability"],
            )
            for s in r["schemas"]
        ]
        pools = {c: [parse_path(v) for v in paths] for c, paths in r["paths"].items()}
        result = p.infer_pool(e.question, anchors, schemas, pools)
        row = build_row(e, result, p.kg, labels)
        row.update(pool_sha256=pool_hash, checkpoint_sha256=ckpt)
        rows.append(row)
    if Path(args.pool).read_bytes() != payload:
        raise RuntimeError("Pool changed during scoring.")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    write_json(
        out / "summary.json",
        dict(metrics=summarize_rows(rows), pool_sha256=pool_hash, checkpoint_sha256=ckpt),
    )


if __name__ == "__main__":
    main()
