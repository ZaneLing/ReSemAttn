#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from resemreason.adapters import benchmark_outputs
from resemreason.data import load_questions
from resemreason.evaluation import (
    build_row,
    checkpoint_for_evaluation,
    load_validity_labels,
    sha256,
    write_json,
)
from resemreason.metrics import calibration_metrics, summarize_rows
from resemreason.pipeline import ReSemReasonPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-untrained-toy", action="store_true")
    parser.add_argument("--toy-oracle", action="store_true")
    parser.add_argument("--questions")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--validity-labels")
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--variant")
    args = parser.parse_args()
    p = ReSemReasonPipeline.from_yaml(args.config, variant=args.variant)
    ckpt = checkpoint_for_evaluation(p, args.checkpoint, args.allow_untrained_toy)
    toy = p.config["data"].get("kind") == "toy"
    if args.toy_oracle and not toy:
        parser.error("--toy-oracle is restricted to toy data.")
    data_path = p._resolve(
        args.questions or p.config["data"].get(args.split, p.config["data"]["questions"])
    )
    examples = load_questions(data_path)
    if any(e.split not in (None, args.split) for e in examples):
        parser.error("Question split does not match --split.")
    labels = load_validity_labels(args.validity_labels)
    rows = []
    for example in examples:
        result = p.infer(
            example.question,
            use_gold_anchors=example.gold_anchors if args.toy_oracle else None,
            use_gold_schemas=example.gold_schemas if args.toy_oracle else None,
        )
        row = build_row(example, result, p.kg, labels)
        generator = None
        if p.config.get("output", {}).get("open_answer") == "generate":
            if not hasattr(p.text_encoder, "generate_answer"):
                raise ValueError("Generation output requires the Hugging Face causal-LM backend.")
            generator = p.text_encoder.generate_answer
        row.update(benchmark_outputs(example, result, p.kg, generator))
        row.update(
            checkpoint_sha256=ckpt,
            seed=p.config["seed"],
            split=args.split,
            schema_mode="toy_oracle" if args.toy_oracle else "question_only",
            search_stats=p.searcher.last_stats,
        )
        rows.append(row)
    summary = summarize_rows(rows)
    for name in (
        "MCAccuracy",
        "MCMRR",
        "OpenEM",
        "TokenF1",
        "RelationF1",
        "SchemaHit@1",
        "SchemaHit@M",
        "SchemaRecall@M",
        "Schema.hops",
        "Schema.relations",
        "Schema.directions",
        "Schema.entity_types",
    ):
        values = [r[name] for r in rows if name in r]
        if values:
            summary[name] = mean(values)
    calibrated = [
        (r["SchemaTopProbability"], r["SchemaHit@1"]) for r in rows if "SchemaTopProbability" in r
    ]
    summary["schema_calibration"] = calibration_metrics(
        [v[0] for v in calibrated], [v[1] for v in calibrated]
    )
    summary["metadata"] = {
        "checkpoint_sha256": ckpt,
        "seed": p.config["seed"],
        "split": args.split,
        "config_sha256": sha256(args.config),
        "questions_sha256": sha256(data_path),
        "entities_sha256": sha256(p._resolve(p.config["data"]["entities"])),
        "edges_sha256": sha256(p._resolve(p.config["data"]["edges"])),
        "labels_sha256": sha256(args.validity_labels) if args.validity_labels else None,
        "kind": "toy" if toy else "benchmark",
        "oracle": args.toy_oracle,
        "effective_config": p.config,
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    )
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
