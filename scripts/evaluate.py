#!/usr/bin/env python
from __future__ import annotations

import argparse
from statistics import mean

from resemreason.data import load_questions
from resemreason.metrics import coverage, hits_at_k, joint_at_1, reciprocal_rank
from resemreason.pipeline import ReSemReasonPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--toy-oracle",
        action="store_true",
        help="Use toy gold anchors and schemas to test search/reasoning independently.",
    )
    args = parser.parse_args()

    pipeline = ReSemReasonPipeline.from_yaml(args.config)
    examples = load_questions(pipeline._resolve(pipeline.config["data"]["questions"]))
    rows = []
    for example in examples:
        result = pipeline.infer(
            example.question,
            use_gold_anchors=example.gold_anchors if args.toy_oracle else None,
            use_gold_schemas=example.gold_schemas if args.toy_oracle else None,
        )
        ranked = [item.entity_id for item in result.predictions]
        gold = set(example.gold_answers)
        top_prediction = result.predictions[0] if result.predictions else None
        top_path = top_prediction.supporting_paths[0] if top_prediction and top_prediction.supporting_paths else None
        rows.append(
            {
                "H@1": hits_at_k(ranked, gold, 1),
                "H@5": hits_at_k(ranked, gold, 5),
                "MRR": reciprocal_rank(ranked, gold),
                "Coverage": coverage(ranked, gold),
                "Joint@1": joint_at_1(
                    top_prediction.entity_id if top_prediction else None,
                    top_path,
                    gold,
                    pipeline.kg,
                    example.gold_schemas,
                ),
            }
        )
    for metric in rows[0]:
        print(f"{metric:10s}: {mean(row[metric] for row in rows):.4f}")


if __name__ == "__main__":
    main()
