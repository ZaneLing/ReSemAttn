#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from resemreason.pipeline import ReSemReasonPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--toy-oracle",
        action="store_true",
        help="Use toy gold anchors and schemas for a deterministic pipeline demonstration.",
    )
    args = parser.parse_args()

    pipeline = ReSemReasonPipeline.from_yaml(args.config)
    gold_anchors = None
    gold_schemas = None
    if args.toy_oracle:
        from resemreason.data import load_questions

        examples = load_questions(pipeline._resolve(pipeline.config["data"]["questions"]))
        match = next((item for item in examples if item.question == args.question), None)
        if match is not None:
            gold_anchors = match.gold_anchors
            gold_schemas = match.gold_schemas

    result = pipeline.infer(
        args.question,
        use_gold_anchors=gold_anchors,
        use_gold_schemas=gold_schemas,
    )

    print("\nGrounded anchors")
    for item in result.groundings:
        entity = pipeline.kg.entity(item.entity_id)
        print(f"  {item.probability:7.4f}  {entity.name} ({entity.entity_type})")

    print("\nSoft schemas")
    for schema in result.schemas:
        print(
            f"  p={schema.probability:.4f} hops={schema.hops} "
            f"types={schema.entity_types} relations={schema.relations} "
            f"directions={schema.directions}"
        )

    print("\nRanked candidates")
    for prediction in result.predictions[: args.top_k]:
        entity = pipeline.kg.entity(prediction.entity_id)
        print(
            f"  score={prediction.score:8.4f} p={prediction.probability:7.4f} "
            f"{entity.name} ({entity.entity_type})"
        )
        for path in prediction.supporting_paths[:2]:
            print("    " + pipeline.kg.describe_path(path.nodes, path.relations, path.directions))

    print("\nPredicted answer set")
    print(json.dumps(result.answer_ids, indent=2))


if __name__ == "__main__":
    main()
