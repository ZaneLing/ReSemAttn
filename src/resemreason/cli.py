from __future__ import annotations

import argparse

from .pipeline import ReSemReasonPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReSemReason inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--variant")
    parser.add_argument("--allow-untrained-toy", action="store_true")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    pipeline = ReSemReasonPipeline.from_yaml(args.config, variant=args.variant)
    from .evaluation import checkpoint_for_evaluation

    status = checkpoint_for_evaluation(pipeline, args.checkpoint, args.allow_untrained_toy)
    if status == "untrained_toy":
        print("UNTRAINED TOY DEMONSTRATION: not benchmark measurements.")
    result = pipeline.infer(args.question)
    print("Answers:")
    for answer_id in result.answer_ids:
        print(f"- {answer_id}: {pipeline.kg.entity(answer_id).name}")
