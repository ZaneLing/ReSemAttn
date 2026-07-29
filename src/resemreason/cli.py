from __future__ import annotations

import argparse

from .pipeline import ReSemReasonPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReSemReason inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    pipeline = ReSemReasonPipeline.from_yaml(args.config)
    result = pipeline.infer(args.question)
    print("Answers:")
    for answer_id in result.answer_ids:
        print(f"- {answer_id}: {pipeline.kg.entity(answer_id).name}")
