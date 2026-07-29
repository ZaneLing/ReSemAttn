#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from resemreason.data import load_questions
from resemreason.pipeline import ReSemReasonPipeline
from resemreason.trainer import ReSemReasonTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", default="outputs/resemreason.pt")
    args = parser.parse_args()

    pipeline = ReSemReasonPipeline.from_yaml(args.config)
    trainer = ReSemReasonTrainer(pipeline)
    examples = load_questions(pipeline._resolve(pipeline.config["data"]["questions"]))
    epochs = args.epochs or int(pipeline.config["training"].get("epochs", 5))
    for epoch in range(1, epochs + 1):
        stats = trainer.train_epoch(examples)
        print(f"epoch={epoch} loss={stats.loss:.6f} examples={stats.examples}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(output)
    print(f"saved checkpoint to {output}")


if __name__ == "__main__":
    main()
