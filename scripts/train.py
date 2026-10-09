#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from resemreason.data import load_questions
from resemreason.evaluation import build_row, select_threshold, sha256, write_json
from resemreason.pipeline import ReSemReasonPipeline
from resemreason.trainer import ReSemReasonTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variant")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", default="outputs/resemreason.pt")
    parser.add_argument("--dev-questions")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive.")
    p = ReSemReasonPipeline.from_yaml(args.config, seed=args.seed, variant=args.variant)
    if args.epochs is not None:
        p.config["training"]["epochs"] = args.epochs
    if p.config["data"].get("kind") != "toy" and not p.config["data"].get("train"):
        raise ValueError("Benchmark training requires an explicit data.train file.")
    train_path = p._resolve(p.config["data"].get("train", p.config["data"]["questions"]))
    examples = load_questions(train_path)
    if not examples:
        raise ValueError("Training set is empty.")
    trainer = ReSemReasonTrainer(p)
    history = []
    for epoch in range(int(p.config["training"]["epochs"])):
        stats = trainer.train_epoch(examples)
        history.append(asdict(stats))
        print(
            f"epoch={epoch + 1} loss={stats.loss:.6f} examples={stats.examples} skipped={stats.skipped}"
        )
    dev_path = args.dev_questions or p.config["data"].get("dev")
    calibration = None
    if dev_path:
        dev_path = p._resolve(dev_path)
        if dev_path.resolve() == train_path.resolve():
            raise ValueError("Development file must be separate from training.")
        dev = load_questions(dev_path)
        if {e.question_id for e in dev} & {e.question_id for e in examples}:
            raise ValueError("Train/dev question leakage.")
        train_sources = {e.source_id for e in examples if e.source_id is not None}
        if train_sources & {e.source_id for e in dev if e.source_id is not None}:
            raise ValueError("Train/dev source leakage.")
        if any(e.split not in (None, "dev") for e in dev):
            raise ValueError("Threshold fitting accepts dev only.")
        rows = [build_row(e, p.infer(e.question), p.kg, {}) for e in dev]
        threshold, f1 = select_threshold(rows)
        p.config["model"]["answer_threshold"] = threshold
        p.model.answer_threshold = threshold
        calibration = {"threshold": threshold, "dev_macro_f1": f1, "dev_sha256": sha256(dev_path)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out)
    write_json(
        out.with_suffix(".run.json"),
        {
            "config": p.config,
            "history": history,
            "train_sha256": sha256(train_path),
            "entities_sha256": sha256(p._resolve(p.config["data"]["entities"])),
            "edges_sha256": sha256(p._resolve(p.config["data"]["edges"])),
            "calibration": calibration,
            "checkpoint_sha256": sha256(out),
            "kind": p.config["data"].get("kind", "benchmark"),
        },
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
