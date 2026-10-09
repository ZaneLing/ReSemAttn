#!/usr/bin/env python
"""Refit a checkpoint's set threshold only from explicitly marked dev rows."""

import argparse
import json
from pathlib import Path

from resemreason.evaluation import select_threshold, sha256, write_json
from resemreason.pipeline import ReSemReasonPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--variant")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dev-rows", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in Path(args.dev_rows).read_text().splitlines() if s.strip()]
    digest = sha256(args.checkpoint)
    if any(r.get("split") != "dev" or r.get("checkpoint_sha256") != digest for r in rows):
        raise ValueError("Calibration requires dev rows from this exact checkpoint.")
    threshold, f1 = select_threshold(rows)
    p = ReSemReasonPipeline.from_yaml(args.config, variant=args.variant)
    p.load(args.checkpoint)
    p.config["model"]["answer_threshold"] = threshold
    p.model.answer_threshold = threshold
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out)
    write_json(
        out.with_suffix(".calibration.json"),
        dict(
            threshold=threshold,
            dev_macro_f1=f1,
            source_checkpoint_sha256=digest,
            dev_rows_sha256=sha256(args.dev_rows),
        ),
    )


if __name__ == "__main__":
    main()
