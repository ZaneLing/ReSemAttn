#!/usr/bin/env python
"""Train and evaluate the five prespecified seeds; never synthesize result rows."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--validity-labels")
    parser.add_argument("--output", default="outputs/five_seeds")
    args = parser.parse_args()
    scripts = Path(__file__).parent
    for seed in [13, 21, 42, 87, 100]:
        out = Path(args.output) / str(seed)
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = out / "model.pt"
        subprocess.run(
            [
                sys.executable,
                str(scripts / "train.py"),
                "--config",
                args.config,
                "--seed",
                str(seed),
                "--output",
                str(checkpoint),
            ],
            check=True,
        )
        command = [
            sys.executable,
            str(scripts / "evaluate.py"),
            "--config",
            args.config,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "evaluation"),
        ]
        if args.validity_labels:
            command += ["--validity-labels", args.validity_labels]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
