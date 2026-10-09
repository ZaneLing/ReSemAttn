#!/usr/bin/env python
"""Aggregate real exported rows, annotation strata, or aligned five-seed comparisons."""

import argparse
import json
from pathlib import Path

from resemreason.evaluation import write_json
from resemreason.metrics import summarize_rows
from resemreason.statistics import holm_adjust, paired_bootstrap


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows")
    parser.add_argument("--strata", help="JSON mapping question IDs to one predeclared stratum")
    parser.add_argument(
        "--comparisons", help="JSON list: {name, metric, a:{seed:rows}, b:{seed:rows}}"
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = {}
    if args.rows:
        rows = read_rows(args.rows)
        output["overall"] = summarize_rows(rows)
        if args.strata:
            labels = json.loads(Path(args.strata).read_text())
            if any(r["question_id"] not in labels for r in rows):
                raise ValueError("Missing question strata.")
            output["strata"] = {
                stratum: summarize_rows([r for r in rows if labels[r["question_id"]] == stratum])
                for stratum in sorted({labels[r["question_id"]] for r in rows})
            }
    if args.comparisons:
        comparisons = json.loads(Path(args.comparisons).read_text())
        tests = []
        for c in comparisons:
            result = paired_bootstrap(
                {int(k): read_rows(v) for k, v in c["a"].items()},
                {int(k): read_rows(v) for k, v in c["b"].items()},
                c["metric"],
                samples=args.samples,
            )
            tests.append(dict(name=c["name"], metric=c["metric"], **result))
        for test, p in zip(tests, holm_adjust([t["p_value"] for t in tests])):
            test["p_holm"] = p
        output["paired_tests"] = tests
    if not output:
        parser.error("Provide --rows or --comparisons.")
    write_json(args.output, output)


if __name__ == "__main__":
    main()
