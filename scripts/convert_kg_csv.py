#!/usr/bin/env python
"""Convert generic entity/edge CSV files to the ReSemReason JSONL schema."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def convert_entities(args: argparse.Namespace) -> None:
    output = Path(args.output_entities)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.entities_csv).open("r", encoding="utf-8", newline="") as src, output.open(
        "w", encoding="utf-8"
    ) as dst:
        for row in csv.DictReader(src):
            aliases = [item.strip() for item in row.get(args.aliases_col, "").split(args.alias_sep) if item.strip()]
            dst.write(
                json.dumps(
                    {
                        "id": row[args.entity_id_col],
                        "name": row[args.entity_name_col],
                        "type": row[args.entity_type_col],
                        "aliases": aliases,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def convert_edges(args: argparse.Namespace) -> None:
    output = Path(args.output_edges)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.edges_csv).open("r", encoding="utf-8", newline="") as src, output.open(
        "w", encoding="utf-8"
    ) as dst:
        for row in csv.DictReader(src):
            payload = {
                "source": row[args.source_col],
                "relation": row[args.relation_col],
                "target": row[args.target_col],
            }
            if args.provenance_col and row.get(args.provenance_col):
                payload["provenance"] = row[args.provenance_col]
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities-csv", required=True)
    parser.add_argument("--edges-csv", required=True)
    parser.add_argument("--output-entities", required=True)
    parser.add_argument("--output-edges", required=True)
    parser.add_argument("--entity-id-col", default="id")
    parser.add_argument("--entity-name-col", default="name")
    parser.add_argument("--entity-type-col", default="type")
    parser.add_argument("--aliases-col", default="aliases")
    parser.add_argument("--alias-sep", default="|")
    parser.add_argument("--source-col", default="source")
    parser.add_argument("--relation-col", default="relation")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--provenance-col", default="provenance")
    args = parser.parse_args()
    convert_entities(args)
    convert_edges(args)


if __name__ == "__main__":
    main()
