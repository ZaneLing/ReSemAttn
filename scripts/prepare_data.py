#!/usr/bin/env python
"""Deterministic source/template/composition grouping with auditable split manifests."""

import argparse
import json
import random
from pathlib import Path

from resemreason.evaluation import sha256, write_json


def grouped_split(rows, mode="source", seed=42):
    # Connected components keep all shared sources / compositions in one split.
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, row in enumerate(rows):
        if mode == "source":
            if not row.get("source_id"):
                raise ValueError("Source split requires source_id for every row.")
            keys = [row["source_id"]]
        elif mode == "template":
            if not row.get("template_id"):
                raise ValueError("Template split requires template_id.")
            keys = [row["template_id"]]
        else:
            keys = [
                json.dumps([s["relations"], s["directions"]]) for s in row.get("gold_schemas", [])
            ]
            if not keys:
                raise ValueError("Composition split requires relation-chain labels.")
        # Exact duplicate questions always stay together, even under different provenance IDs.
        keys = [mode + ":" + k for k in keys] + [
            "question:" + " ".join(row["question"].lower().split())
        ]
        if row.get("source_id") is not None:
            keys.append("source:" + str(row["source_id"]))
        for key in keys:
            if key in owner:
                parent[find(i)] = find(owner[key])
            else:
                owner[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    buckets = list(groups.values())
    random.Random(seed).shuffle(buckets)
    result = {"train": [], "dev": [], "test": []}
    # Ratios apply to indivisible groups; exact question counts are not guaranteed.
    n = len(buckets)
    a = int(0.8 * n)
    c = int(0.9 * n)
    for i, group in enumerate(buckets):
        split = "train" if i < a else "dev" if i < c else "test"
        result[split].extend({**row, "split": split} for row in group)
    return result


def primitive_audit(splits):
    def primitives(rows):
        return {
            json.dumps([relation, direction])
            for row in rows
            for schema in row.get("gold_schemas", [])
            for relation, direction in zip(schema["relations"], schema["directions"])
        }

    train = primitives(splits["train"])
    unseen = {k: sorted(primitives(splits[k]) - train) for k in ("dev", "test")}
    return dict(
        train_directed_relations=sorted(train),
        unseen_primitives=unseen,
        heldout_relations_known=bool(train) and not any(unseen.values()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["source", "template", "composition"], default="source")
    parser.add_argument("--require-known-primitives", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate question IDs.")
    splits = grouped_split(rows, args.mode, args.seed)
    audit = primitive_audit(splits)
    if args.require_known_primitives and not audit["heldout_relations_known"]:
        raise ValueError(
            "Held-out atomic relations are not all present in training; revise the split."
        )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items)
        )
    write_json(
        out / "manifest.json",
        dict(
            source_sha256=sha256(args.input),
            mode=args.mode,
            seed=args.seed,
            proportions=[0.8, 0.1, 0.1],
            counts={k: len(v) for k, v in splits.items()},
            split_sha256={k: sha256(out / f"{k}.jsonl") for k in splits},
            question_ids={k: [r["id"] for r in v] for k, v in splits.items()},
            primitive_audit=audit,
        ),
    )


if __name__ == "__main__":
    main()
