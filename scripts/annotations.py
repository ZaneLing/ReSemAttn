#!/usr/bin/env python
"""Blind path-label export and disagreement-preserving label reconciliation (E6)."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from resemreason.evaluation import write_json
from resemreason.graph import BiomedicalKG
from resemreason.metrics import canonical_path_key
from resemreason.statistics import annotation_agreement


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def unique_rows(path, key="item_id"):
    rows = read(path)
    if len({r[key] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate {key} in annotation input.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["export", "reconcile"])
    parser.add_argument("--rows", nargs="+")
    parser.add_argument("--questions")
    parser.add_argument("--batch")
    parser.add_argument("--annotator-a")
    parser.add_argument("--annotator-b")
    parser.add_argument("--adjudications")
    parser.add_argument("--entities")
    parser.add_argument("--edges")
    parser.add_argument("--relation-definitions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "export":
        if not all([args.rows, args.questions, args.entities, args.edges]):
            parser.error("export requires --rows, --questions, --entities and --edges")
        kg = BiomedicalKG.from_jsonl(args.entities, args.edges)
        definitions = (
            json.loads(Path(args.relation_definitions).read_text())
            if args.relation_definitions
            else {}
        )
        questions = {str(r["id"]): r["question"] for r in unique_rows(args.questions, "id")}
        items = {}
        for path in args.rows:
            for row in read(path):
                for prediction in row["predictions"]:
                    for p in prediction["paths"]:
                        key = canonical_path_key(
                            row["question_id"], p["nodes"], p["relations"], p["directions"]
                        )
                        uid = hashlib.sha256(key.encode()).hexdigest()
                        items[uid] = dict(
                            item_id=uid,
                            question_id=row["question_id"],
                            question=questions[row["question_id"]],
                            nodes=p["nodes"],
                            relations=p["relations"],
                            directions=p["directions"],
                            candidate=kg.entity(p["nodes"][-1]).text,
                            node_descriptions=[kg.entity(n).text for n in p["nodes"]],
                            relation_definitions={r: definitions.get(r) for r in p["relations"]},
                            provenance=[
                                [
                                    edge.provenance
                                    for edge in kg.neighbors(p["nodes"][i])
                                    if edge.target == p["nodes"][i + 1]
                                    and edge.relation == relation
                                    and edge.direction == p["directions"][i]
                                ]
                                for i, relation in enumerate(p["relations"])
                            ],
                            valid=None,
                        )
        rows = list(items.values())
        random.Random(12345).shuffle(rows)
        (out / "blind.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return
    if not all([args.batch, args.annotator_a, args.annotator_b]):
        parser.error("reconcile requires batch and both annotators")
    batch = unique_rows(args.batch)
    a = {r["item_id"]: r["valid"] for r in unique_rows(args.annotator_a)}
    b = {r["item_id"]: r["valid"] for r in unique_rows(args.annotator_b)}
    if set(a) != set(b) or set(a) != {r["item_id"] for r in batch}:
        raise ValueError("Annotation item IDs must match exactly.")
    arb = (
        {r["item_id"]: r["valid"] for r in unique_rows(args.adjudications)}
        if args.adjudications
        else {}
    )
    if not set(arb).issubset(a):
        raise ValueError("Adjudication IDs must belong to this annotation batch.")
    rows = []
    la = []
    lb = []
    for row in batch:
        uid = row["item_id"]
        va, vb = a[uid], b[uid]
        if any(v not in (0, 1, None) for v in (va, vb, arb.get(uid))):
            raise ValueError("Labels must be 0/1/null.")
        if va is not None and vb is not None:
            la.append(va)
            lb.append(vb)
        valid = arb.get(uid, va if va == vb else None)
        rows.append(
            {
                **row,
                "valid": valid,
                "label_source": "adjudicated"
                if uid in arb
                else "agreement"
                if valid is not None
                else "unresolved",
            }
        )
    (out / "final_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(
        out / "agreement.json",
        dict(
            items=len(rows),
            paired_labels=len(la),
            **(annotation_agreement(la, lb) if la else {"cohen_kappa": None}),
            unresolved=sum(r["valid"] is None for r in rows),
        ),
    )


if __name__ == "__main__":
    main()
