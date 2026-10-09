#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from resemreason.adapters import execute_select_chain, parse_select_chain
from resemreason.evaluation import sha256, write_json
from resemreason.graph import BiomedicalKG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--entities", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument(
        "--mapping", help="JSON with entity_map and relation_map URI-to-canonical dictionaries"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    kg = BiomedicalKG.from_jsonl(args.entities, args.edges)
    mapping = json.loads(Path(args.mapping).read_text()) if args.mapping else {}
    rows = []
    unsupported = []
    for line in Path(args.input).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            anchor, chain = parse_select_chain(row["sparql"])
            answers = execute_select_chain(kg, anchor, chain, **mapping)
            row.update(
                dataset="primekgqa",
                gold_answers=answers,
                gold_anchors=[mapping.get("entity_map", {}).get(anchor, anchor)],
                gold_relations=[[mapping.get("relation_map", {}).get(r, r), d] for r, d in chain],
            )
            rows.append(row)
        except ValueError as error:
            unsupported.append(dict(id=row["id"], reason=str(error)))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_json(
        out.with_suffix(".manifest.json"),
        dict(
            input_sha256=sha256(args.input),
            entities_sha256=sha256(args.entities),
            edges_sha256=sha256(args.edges),
            supported=len(rows),
            unsupported=unsupported,
        ),
    )


if __name__ == "__main__":
    main()
