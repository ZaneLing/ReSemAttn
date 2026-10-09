#!/usr/bin/env python
"""Score independently labeled, same-endpoint path pairs without gold inference schemas."""

import argparse
from collections import defaultdict
from statistics import mean

import torch

from resemreason.data import load_questions
from resemreason.evaluation import checkpoint_for_evaluation, write_json
from resemreason.metrics import average_precision, canonical_path_key, pair_accuracy
from resemreason.pipeline import ReSemReasonPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-untrained-toy", action="store_true")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", default="outputs/pair_scores.json")
    parser.add_argument("--variant")
    args = parser.parse_args()
    p = ReSemReasonPipeline.from_yaml(args.config, variant=args.variant)
    ckpt = checkpoint_for_evaluation(p, args.checkpoint, args.allow_untrained_toy)
    rows = []
    all_paths = {}
    p.model.eval()
    with torch.no_grad():
        for e in load_questions(args.questions):
            schemas = p.schema_predictor.predict(
                e.question, p.config["schema"]["top_m"], p.config["schema"]["beam_size"], p.device
            )
            schemas = p.adjust_schemas(schemas)
            for i, (valid, invalid) in enumerate(e.path_pairs):
                p.kg.validate_path(valid)
                p.kg.validate_path(invalid)
                output = p.model.forward_candidate(
                    e.question, valid.end, [valid, invalid], schemas, p.device
                )
                scores = output.path_scores.tolist()
                rows.append(
                    dict(
                        question_id=e.question_id,
                        pair=i,
                        valid_score=scores[0],
                        invalid_score=scores[1],
                        win=int(scores[0] > scores[1]),
                        negative_type=e.path_pair_types[i],
                    )
                )
                for path, score, label in zip([valid, invalid], scores, [1, 0]):
                    key = canonical_path_key(
                        e.question_id, path.nodes, path.relations, path.directions
                    )
                    if key in all_paths and all_paths[key][1] != label:
                        raise ValueError("Conflicting validity labels.")
                    all_paths[key] = (score, label)
    if not rows:
        raise ValueError("No independently labeled path pairs supplied.")
    groups = defaultdict(list)
    for row in rows:
        groups[row["negative_type"]].append(row["win"])
    by_type = {k: dict(PairAcc=mean(v), pairs=len(v)) for k, v in groups.items()}
    write_json(
        args.output,
        dict(
            checkpoint_sha256=ckpt,
            rows=rows,
            by_negative_type=by_type,
            MacroPairAcc=mean(v["PairAcc"] for v in by_type.values())
            if "unspecified" not in groups
            else None,
            PairAcc=pair_accuracy((r["valid_score"], r["invalid_score"]) for r in rows),
            AUPRC=average_precision(
                [v[0] for v in all_paths.values()], [v[1] for v in all_paths.values()]
            ),
        ),
    )


if __name__ == "__main__":
    main()
