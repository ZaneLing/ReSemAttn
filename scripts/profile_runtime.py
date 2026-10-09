#!/usr/bin/env python
"""Stage timings; loading excluded, synchronized CUDA boundaries, real observations only."""

import argparse
import math
import statistics
import time

import torch

from resemreason.data import load_questions
from resemreason.evaluation import checkpoint_for_evaluation, write_json
from resemreason.pipeline import ReSemReasonPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-untrained-toy", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--output", default="outputs/profile.json")
    parser.add_argument("--variant")
    args = parser.parse_args()
    if args.queries < 1 or args.warmup < 0:
        parser.error("Invalid timing counts.")
    p = ReSemReasonPipeline.from_yaml(args.config, variant=args.variant)
    ckpt = checkpoint_for_evaluation(p, args.checkpoint, args.allow_untrained_toy)
    examples = load_questions(p._resolve(p.config["data"]["questions"]))
    if not examples:
        raise ValueError("No profiling questions.")
    cuda = p.device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize(p.device)

    for i in range(args.warmup):
        p.infer(examples[i % len(examples)].question)
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats(p.device)
    rows = []
    for i in range(args.queries):
        q = examples[i % len(examples)].question
        sync()
        t = time.perf_counter()
        g, s = p.predict_structure(q)
        sync()
        prepared = time.perf_counter()
        pools = p.searcher.search(q, g, s, p.device)
        sync()
        r = time.perf_counter()
        p.infer_pool(q, g, s, pools)
        sync()
        end = time.perf_counter()
        rows.append(
            dict(
                grounding_schema_ms=1000 * (prepared - t),
                search_ms=1000 * (r - prepared),
                retrieval_ms=1000 * (r - t),
                reasoning_ms=1000 * (end - r),
                total_ms=1000 * (end - t),
                candidates=len(pools),
                paths=sum(map(len, pools.values())),
                **p.searcher.last_stats,
            )
        )
    times = sorted(r["total_ms"] for r in rows)
    write_json(
        args.output,
        dict(
            checkpoint_sha256=ckpt,
            device=str(p.device),
            batch_size=1,
            warmup=args.warmup,
            timed_queries=len(rows),
            unique_input_questions=len(examples),
            loading_excluded=True,
            mean_ms=statistics.mean(times),
            median_ms=statistics.median(times),
            p95_ms=times[math.ceil(0.95 * len(times)) - 1],
            graph_checksums=p.graph_checksums(),
            effective_config=p.config,
            peak_memory_gb=torch.cuda.max_memory_allocated(p.device) / 1e9 if cuda else None,
            rows=rows,
        ),
    )


if __name__ == "__main__":
    main()
