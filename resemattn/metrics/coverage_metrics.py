from __future__ import annotations
from typing import Any, Dict, List
from resemattn.metrics.answer_metrics import rank_candidates


def coverage_diagnostics(predictions: List[Dict[str, Any]], ks=(1,5,10)) -> Dict[str, float]:
    n = len(predictions)
    covered = 0
    cond_hits = {k: 0 for k in ks}
    overall_hits = {k: 0 for k in ks}
    total_candidates = 0
    total_positive = 0
    for row in predictions:
        candidates = row["candidates"]
        ranked = rank_candidates(candidates)
        gold = {c.get("id", c.get("name")) for c in candidates if c.get("label", 0) == 1}
        total_candidates += len(candidates)
        total_positive += len(gold)
        has_gold = len(gold) > 0
        covered += int(has_gold)
        for k in ks:
            top_ids = {c.get("id", c.get("name")) for c in ranked[:k]}
            hit = len(top_ids & gold) > 0
            overall_hits[k] += int(hit)
            if has_gold:
                cond_hits[k] += int(hit)
    coverage = covered / max(n, 1)
    out = {
        "n": n,
        "covered_questions": covered,
        "uncovered_questions": n - covered,
        "coverage": coverage,
        "no_positive_rate": 1.0 - coverage,
        "positive_density": total_positive / max(total_candidates, 1),
    }
    for k in ks:
        hit = overall_hits[k] / max(n, 1)
        cond = cond_hits[k] / max(covered, 1)
        out[f"hit@{k}"] = hit
        out[f"condhit@{k}"] = cond
        out[f"ranking_failure@{k}"] = 1.0 - cond
        out[f"coverage_normalized_hit@{k}"] = hit / max(coverage, 1e-12)
    return out
