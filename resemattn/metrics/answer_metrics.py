from __future__ import annotations
from typing import Dict, List, Any
from collections import defaultdict, Counter
import re


def rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)


def multi_answer_ranking_metrics(predictions: List[Dict[str, Any]], ks=(1,5,10)) -> Dict[str, float]:
    totals = {f"hit@{k}": 0.0 for k in ks}
    totals.update({f"precision@{k}": 0.0 for k in ks})
    totals.update({f"recall@{k}": 0.0 for k in ks})
    mrr = 0.0
    n = len(predictions)
    for row in predictions:
        ranked = rank_candidates(row["candidates"])
        gold = {c.get("id", c.get("name")) for c in row["candidates"] if c.get("label", 0) == 1}
        if not gold:
            # no-gold protocol: metrics are zero
            continue
        first_rr = 0.0
        for idx, c in enumerate(ranked, start=1):
            cid = c.get("id", c.get("name"))
            if cid in gold:
                first_rr = 1.0 / idx
                break
        mrr += first_rr
        for k in ks:
            top = ranked[:k]
            correct = sum(1 for c in top if c.get("id", c.get("name")) in gold)
            totals[f"hit@{k}"] += float(correct > 0)
            totals[f"precision@{k}"] += correct / float(k)
            totals[f"recall@{k}"] += correct / float(len(gold))
    out = {key: val / max(n, 1) for key, val in totals.items()}
    out["mrr"] = mrr / max(n, 1)
    out["n"] = n
    return out


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def token_f1(pred: str, gold: str) -> Dict[str, float]:
    pt = _norm(pred).split()
    gt = _norm(gold).split()
    if not pt and not gt:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pt or not gt:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pc, gc = Counter(pt), Counter(gt)
    overlap = sum((pc & gc).values())
    prec = overlap / max(len(pt), 1)
    rec = overlap / max(len(gt), 1)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"precision": prec, "recall": rec, "f1": f1}


def single_answer_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    em = f1 = rec = 0.0
    for r in rows:
        pred = r.get("prediction", "")
        gold = r.get("gold", "")
        em += float(_norm(pred) == _norm(gold))
        tf = token_f1(pred, gold)
        f1 += tf["f1"]
        rec += tf["recall"]
    n = max(len(rows), 1)
    return {"em": em/n, "f1": f1/n, "recall": rec/n, "n": len(rows)}
