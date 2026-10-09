"""Paper metrics. Independent validity and answer quality share one explicit cohort."""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from statistics import mean


def canonical_path_key(question_id, nodes, relations, directions):
    return json.dumps(
        [str(question_id), list(nodes), list(relations), list(directions)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def unique(ids):
    return list(dict.fromkeys(ids))


def hits_at_k(ranked_ids, gold_ids, k):
    return float(bool(set(unique(ranked_ids)[:k]) & set(gold_ids)))


def reciprocal_rank(ranked_ids, gold_ids):
    return next(
        (1 / i for i, entity in enumerate(unique(ranked_ids), 1) if entity in gold_ids), 0.0
    )


def precision_at_k(ranked_ids, gold_ids, k):
    top = unique(ranked_ids)[:k]
    return len(set(top) & set(gold_ids)) / len(top) if top else 0.0


def recall_at_k(ranked_ids, gold_ids, k):
    return (
        len(set(unique(ranked_ids)[:k]) & set(gold_ids)) / len(set(gold_ids)) if gold_ids else 0.0
    )


def coverage(candidate_ids, gold_ids):
    return float(bool(set(candidate_ids) & set(gold_ids)))


def candidate_recall(candidate_ids, gold_ids):
    return len(set(candidate_ids) & set(gold_ids)) / len(set(gold_ids)) if gold_ids else 0.0


def set_scores(predicted, gold):
    predicted, gold = set(predicted), set(gold)
    n = len(predicted & gold)
    precision = n / len(predicted) if predicted else float(not gold)
    recall = n / len(gold) if gold else float(not predicted)
    return {
        "SetEM": float(predicted == gold),
        "SetP": precision,
        "SetR": recall,
        "SetF1": 2 * n / (len(predicted) + len(gold)) if predicted or gold else 1.0,
    }


def schema_path_validity(kg, path, schemas):
    """Diagnostic only. Never use this proxy for independent paper validity metrics."""
    return any(
        path.hops == s.hops
        and path.relations == s.relations
        and path.directions == s.directions
        and tuple(kg.entity(n).entity_type for n in path.nodes) == s.entity_types
        for s in schemas
    )


def joint_at_1(top_answer, top_path, gold_ids, validity):
    if top_path is None or top_answer is None or top_path.end != top_answer:
        return 0.0
    return None if validity is None else float(top_answer in gold_ids and validity)


def answer_row(ranked, answers, gold):
    row = {f"H@{k}": hits_at_k(ranked, gold, k) for k in (1, 5, 10)}
    row.update(
        {
            "P@10": precision_at_k(ranked, gold, 10),
            "R@10": recall_at_k(ranked, gold, 10),
            "MRR": reciprocal_rank(ranked, gold),
            "Coverage": coverage(ranked, gold),
            "CandidateRecall": candidate_recall(ranked, gold),
            **set_scores(answers, gold),
        }
    )
    return row


def summarize_rows(rows):
    if not rows:
        raise ValueError("Cannot evaluate an empty question set.")
    result = {
        key: mean(row[key] for row in rows)
        for key in rows[0]
        if key
        in {
            "H@1",
            "H@5",
            "H@10",
            "P@10",
            "R@10",
            "MRR",
            "Coverage",
            "CandidateRecall",
            "SetEM",
            "SetP",
            "SetR",
            "SetF1",
        }
    }
    available = [r for r in rows if r["Coverage"]]
    result["Cond.H@1"] = mean(r["H@1"] for r in available) if available else None
    measured = [r for r in rows if r.get("top_path_valid") is not None]
    n = len(measured)
    correct = sum(r["H@1"] for r in measured)
    both = sum(r["H@1"] * r["top_path_valid"] for r in measured)
    result.update(
        {
            "questions": len(rows),
            "validity_questions": n,
            "AnnotationCoverage": n / len(rows),
            "Unresolved": len(rows) - n,
            "SelectedPaths": len({r["top_path_key"] for r in measured if "top_path_key" in r}),
            "Cohort.H@1": correct / n if n else None,
            "ValidPath@1": mean(r["top_path_valid"] for r in measured) if n else None,
            "Joint@1": both / n if n else None,
            "WPR": (correct - both) / correct if correct else None,
        }
    )
    invalid_pools = [r for r in rows if r.get("all_invalid_pool") is True]
    singletons = [r for r in invalid_pools if r["pool_paths"] == 1]
    result.update(
        FSR=mean(r["AcceptedAny"] for r in invalid_pools) if invalid_pools else None,
        all_invalid_questions=len(invalid_pools),
        SingletonFSR=mean(r["AcceptedAny"] for r in singletons) if singletons else None,
        invalid_singleton_questions=len(singletons),
    )
    # Same cohort/labels guarantee Joint@1 = Cohort.H@1 * (1 - WPR).
    return result


def pair_accuracy(pairs):
    """Strict within-model wins; ties count as incorrect, no score-scale comparison."""
    pairs = list(pairs)
    return mean(float(positive > negative) for positive, negative in pairs) if pairs else None


def average_precision(scores, labels):
    """Non-interpolated AP with score ties processed as a group."""
    if len(scores) != len(labels) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("AP requires equally sized labels and finite scores.")
    groups = {}
    for score, label in zip(scores, labels):
        if label not in (0, 1):
            raise ValueError("AP requires binary labels.")
        groups.setdefault(float(score), []).append(label)
    total = sum(labels)
    if not total:
        return None
    seen = positives = 0
    ap = 0.0
    for score in sorted(groups, reverse=True):
        bucket = groups[score]
        seen += len(bucket)
        positives += sum(bucket)
        ap += (sum(bucket) / total) * (positives / seen)
    return ap


def calibration_metrics(probabilities, labels, bins=10):
    if not probabilities:
        return {"Brier": None, "ECE": None}
    if len(probabilities) != len(labels) or bins < 1:
        raise ValueError("Invalid calibration inputs.")
    buckets = [[] for _ in range(bins)]
    for p, y in zip(probabilities, labels):
        if not 0 <= p <= 1 or y not in (0, 1):
            raise ValueError("Calibration requires probabilities and binary labels.")
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    ece = sum(
        len(bucket) / len(labels) * abs(mean(p for p, y in bucket) - mean(y for p, y in bucket))
        for bucket in buckets
        if bucket
    )
    return {"Brier": mean((p - y) ** 2 for p, y in zip(probabilities, labels)), "ECE": ece}


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def open_answer_scores(prediction, references):
    if not references:
        raise ValueError("Open-answer evaluation requires reference text.")
    p = normalize_answer(prediction)
    exact, f1 = [], []
    for reference in references:
        g = normalize_answer(reference)
        common = sum((Counter(p.split()) & Counter(g.split())).values())
        exact.append(float(p == g))
        f1.append(2 * common / (len(p.split()) + len(g.split())) if p or g else 1.0)
    return {"OpenEM": max(exact), "TokenF1": max(f1)}


def relation_f1(predicted, gold):
    p, g = Counter(map(tuple, predicted)), Counter(map(tuple, gold))
    denom = sum(p.values()) + sum(g.values())
    return 2 * sum((p & g).values()) / denom if denom else 1.0


def schema_metrics(predicted, gold):
    def signature(s):
        return s.hops, s.entity_types, s.relations, s.directions

    if not gold:
        return {}
    gold_set = {signature(s) for s in gold}
    p = predicted[0] if predicted else None
    result = {
        "SchemaHit@1": float(p is not None and signature(p) in gold_set),
        "SchemaHit@M": float(bool({signature(s) for s in predicted} & gold_set)),
        "SchemaRecall@M": len({signature(s) for s in predicted} & gold_set) / len(gold_set),
    }
    for field in ("hops", "relations", "directions", "entity_types"):
        result["Schema." + field] = float(
            p is not None and any(getattr(p, field) == getattr(g, field) for g in gold)
        )
    if p is not None:
        result["SchemaTopProbability"] = p.probability
    return result
