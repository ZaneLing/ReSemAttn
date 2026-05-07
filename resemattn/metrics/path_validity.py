from __future__ import annotations
from typing import Any, Dict, List
from resemattn.data.schema import infer_schema, normalize_relation_name, inverse_relation
from resemattn.metrics.answer_metrics import rank_candidates


def _type_match(expected: str, observed: str) -> bool:
    return str(expected).lower() == str(observed).lower()


def _rel_match(expected_family: List[str], observed: str) -> bool:
    obs = normalize_relation_name(observed)
    fam = {normalize_relation_name(r) for r in expected_family}
    fam |= {inverse_relation(r) for r in fam}
    return obs in fam


def valid_path(question_row: Dict[str, Any], candidate: Dict[str, Any], path: Dict[str, Any]) -> bool:
    schema = infer_schema(question_row)
    nodes = path.get("nodes", [])
    rels = path.get("relations", [])
    L = schema.num_hops
    if len(nodes) < L + 1 or len(rels) < L:
        return False
    # Entity type sequence.
    for l in range(L + 1):
        if not _type_match(schema.entity_types[l], nodes[l].get("type", "unknown")):
            return False
    # Relation family sequence.
    for l in range(L):
        r = rels[l]
        name = r.get("name", r) if isinstance(r, dict) else str(r)
        if not _rel_match(schema.relation_families[l], name):
            return False
    # Endpoint match and answer type.
    endpoint = nodes[L]
    endpoint_id = str(endpoint.get("id", endpoint.get("name", "")))
    cand_id = str(candidate.get("id", candidate.get("name", "")))
    cand_name = str(candidate.get("name", candidate.get("id", "")))
    return endpoint_id == cand_id or str(endpoint.get("name", "")) == cand_name


def candidate_valid(question_row: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    return any(valid_path(question_row, candidate, p) for p in candidate.get("paths", []))


def path_validity_metrics(predictions: List[Dict[str, Any]], ks=(1,5,10)) -> Dict[str, float]:
    totals = {f"validpath@{k}": 0.0 for k in ks}
    totals.update({f"invalidpath@{k}": 0.0 for k in ks})
    n = len(predictions)
    for row in predictions:
        ranked = rank_candidates(row["candidates"])
        for k in ks:
            top = ranked[:k]
            if not top:
                continue
            valid_count = sum(1 for c in top if candidate_valid(row.get("raw", row), c))
            vp = valid_count / float(k)
            totals[f"validpath@{k}"] += vp
            totals[f"invalidpath@{k}"] += 1.0 - vp
    return {key: val / max(n,1) for key, val in totals.items()}


def attention_validity(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute AttnValid if per-path attention weights are saved.

    Expected candidate field:
      "path_attention": [float for each path]
    """
    num = den = 0.0
    for row in predictions:
        ranked = rank_candidates(row["candidates"])
        if not ranked:
            continue
        c = ranked[0]
        attn = c.get("path_attention", [])
        for w, p in zip(attn, c.get("paths", [])):
            den += float(w)
            num += float(w) * float(valid_path(row.get("raw", row), c, p))
    return {"attnvalid": num / den if den > 0 else 0.0}
