"""Explicit benchmark outputs; no gold schema is sent to the inference interface."""

from __future__ import annotations

import re

from .metrics import open_answer_scores, reciprocal_rank, relation_f1


def benchmark_outputs(example, result, kg, generator=None):
    output = {}
    if example.dataset == "primekgqa":
        selected = set(result.answer_ids)
        relations = []
        # Decode one relation chain from the highest-ranked selected explanation.
        # Do not multiply the query's relation multiset by the number of answers.
        for prediction in result.predictions:
            if prediction.entity_id in selected and prediction.supporting_paths:
                path = prediction.supporting_paths[0]
                relations = list(zip(path.relations, path.directions))
                break
        output["predicted_relations"] = [list(item) for item in relations]
        output["RelationF1"] = relation_f1(relations, example.gold_relations)
    if example.options:
        # Each option is mapped to canonical entity IDs during preprocessing.
        scores = {p.entity_id: p.score for p in result.predictions}
        ranked = sorted(
            (
                (max(scores[e] for e in entities if e in scores), option)
                for option, entities in example.options.items()
                if any(e in scores for e in entities)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        options = [option for _, option in ranked]
        output.update(
            predicted_option=options[0] if options else None,
            MCAccuracy=float(bool(options) and options[0] == example.gold_option),
            MCMRR=reciprocal_rank(options, {example.gold_option}),
        )
    if example.gold_texts:
        # Canonical entity-name realization, explicitly distinct from a free-form LM generator.
        if generator is None:
            answer = "; ".join(kg.entity(e).name for e in result.answer_ids)
        else:
            evidence = []
            for prediction in result.predictions:
                if prediction.entity_id in result.answer_ids and prediction.supporting_paths:
                    path = prediction.supporting_paths[0]
                    evidence.append(kg.describe_path(path.nodes, path.relations, path.directions))
            answer = generator(example.question, "\n".join(evidence))
        output.update(answer_text=answer, **open_answer_scores(answer, example.gold_texts))
    return output


def parse_select_chain(query: str):
    """One projected variable, 1-3 triple chain, URI/prefixed terms; fail on other SPARQL."""
    prefixes = dict(re.findall(r"PREFIX\s+(\w+):\s*<([^>]+)>", query, flags=re.I))
    body = re.sub(r"PREFIX\s+\w+:\s*<[^>]+>\s*", "", query, flags=re.I).strip()
    match = re.fullmatch(
        r"SELECT\s+(?:DISTINCT\s+)?(\?\w+)\s+WHERE\s*\{(.*)\}\s*", body, flags=re.I | re.S
    )
    if not match:
        raise ValueError("Supported: one-variable SELECT [DISTINCT] WHERE { triple chain }.")
    target, content = match.groups()
    token = r"(?:<[^>]+>|\?\w+|[A-Za-z_][\w-]*:[\w./#-]+)"
    triple = re.compile(rf"\s*({token})\s+({token})\s+({token})\s*(?:\.|$)")
    rows, pos = [], 0

    def expand(value):
        if value.startswith("<"):
            return value[1:-1]
        if ":" in value:
            prefix, suffix = value.split(":", 1)
            return prefixes[prefix] + suffix if prefix in prefixes else value
        return value

    while content[pos:].strip():
        m = triple.match(content, pos)
        if not m:
            raise ValueError(
                "Unsupported query: branching/aggregation/OPTIONAL/UNION/FILTER or syntax."
            )
        s, r, t = map(expand, m.groups())
        if r.startswith("?"):
            raise ValueError("Variable relation predicates are unsupported.")
        rows.append((s, r, t))
        pos = m.end()
    if not 1 <= len(rows) <= 3:
        raise ValueError("Queries must have 1-3 hops.")
    constants = {x for s, r, t in rows for x in (s, t) if not x.startswith("?")}
    if len(constants) != 1:
        raise ValueError("A chain must have exactly one constant anchor.")
    anchor = next(iter(constants))
    node = anchor
    chain = []
    remaining = list(rows)
    seen = {node}
    while remaining:
        choices = [
            (i, r, t, 1) if s == node else (i, r, s, -1)
            for i, (s, r, t) in enumerate(remaining)
            if s == node or t == node
        ]
        if len(choices) != 1:
            raise ValueError("Query is not a simple chain.")
        i, relation, end, direction = choices[0]
        if end in seen:
            raise ValueError("Cyclic query is unsupported.")
        chain.append((relation, direction))
        seen.add(end)
        node = end
        remaining.pop(i)
    if node != target:
        raise ValueError("Projected variable must be the chain endpoint.")
    return anchor, chain


def execute_select_chain(kg, anchor, chain, entity_map=None, relation_map=None):
    entity_map, relation_map = entity_map or {}, relation_map or {}
    anchor = entity_map.get(anchor, anchor)
    if anchor not in kg.entities:
        raise ValueError("Query anchor absent from frozen KG/mapping.")
    endpoints = {anchor}
    for relation, direction in chain:
        relation = relation_map.get(relation, relation)
        if relation not in kg.relations:
            raise ValueError("Query relation absent from frozen KG/mapping.")
        endpoints = {
            edge.target
            for node in endpoints
            for edge in kg.neighbors(node)
            if edge.relation == relation and edge.direction == direction
        }
    return sorted(endpoints)
