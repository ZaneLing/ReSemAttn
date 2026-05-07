"""Stage 1: deterministic schema construction.

This module implements the paper's deterministic schema parser for KG-derived
benchmarks. It maps question templates and metadata to
S_q = (tau_0, R_1^*, tau_1, ..., R_L^*, tau_L).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class SchemaHop:
    src_type: str
    relation_family: List[str]
    dst_type: str


@dataclass
class QuestionSchema:
    entity_types: List[str]
    relation_families: List[List[str]]

    @property
    def num_hops(self) -> int:
        return len(self.relation_families)

    def to_text(self) -> str:
        parts = [self.entity_types[0]]
        for rels, dst in zip(self.relation_families, self.entity_types[1:]):
            parts.append("/".join(rels))
            parts.append(dst)
        return " -> ".join(parts)


def normalize_relation_name(rel: str) -> str:
    rel = str(rel).strip().lower().replace("_", " ").replace("-", " ")
    rel = " ".join(rel.split())
    return rel


def inverse_relation(rel: str) -> str:
    rel = normalize_relation_name(rel)
    if rel.startswith("inverse "):
        return rel[len("inverse "):]
    return f"inverse {rel}"


def relation_family(rel: str, include_inverse: bool = True) -> List[str]:
    rel = normalize_relation_name(rel)
    fam = [rel]
    inv = inverse_relation(rel)
    if include_inverse and inv not in fam:
        fam.append(inv)
    return fam


# Template-to-schema lookup for BioHopR/PrimeKGQA-style data.
# Add more templates as needed; relation-hop metadata in the input overrides this table.
TEMPLATE_SCHEMA_MAP: Dict[str, QuestionSchema] = {
    "disease:drug:effect/phenotype": QuestionSchema(
        entity_types=["disease", "drug", "effect/phenotype"],
        relation_families=[relation_family("indication"), relation_family("side effect")],
    ),
    "drug:disease:gene/protein": QuestionSchema(
        entity_types=["drug", "disease", "gene/protein"],
        relation_families=[relation_family("indication"), relation_family("associated with")],
    ),
    "drug:gene/protein:disease": QuestionSchema(
        entity_types=["drug", "gene/protein", "disease"],
        relation_families=[relation_family("target"), relation_family("associated with")],
    ),
    "disease:gene/protein:drug": QuestionSchema(
        entity_types=["disease", "gene/protein", "drug"],
        relation_families=[relation_family("associated with"), relation_family("target")],
    ),
    "gene/protein:drug:disease": QuestionSchema(
        entity_types=["gene/protein", "drug", "disease"],
        relation_families=[relation_family("target"), relation_family("indication")],
    ),
    "effect/phenotype:drug:disease": QuestionSchema(
        entity_types=["effect/phenotype", "drug", "disease"],
        relation_families=[relation_family("side effect"), relation_family("indication")],
    ),
}


def _from_relation_hops(example: Dict[str, Any]) -> Optional[QuestionSchema]:
    """Parse explicit relation-hop fields when present."""
    hop_types = []
    rels = []
    # Supported compact schema field: {"entity_types": [...], "relations": [[...], ...]}
    if isinstance(example.get("schema"), dict):
        sch = example["schema"]
        if "entity_types" in sch and "relations" in sch:
            return QuestionSchema(
                entity_types=[str(x).lower() for x in sch["entity_types"]],
                relation_families=[[normalize_relation_name(r) for r in rel] for rel in sch["relations"]],
            )

    # Common BioHopR-like explicit metadata.
    fields = example.get("relation_hops") or example.get("relations")
    types = example.get("type_hops") or example.get("entity_types")
    if fields and types and len(types) == len(fields) + 1:
        return QuestionSchema(
            entity_types=[str(x).lower() for x in types],
            relation_families=[relation_family(r) if isinstance(r, str) else [normalize_relation_name(x) for x in r] for r in fields],
        )

    # Field names: hop1_type, relation_hop1, hop2_type, relation_hop2, target_type.
    if "relation_hop1" in example and "relation_hop2" in example:
        anchor_type = example.get("anchor_type") or example.get("anchor", {}).get("type") or example.get("source_type")
        hop1_type = example.get("hop1_type") or example.get("intermediate_type")
        target_type = example.get("target_type") or example.get("answer_type")
        if anchor_type and hop1_type and target_type:
            return QuestionSchema(
                entity_types=[str(anchor_type).lower(), str(hop1_type).lower(), str(target_type).lower()],
                relation_families=[relation_family(example["relation_hop1"]), relation_family(example["relation_hop2"])],
            )
    return None


def infer_schema(example: Dict[str, Any]) -> QuestionSchema:
    """Deterministically infer S_q from dataset metadata.

    Priority:
    1) explicit schema/relation-hop metadata;
    2) template lookup;
    3) fallback to observed first candidate path types and relations.
    """
    parsed = _from_relation_hops(example)
    if parsed is not None:
        return parsed

    template = str(example.get("template") or example.get("question_template") or "").strip().lower()
    if template in TEMPLATE_SCHEMA_MAP:
        return TEMPLATE_SCHEMA_MAP[template]

    # Fallback: infer from first available path. This is deterministic but should be avoided for final papers.
    for cand in example.get("candidates", []):
        for path in cand.get("paths", []):
            nodes = path.get("nodes", [])
            relations = path.get("relations", [])
            if len(nodes) >= 2 and len(relations) == len(nodes) - 1:
                entity_types = [str(n.get("type", "unknown")).lower() for n in nodes]
                rel_fams = [relation_family(r.get("name", r) if isinstance(r, dict) else r) for r in relations]
                return QuestionSchema(entity_types=entity_types, relation_families=rel_fams)

    anchor_type = example.get("anchor", {}).get("type", "unknown")
    answer_type = example.get("answer_type", "unknown")
    return QuestionSchema(entity_types=[anchor_type, answer_type], relation_families=[relation_family("related to")])
