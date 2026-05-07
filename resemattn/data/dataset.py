from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import torch
from torch.utils.data import Dataset

from resemattn.data.schema import infer_schema, normalize_relation_name
from resemattn.utils.io import read_jsonl
from resemattn.utils.text import simple_tokenize, hash_tokens, stable_hash


@dataclass
class TensorizedQuestion:
    qid: str
    token_ids: torch.Tensor              # [N, n]
    attn_mask: torch.Tensor              # [N, n]
    candidate_span: torch.Tensor         # [N, 2]
    schema_span: torch.Tensor            # [N, 2]
    labels: torch.Tensor                 # [N]
    path_node_ids: torch.Tensor          # [N, K, L+1]
    path_type_ids: torch.Tensor          # [N, K, L+1]
    path_rel_ids: torch.Tensor           # [N, K, L]
    path_dir_ids: torch.Tensor           # [N, K, L]
    path_topology: torch.Tensor          # [N, K, 4]
    path_mask: torch.Tensor              # [N, K]
    schema_type_ids: torch.Tensor        # [L+1]
    schema_rel_ids: torch.Tensor         # [L, Rmax]
    schema_rel_mask: torch.Tensor        # [L, Rmax]
    candidate_ids: List[str]
    candidate_names: List[str]
    raw: Dict[str, Any]


class ReSemAttnDataset(Dataset):
    def __init__(self, path: str, cfg: Dict[str, Any]):
        self.rows = read_jsonl(path)
        self.cfg = cfg
        m = cfg["model"]
        self.vocab_size = int(m.get("vocab_size", 50000))
        self.entity_vocab_size = int(m.get("entity_vocab_size", 50000))
        self.relation_vocab_size = int(m.get("relation_vocab_size", 4096))
        self.type_vocab_size = int(m.get("type_vocab_size", 256))
        self.max_paths = int(m.get("max_paths", 8))
        self.max_candidates = int(m.get("max_candidates", 64))
        self.max_seq_len = int(m.get("max_seq_len", 256))
        self.max_hops = int(m.get("max_hops", 4))

    def __len__(self) -> int:
        return len(self.rows)

    def _type_id(self, t: str) -> int:
        return stable_hash(str(t).lower(), self.type_vocab_size)

    def _entity_id(self, e: str) -> int:
        return stable_hash(str(e).lower(), self.entity_vocab_size)

    def _rel_id(self, r: str) -> int:
        return stable_hash(normalize_relation_name(r), self.relation_vocab_size)

    def _dir_id(self, d: str) -> int:
        return {"forward": 1, "reverse": 2, "inverse": 2, "unknown": 0}.get(str(d).lower(), 0)

    def _build_text(self, row: Dict[str, Any], schema_text: str, cand: Dict[str, Any]) -> Tuple[List[int], Tuple[int,int], Tuple[int,int]]:
        question_toks = simple_tokenize(row.get("question", ""))
        schema_toks = simple_tokenize("schema " + schema_text)
        cand_toks = simple_tokenize("candidate " + cand.get("name", cand.get("id", "")) + " type " + cand.get("type", ""))
        path_toks: List[str] = []
        for path in cand.get("paths", [])[: self.max_paths]:
            for n in path.get("nodes", []):
                path_toks += simple_tokenize(n.get("name", n.get("id", "")) + " " + n.get("type", ""))
            for r in path.get("relations", []):
                if isinstance(r, dict):
                    path_toks += simple_tokenize(r.get("name", ""))
                else:
                    path_toks += simple_tokenize(r)
        # [BOS] question [SEP] schema [SEP] candidate [SEP] path [EOS]
        sep = ["[sep]"]
        all_toks = question_toks + sep
        schema_start = len(all_toks) + 1  # after BOS
        all_toks += schema_toks + sep
        schema_end = len(question_toks) + 1 + len(schema_toks)
        cand_start = len(all_toks) + 1
        all_toks += cand_toks + sep + path_toks
        cand_end = cand_start + len(cand_toks)
        ids = hash_tokens(all_toks, self.vocab_size)
        ids = ids[: self.max_seq_len]
        schema_span = (min(schema_start, len(ids)-1), min(schema_end, len(ids)-1))
        candidate_span = (min(cand_start, len(ids)-1), min(cand_end, len(ids)-1))
        return ids, schema_span, candidate_span

    def _tensorize_paths(self, cand: Dict[str, Any], schema) -> Tuple[torch.Tensor, ...]:
        K = self.max_paths
        L = min(max(schema.num_hops, 1), self.max_hops)
        node_ids = torch.zeros(K, L + 1, dtype=torch.long)
        type_ids = torch.zeros(K, L + 1, dtype=torch.long)
        rel_ids = torch.zeros(K, L, dtype=torch.long)
        dir_ids = torch.zeros(K, L, dtype=torch.long)
        topo = torch.zeros(K, 4, dtype=torch.float)
        pmask = torch.zeros(K, dtype=torch.bool)
        for k, path in enumerate(cand.get("paths", [])[:K]):
            nodes = path.get("nodes", [])
            rels = path.get("relations", [])
            pmask[k] = True
            for l in range(L + 1):
                n = nodes[l] if l < len(nodes) else (nodes[-1] if nodes else {})
                node_ids[k, l] = self._entity_id(n.get("id", n.get("name", "pad")))
                type_ids[k, l] = self._type_id(n.get("type", "unknown"))
            for l in range(L):
                r = rels[l] if l < len(rels) else {"name": "missing", "direction": "unknown"}
                if isinstance(r, dict):
                    name = r.get("name", "missing")
                    direction = r.get("direction", "unknown")
                else:
                    name, direction = str(r), "unknown"
                rel_ids[k, l] = self._rel_id(name)
                dir_ids[k, l] = self._dir_id(direction)
            t = path.get("topology", {}) or {}
            topo[k, 0] = float(t.get("bridge_degree", t.get("bridge", 0.0)))
            topo[k, 1] = float(t.get("endpoint_degree", t.get("endpoint", 0.0)))
            topo[k, 2] = float(t.get("branch", t.get("branching", 0.0)))
            topo[k, 3] = float(t.get("pathcount", t.get("path_count", 1.0)))
        return node_ids, type_ids, rel_ids, dir_ids, torch.log1p(topo), pmask

    def __getitem__(self, idx: int) -> TensorizedQuestion:
        row = self.rows[idx]
        schema = infer_schema(row)
        schema_text = schema.to_text()
        candidates = row.get("candidates", [])[: self.max_candidates]
        if not candidates:
            raise ValueError(f"Question {row.get('qid', idx)} has no candidates")
        token_rows, mask_rows, cspans, sspans, labels = [], [], [], [], []
        path_node, path_type, path_rel, path_dir, path_topo, path_mask = [], [], [], [], [], []
        candidate_ids, candidate_names = [], []
        for cand in candidates:
            ids, sspan, cspan = self._build_text(row, schema_text, cand)
            attn = [1] * len(ids)
            if len(ids) < self.max_seq_len:
                ids = ids + [0] * (self.max_seq_len - len(ids))
                attn = attn + [0] * (self.max_seq_len - len(attn))
            token_rows.append(torch.tensor(ids, dtype=torch.long))
            mask_rows.append(torch.tensor(attn, dtype=torch.bool))
            cspans.append(torch.tensor(cspan, dtype=torch.long))
            sspans.append(torch.tensor(sspan, dtype=torch.long))
            labels.append(float(cand.get("label", int(cand.get("name") in row.get("gold_answers", [])))))
            a,b,c,d,e,f = self._tensorize_paths(cand, schema)
            path_node.append(a); path_type.append(b); path_rel.append(c); path_dir.append(d); path_topo.append(e); path_mask.append(f)
            candidate_ids.append(str(cand.get("id", cand.get("name", ""))))
            candidate_names.append(str(cand.get("name", cand.get("id", ""))))
        # Schema tensors.
        L = min(max(schema.num_hops, 1), self.max_hops)
        Rmax = 4
        stypes = torch.zeros(L + 1, dtype=torch.long)
        srels = torch.zeros(L, Rmax, dtype=torch.long)
        srel_mask = torch.zeros(L, Rmax, dtype=torch.bool)
        for l in range(L + 1):
            typ = schema.entity_types[l] if l < len(schema.entity_types) else schema.entity_types[-1]
            stypes[l] = self._type_id(typ)
        for l in range(L):
            relfam = schema.relation_families[l] if l < len(schema.relation_families) else ["related to"]
            for j, r in enumerate(relfam[:Rmax]):
                srels[l, j] = self._rel_id(r)
                srel_mask[l, j] = True
        return TensorizedQuestion(
            qid=str(row.get("qid", idx)),
            token_ids=torch.stack(token_rows),
            attn_mask=torch.stack(mask_rows),
            candidate_span=torch.stack(cspans),
            schema_span=torch.stack(sspans),
            labels=torch.tensor(labels, dtype=torch.float),
            path_node_ids=torch.stack(path_node),
            path_type_ids=torch.stack(path_type),
            path_rel_ids=torch.stack(path_rel),
            path_dir_ids=torch.stack(path_dir),
            path_topology=torch.stack(path_topo),
            path_mask=torch.stack(path_mask),
            schema_type_ids=stypes,
            schema_rel_ids=srels,
            schema_rel_mask=srel_mask,
            candidate_ids=candidate_ids,
            candidate_names=candidate_names,
            raw=row,
        )


def move_question_to_device(q: TensorizedQuestion, device: torch.device) -> TensorizedQuestion:
    kwargs = q.__dict__.copy()
    for k, v in list(kwargs.items()):
        if torch.is_tensor(v):
            kwargs[k] = v.to(device)
    return TensorizedQuestion(**kwargs)
