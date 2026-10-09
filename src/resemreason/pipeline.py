from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .graph import BiomedicalKG
from .grounding import EntityGrounder
from .model import ReSemReasonModel
from .schema import SoftSchemaPredictor
from .search import RelationGuidedSearcher
from .text_encoder import build_text_encoder
from .types import InferenceResult


class ReSemReasonPipeline:
    def __init__(self, config: dict, base_dir: str | Path = ".") -> None:
        self.config = config
        self.base_dir = Path(base_dir)
        self._set_seed(int(config.get("seed", 42)))
        requested_device = config.get("device", "cpu")
        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; choose a CPU/toy configuration explicitly."
            )
        self.device = torch.device(requested_device)
        data = config["data"]
        self.kg = BiomedicalKG.from_jsonl(
            self._resolve(data["entities"]),
            self._resolve(data["edges"]),
            add_inverse=bool(data.get("add_inverse_edges", True)),
        )
        self.text_encoder = build_text_encoder(config["encoder"])
        self.text_encoder.to(self.device)
        self.grounder = EntityGrounder(self.text_encoder, self.kg).to(self.device)
        self.schema_predictor = SoftSchemaPredictor(
            self.text_encoder,
            self.kg,
            max_hops=int(config["schema"].get("max_hops", 3)),
        ).to(self.device)
        self.searcher = RelationGuidedSearcher(self.kg, self.text_encoder, config["search"])
        self.model = ReSemReasonModel(
            self.kg,
            self.text_encoder,
            self.schema_predictor,
            config["model"],
        ).to(self.device)

    @classmethod
    def from_yaml(
        cls, path: str | Path, seed: int | None = None, variant: str | None = None
    ) -> "ReSemReasonPipeline":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if seed is not None:
            config["seed"] = seed
        if variant is not None:
            variants = yaml.safe_load((path.parent / "controls.yaml").read_text())
            if variant not in variants:
                raise ValueError(f"Unknown control variant: {variant}")
            for section, values in variants[variant].items():
                config[section].update(values)
            config["variant"] = variant
        # Resolve paths against repository root when configs live in configs/.
        base_dir = path.parent.parent if path.parent.name == "configs" else path.parent
        return cls(config, base_dir=base_dir)

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.base_dir / candidate

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def predict_structure(self, question: str, use_gold_schemas=None, use_gold_anchors=None):
        self.model.eval()
        if (use_gold_schemas is not None or use_gold_anchors is not None) and self.config[
            "data"
        ].get("kind") != "toy":
            raise ValueError("Gold inference overrides are restricted to explicit toy data.")
        search_cfg = self.config["search"]
        schema_cfg = self.config["schema"]
        if use_gold_anchors:
            probability = 1.0 / len(use_gold_anchors)
            from .types import GroundingCandidate

            groundings = [
                GroundingCandidate(entity_id=item, probability=probability, score=0.0)
                for item in use_gold_anchors
            ]
        else:
            groundings = self.grounder.ground(
                question,
                top_n=int(search_cfg.get("top_n_anchors", 5)),
                device=self.device,
            )
        schemas = (
            list(use_gold_schemas)
            if use_gold_schemas
            else self.schema_predictor.predict(
                question,
                top_m=int(schema_cfg.get("top_m", 4)),
                beam_size=int(schema_cfg.get("beam_size", 12)),
                device=self.device,
            )
        )
        schemas = self.adjust_schemas(schemas)
        return groundings, schemas

    @torch.no_grad()
    def retrieve(self, question: str, use_gold_schemas=None, use_gold_anchors=None):
        groundings, schemas = self.predict_structure(question, use_gold_schemas, use_gold_anchors)
        candidate_paths = self.searcher.search(
            question,
            groundings=groundings,
            schemas=schemas,
            device=self.device,
        )
        return groundings, schemas, candidate_paths

    @torch.no_grad()
    def infer(self, question: str, use_gold_schemas=None, use_gold_anchors=None) -> InferenceResult:
        groundings, schemas, candidate_paths = self.retrieve(
            question, use_gold_schemas, use_gold_anchors
        )
        return self.infer_pool(question, groundings, schemas, candidate_paths)

    @torch.no_grad()
    def infer_pool(self, question, groundings, schemas, candidate_paths):
        self.model.eval()
        for candidate, paths in candidate_paths.items():
            for path in paths:
                if path.end != candidate:
                    raise ValueError("Frozen pool endpoint mismatch.")
                self.kg.validate_path(path)
        schemas = self.adjust_schemas(schemas)
        predictions = self.model.score_candidates(
            question,
            candidate_paths=candidate_paths,
            schemas=schemas,
            device=self.device,
        )
        threshold = float(self.config["model"].get("answer_threshold", 0.5))
        answer_ids = [item.entity_id for item in predictions if item.probability >= threshold]
        return InferenceResult(
            question=question,
            groundings=groundings,
            schemas=schemas,
            predictions=predictions,
            answer_ids=answer_ids,
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "format_version": 2,
                "model": self.model.state_dict(),
                "grounder_temperature": self.grounder.log_temperature.detach().cpu(),
                "config": self.config,
                "graph_checksums": self.graph_checksums(),
                "kg_signature": {
                    "entities": sorted(self.kg.entities),
                    "types": self.kg.entity_types,
                    "relations": self.kg.relations,
                },
            },
            path,
        )

    def load(self, path: str | Path, strict: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("format_version") != 2:
            raise ValueError(
                "Legacy surrogate-trained checkpoint: retrain with the full-forward trainer."
            )
        if checkpoint.get("graph_checksums") != self.graph_checksums():
            raise ValueError("Checkpoint graph snapshot differs from the configured files.")
        signature = checkpoint.get("kg_signature")
        expected = {
            "entities": sorted(self.kg.entities),
            "types": self.kg.entity_types,
            "relations": self.kg.relations,
        }
        if signature is not None and signature != expected:
            raise ValueError(
                "Checkpoint KG inventory/vocabulary differs from the configured graph."
            )
        saved = checkpoint["config"]["model"]
        defaults = {
            "reasoner": "resemattn",
            "relation_conditioned": True,
            "use_attention_prior": True,
            "path_representation": "structured",
        }
        for key, default in defaults.items():
            if self.config["model"].get(key, default) != saved.get(key, default):
                raise ValueError(
                    f"Checkpoint protocol mismatch for {key}; use the matching --variant/config."
                )
        if self.config["model"].get("path_score_weight", 0.5) != saved.get(
            "path_score_weight", 0.5
        ):
            raise ValueError("Checkpoint absolute-support weight differs from configuration.")
        for section in ("schema", "search"):
            if self.config[section] != checkpoint["config"][section]:
                raise ValueError(
                    f"Checkpoint {section} protocol differs; use the matching --variant/config."
                )
        if self.config["data"].get("add_inverse_edges", True) != checkpoint["config"]["data"].get(
            "add_inverse_edges", True
        ):
            raise ValueError("Checkpoint inverse-edge policy differs from configuration.")
        self.model.load_state_dict(checkpoint["model"], strict=strict)
        self.config["seed"] = int(checkpoint["config"]["seed"])
        self._set_seed(self.config["seed"])
        if "grounder_temperature" in checkpoint:
            self.grounder.log_temperature.copy_(checkpoint["grounder_temperature"])
        threshold = checkpoint.get("config", {}).get("model", {}).get("answer_threshold")
        if threshold is not None:
            self.config["model"]["answer_threshold"] = float(threshold)
            self.model.answer_threshold = float(threshold)

    def graph_checksums(self):
        from .evaluation import sha256

        return {
            key: sha256(self._resolve(self.config["data"][key])) for key in ("entities", "edges")
        }

    def adjust_schemas(self, schemas):
        mode = self.config["schema"].get("probability_mode", "predicted")
        if mode == "predicted":
            return schemas
        if mode != "uniform":
            raise ValueError("Unsupported schema probability intervention.")
        from dataclasses import replace

        return [replace(s, probability=1 / len(schemas)) for s in schemas]
