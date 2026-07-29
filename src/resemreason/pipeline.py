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
        self.device = torch.device(
            requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
        )
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
    def from_yaml(cls, path: str | Path) -> "ReSemReasonPipeline":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
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
    def infer(self, question: str, use_gold_schemas=None, use_gold_anchors=None) -> InferenceResult:
        self.model.eval()
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
        schemas = list(use_gold_schemas) if use_gold_schemas else self.schema_predictor.predict(
            question,
            top_m=int(schema_cfg.get("top_m", 4)),
            beam_size=int(schema_cfg.get("beam_size", 12)),
            device=self.device,
        )
        candidate_paths = self.searcher.search(
            question,
            groundings=groundings,
            schemas=schemas,
            device=self.device,
        )
        predictions = self.model.score_candidates(
            question,
            candidate_paths=candidate_paths,
            schemas=schemas,
            device=self.device,
        )
        threshold = float(self.config["model"].get("answer_threshold", 0.5))
        answer_ids = [item.entity_id for item in predictions if item.probability >= threshold]
        if not answer_ids and predictions:
            answer_ids = [predictions[0].entity_id]
        return InferenceResult(
            question=question,
            groundings=groundings,
            schemas=schemas,
            predictions=predictions,
            answer_ids=answer_ids,
        )

    def save(self, path: str | Path) -> None:
        torch.save({"model": self.model.state_dict(), "config": self.config}, path)

    def load(self, path: str | Path, strict: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=strict)
