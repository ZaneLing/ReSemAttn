from __future__ import annotations

import math

import torch
from torch import nn

from .attention import ReSemAttentionLayer
from .graph import BiomedicalKG
from .memory import PathMemoryEncoder, RelationSemanticPrior
from .schema import SoftSchemaPredictor
from .text_encoder import BaseTextEncoder
from .types import CandidatePrediction, KGPath, RelationSchema


class ReSemReasonModel(nn.Module):
    def __init__(
        self,
        kg: BiomedicalKG,
        text_encoder: BaseTextEncoder,
        schema_predictor: SoftSchemaPredictor,
        config: dict,
    ) -> None:
        super().__init__()
        hidden_dim = int(config.get("hidden_dim", text_encoder.hidden_dim))
        if hidden_dim != text_encoder.hidden_dim:
            raise ValueError("model.hidden_dim must match encoder hidden_dim in this implementation.")
        self.kg = kg
        self.text_encoder = text_encoder
        self.schema_predictor = schema_predictor
        self.path_encoder = PathMemoryEncoder(
            kg=kg,
            hidden_dim=hidden_dim,
            type_dim=int(config.get("type_dim", 32)),
            relation_dim=int(config.get("relation_dim", 64)),
            topology_dim=int(config.get("topology_dim", 16)),
            dropout=float(config.get("dropout", 0.1)),
        )
        self.path_prior = RelationSemanticPrior()
        self.layers = nn.ModuleList(
            ReSemAttentionLayer(
                hidden_dim=hidden_dim,
                relation_bases=int(config.get("relation_bases", 4)),
                relation_rank=int(config.get("relation_rank", 16)),
                prior_strength=float(config.get("prior_strength", 1.0)),
                dropout=float(config.get("dropout", 0.1)),
            )
            for _ in range(int(config.get("attention_layers", 2)))
        )
        self.schema_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(config.get("dropout", 0.1))),
            nn.Linear(hidden_dim, 1),
        )
        self.path_score_weight = float(config.get("path_score_weight", 0.5))
        self.answer_threshold = float(config.get("answer_threshold", 0.5))

    def _schema_text(self, schemas: list[RelationSchema]) -> str:
        segments = []
        for schema in schemas:
            chain = []
            for idx, relation in enumerate(schema.relations):
                direction = "forward" if schema.directions[idx] == 1 else "inverse"
                chain.append(f"{schema.entity_types[idx]} {direction} {relation}")
            chain.append(schema.entity_types[-1])
            segments.append(f"p={schema.probability:.4f}: " + " -> ".join(chain))
        return " ; ".join(segments)

    def score_candidate(
        self,
        question: str,
        candidate_id: str,
        paths: list[KGPath],
        schemas: list[RelationSchema],
        device: torch.device | str,
    ) -> CandidatePrediction:
        question_batch = self.text_encoder.encode([question], device=device)
        token_states = question_batch.token_states[0, question_batch.attention_mask[0]]
        question_vector = question_batch.pooled[0]
        candidate_vector = self.text_encoder.encode(
            [self.kg.entity(candidate_id).text], device=device
        ).pooled[0]
        schema_vector = self.schema_projection(
            self.text_encoder.encode([self._schema_text(schemas)], device=device).pooled[0]
        )
        path_memory, relation_vectors, topology = self.path_encoder.encode_paths(paths, device)
        absolute_scores, prior = self.path_prior(paths, schemas, topology, self.path_encoder)
        confidence = self.schema_predictor.confidence(schemas)
        attentions = []
        for layer in self.layers:
            token_states, attention = layer(
                token_states=token_states,
                path_memory=path_memory,
                relation_vectors=relation_vectors,
                path_prior=prior,
                schema_vector=schema_vector,
                candidate_vector=candidate_vector,
                schema_confidence=confidence,
            )
            attentions.append(attention.detach().cpu())
        pooled = token_states.mean(dim=0)
        language_score = self.scorer(
            torch.cat([pooled, candidate_vector, question_vector], dim=-1)
        ).squeeze(-1)
        normalized_lse = torch.logsumexp(absolute_scores, dim=0) - math.log(len(paths))
        total_score = language_score + self.path_score_weight * normalized_lse
        sorted_indices = absolute_scores.argsort(descending=True).tolist()
        sorted_paths = [paths[index] for index in sorted_indices]
        sorted_scores = [float(absolute_scores[index].detach().cpu()) for index in sorted_indices]
        return CandidatePrediction(
            entity_id=candidate_id,
            score=float(total_score.detach().cpu()),
            probability=0.0,
            path_scores=sorted_scores,
            supporting_paths=sorted_paths,
            attentions=attentions,
        )

    def score_candidates(
        self,
        question: str,
        candidate_paths: dict[str, list[KGPath]],
        schemas: list[RelationSchema],
        device: torch.device | str,
    ) -> list[CandidatePrediction]:
        predictions = [
            self.score_candidate(question, candidate_id, paths, schemas, device)
            for candidate_id, paths in candidate_paths.items()
            if paths
        ]
        if not predictions:
            return []
        scores = torch.tensor([item.score for item in predictions], device=device)
        probabilities = torch.softmax(scores, dim=0).detach().cpu().tolist()
        for item, probability in zip(predictions, probabilities):
            item.probability = float(probability)
        predictions.sort(key=lambda item: item.score, reverse=True)
        return predictions
