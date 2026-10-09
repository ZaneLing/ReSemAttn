from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .attention import ReSemAttentionLayer
from .graph import BiomedicalKG
from .memory import PathMemoryEncoder, RelationSemanticPrior
from .schema import SoftSchemaPredictor
from .text_encoder import BaseTextEncoder
from .types import CandidatePrediction, KGPath, RelationSchema


@dataclass
class CandidateScore:
    score: torch.Tensor
    path_scores: torch.Tensor
    attentions: list[torch.Tensor]


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
            raise ValueError(
                "model.hidden_dim must match encoder hidden_dim in this implementation."
            )
        self.reasoner = config.get("reasoner", "resemattn")
        self.relation_conditioned = bool(config.get("relation_conditioned", True))
        self.use_prior = bool(config.get("use_attention_prior", True))
        self.path_representation = config.get("path_representation", "structured")
        if self.reasoner not in ("resemattn", "cross_encoder"):
            raise ValueError("Unknown reasoner.")
        if self.path_representation not in ("structured", "text"):
            raise ValueError("Unknown path representation.")
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

    def _schema_text(self, schemas: list[RelationSchema], probabilities=None) -> str:
        segments = []
        for schema_index, schema in enumerate(schemas):
            chain = []
            for idx, relation in enumerate(schema.relations):
                direction = "forward" if schema.directions[idx] == 1 else "inverse"
                chain.append(f"{schema.entity_types[idx]} {direction} {relation}")
            chain.append(schema.entity_types[-1])
            probability = (
                schema.probability
                if probabilities is None
                else float(probabilities[schema_index].detach())
            )
            segments.append(f"p={probability:.4f}: " + " -> ".join(chain))
        return " ; ".join(segments)

    def forward_candidate(
        self,
        question: str,
        candidate_id: str,
        paths: list[KGPath],
        schemas: list[RelationSchema],
        device: torch.device | str,
        schema_probabilities: torch.Tensor | None = None,
    ) -> CandidateScore:
        if not paths or any(path.end != candidate_id for path in paths):
            raise ValueError("Every nonempty candidate path must end at its candidate.")
        question_batch = self.text_encoder.encode([question], device=device)
        # Figure 2: H^(0)=Encoder(q,a_i); h_q remains the original question representation.
        token_batch = self.text_encoder.encode(
            [question + " [CANDIDATE] " + self.kg.entity(candidate_id).text], device=device
        )
        token_states = token_batch.token_states[0, token_batch.attention_mask[0]].float()
        question_vector = question_batch.pooled[0].float()
        candidate_vector = (
            self.text_encoder.encode([self.kg.entity(candidate_id).text], device=device)
            .pooled[0]
            .float()
        )
        if self.reasoner == "cross_encoder":
            texts = [
                question
                + " [ANSWER] "
                + self.kg.entity(candidate_id).text
                + " [PATH] "
                + self.kg.describe_path(path.nodes, path.relations, path.directions)
                for path in paths
            ]
            joint = self.text_encoder.encode(texts, device=device).pooled.float()
            path_scores = self.scorer(
                torch.cat(
                    [joint, candidate_vector.expand_as(joint), question_vector.expand_as(joint)], -1
                )
            ).squeeze(-1)
            return CandidateScore(
                torch.logsumexp(path_scores, 0) - math.log(len(paths)), path_scores, []
            )
        schema_vector = self.schema_projection(
            self.text_encoder.encode(
                [self._schema_text(schemas, schema_probabilities)], device=device
            )
            .pooled[0]
            .float()
        )
        path_memory, relation_vectors, topology = self.path_encoder.encode_paths(paths, device)
        if self.path_representation == "text":
            path_memory = self.text_encoder.encode(
                [
                    self.kg.describe_path(path.nodes, path.relations, path.directions)
                    for path in paths
                ],
                device=device,
            ).pooled.float()
        absolute_scores, prior = self.path_prior(
            paths, schemas, topology, self.path_encoder, schema_probabilities
        )
        confidence = (
            self.schema_predictor.confidence_tensor(schema_probabilities)
            if schema_probabilities is not None
            else self.schema_predictor.confidence(schemas)
        )
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
                relation_conditioned=self.relation_conditioned,
                use_prior=self.use_prior,
            )
            attentions.append(attention)
        pooled = token_states.mean(dim=0)
        language_score = self.scorer(
            torch.cat([pooled, candidate_vector, question_vector], dim=-1)
        ).squeeze(-1)
        normalized_lse = torch.logsumexp(absolute_scores, dim=0) - math.log(len(paths))
        total_score = language_score + self.path_score_weight * normalized_lse
        return CandidateScore(total_score, absolute_scores, attentions)

    def forward(self, question, candidate_paths, schemas, device, schema_probabilities=None):
        return {
            candidate: self.forward_candidate(
                question, candidate, paths, schemas, device, schema_probabilities
            )
            for candidate, paths in sorted(candidate_paths.items())
            if paths
        }

    def score_candidate(self, question, candidate_id, paths, schemas, device):
        output = self.forward_candidate(question, candidate_id, paths, schemas, device)
        from .types import path_key

        indices = sorted(
            range(len(paths)),
            key=lambda i: (-float(output.path_scores[i].detach()), path_key(paths[i])),
        )
        return CandidatePrediction(
            entity_id=candidate_id,
            score=float(output.score.detach()),
            probability=float(output.score.detach().sigmoid()),
            path_scores=[float(output.path_scores[i].detach()) for i in indices],
            supporting_paths=[paths[i] for i in indices],
            attentions=[a[:, indices].detach().cpu() for a in output.attentions],
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
        predictions.sort(key=lambda item: (-item.score, item.entity_id))
        return predictions
