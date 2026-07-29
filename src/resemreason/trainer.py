from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import AdamW

from .data import QuestionExample
from .losses import combined_loss
from .pipeline import ReSemReasonPipeline


@dataclass
class TrainStats:
    loss: float
    examples: int


class ReSemReasonTrainer:
    """Small reference trainer for candidate scoring after discrete search."""

    def __init__(self, pipeline: ReSemReasonPipeline) -> None:
        self.pipeline = pipeline
        config = pipeline.config["training"]
        self.optimizer = AdamW(
            pipeline.model.parameters(),
            lr=float(config.get("learning_rate", 5e-4)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
        self.config = config

    def train_epoch(self, examples: list[QuestionExample]) -> TrainStats:
        self.pipeline.model.train()
        total_loss = 0.0
        used = 0
        for example in examples:
            groundings = self.pipeline.grounder.ground(
                example.question,
                top_n=int(self.pipeline.config["search"].get("top_n_anchors", 5)),
                device=self.pipeline.device,
            )
            schemas = list(example.gold_schemas) or self.pipeline.schema_predictor.predict(
                example.question,
                top_m=int(self.pipeline.config["schema"].get("top_m", 4)),
                beam_size=int(self.pipeline.config["schema"].get("beam_size", 12)),
                device=self.pipeline.device,
            )
            candidate_paths = self.pipeline.searcher.search(
                example.question, groundings, schemas, self.pipeline.device
            )
            if not candidate_paths:
                continue
            # score_candidate converts its public outputs to floats, so train directly through
            # the differentiable submodules on a compact surrogate candidate representation.
            question_vector = self.pipeline.text_encoder.encode(
                [example.question], device=self.pipeline.device
            ).pooled[0]
            logits = []
            labels = []
            for candidate_id, paths in candidate_paths.items():
                candidate_vector = self.pipeline.text_encoder.encode(
                    [self.pipeline.kg.entity(candidate_id).text], device=self.pipeline.device
                ).pooled[0]
                path_memory, _, topology = self.pipeline.model.path_encoder.encode_paths(
                    paths, self.pipeline.device
                )
                absolute, _ = self.pipeline.model.path_prior(
                    paths, schemas, topology, self.pipeline.model.path_encoder
                )
                pooled_path = path_memory.mean(dim=0)
                language_logit = self.pipeline.model.scorer(
                    torch.cat([question_vector, candidate_vector, pooled_path], dim=-1)
                ).squeeze(-1)
                path_support = torch.logsumexp(absolute, dim=0) - torch.log(
                    absolute.new_tensor(float(len(paths)))
                )
                logits.append(language_logit + self.pipeline.model.path_score_weight * path_support)
                labels.append(float(candidate_id in example.gold_answers))
            logits_tensor = torch.stack(logits)
            labels_tensor = torch.tensor(labels, device=self.pipeline.device)
            loss, _ = combined_loss(logits_tensor, labels_tensor, self.config)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pipeline.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += float(loss.detach())
            used += 1
        return TrainStats(loss=total_loss / max(1, used), examples=used)
