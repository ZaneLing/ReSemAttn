from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.optim import AdamW

from .data import QuestionExample
from .losses import combined_loss
from .pipeline import ReSemReasonPipeline


@dataclass
class TrainStats:
    loss: float
    examples: int
    skipped: int = 0
    components: dict[str, float] = field(default_factory=dict)


class ReSemReasonTrainer:
    """Appendix C completion: train the actual inference forward, with discrete retrieval."""

    def __init__(self, pipeline: ReSemReasonPipeline) -> None:
        self.pipeline = pipeline
        self.config = pipeline.config["training"]
        self.parameters = list(pipeline.model.parameters())
        self.optimizer = AdamW(
            self.parameters,
            lr=float(self.config.get("learning_rate", 5e-4)),
            weight_decay=float(self.config.get("weight_decay", 0.01)),
        )

    def loss_for_example(self, example: QuestionExample):
        if example.split not in (None, "train"):
            raise ValueError("Supervised training accepts train examples only.")
        p = self.pipeline
        # Inference-time inputs remain question-only; no gold schemas injected into search.
        p.model.eval()
        with torch.no_grad():
            groundings = p.grounder.ground(
                example.question, top_n=int(p.config["search"]["top_n_anchors"]), device=p.device
            )
            schemas = p.schema_predictor.predict(
                example.question,
                top_m=int(p.config["schema"]["top_m"]),
                beam_size=int(p.config["schema"]["beam_size"]),
                device=p.device,
            )
            schemas = p.adjust_schemas(schemas)
            pools = p.searcher.search(example.question, groundings, schemas, p.device)
        p.model.train()
        if not pools:
            return None, {"skipped": 1.0}
        schema_logits = p.schema_predictor(example.question, p.device)
        rho = p.schema_predictor.log_probabilities(schema_logits, schemas).softmax(0)
        if p.config["schema"].get("probability_mode") == "uniform":
            rho = torch.ones_like(rho) / len(rho)
        outputs = p.model(example.question, pools, schemas, p.device, rho)
        candidate_ids = list(outputs)
        logits = torch.stack([outputs[c].score for c in candidate_ids])
        labels = logits.new_tensor([float(c in example.gold_answers) for c in candidate_ids])
        positives, negatives = [], []
        for positive, negative in example.candidate_pairs:
            if positive not in example.gold_answers or negative in example.gold_answers:
                raise ValueError("Matched candidate pairs must agree with gold answer labels.")
            if p.kg.entity(positive).entity_type != p.kg.entity(negative).entity_type:
                raise ValueError("Matched candidate pairs must have the same endpoint type.")
            if positive in outputs and negative in outputs:
                positives.append(outputs[positive].score)
                negatives.append(outputs[negative].score)
        valid_scores, invalid_scores = [], []
        for valid, invalid in example.path_pairs:
            if valid.hops != invalid.hops:
                raise ValueError("Matched path pairs must have equal hop counts.")
            if valid.end != invalid.end:
                raise ValueError("Path pairs must hold the answer fixed.")
            p.kg.validate_path(valid)
            p.kg.validate_path(invalid)
            if p.model.reasoner == "cross_encoder":
                scores = p.model.forward_candidate(
                    example.question, valid.end, [valid, invalid], schemas, p.device, rho
                ).path_scores
            else:
                _, _, topology = p.model.path_encoder.encode_paths([valid, invalid], p.device)
                scores, _ = p.model.path_prior(
                    [valid, invalid], schemas, topology, p.model.path_encoder, rho
                )
            valid_scores.append(scores[0])
            invalid_scores.append(scores[1])

        def stack(xs):
            return torch.stack(xs) if xs else None

        loss, parts = combined_loss(
            logits,
            labels,
            self.config,
            stack(positives),
            stack(negatives),
            stack(valid_scores),
            stack(invalid_scores),
        )
        if example.gold_schemas:
            schema_loss = p.schema_predictor.supervision_loss(schema_logits, example.gold_schemas)
            loss = loss + float(self.config.get("schema_weight", 1.0)) * schema_loss
            parts["schema"] = float(schema_loss.detach())
        if example.gold_anchors:
            scores = p.grounder.score_all(example.question, p.device).float()
            indices = [p.grounder.entity_ids.index(c) for c in set(example.gold_anchors)]
            grounding_loss = torch.logsumexp(scores, 0) - torch.logsumexp(scores[indices], 0)
            loss = loss + float(self.config.get("grounding_weight", 1.0)) * grounding_loss
            parts["grounding"] = float(grounding_loss.detach())
        parts["candidate_pairs"] = len(positives)
        parts["path_pairs"] = len(valid_scores)
        parts["total"] = float(loss.detach())
        return loss, parts

    def train_epoch(self, examples: list[QuestionExample]) -> TrainStats:
        total, used, pending, skipped, components = 0.0, 0, 0, 0, {}
        batch_size = int(self.config.get("batch_size", 1))
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.optimizer.zero_grad(set_to_none=True)
        for example in examples:
            loss, parts = self.loss_for_example(example)
            if loss is None:
                skipped += 1
                continue
            loss.backward()
            total += float(loss.detach())
            used += 1
            pending += 1
            for name, value in parts.items():
                components[name] = components.get(name, 0.0) + value
            if pending == batch_size:
                self._step(pending)
                pending = 0
        if pending:
            self._step(pending)
        return TrainStats(
            total / max(used, 1),
            used,
            skipped,
            {k: v / max(used, 1) for k, v in components.items()},
        )

    def _step(self, count):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.div_(count)
        torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
