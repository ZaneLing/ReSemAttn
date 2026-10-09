from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .graph import BiomedicalKG
from .text_encoder import BaseTextEncoder
from .types import GroundingCandidate


class EntityGrounder(nn.Module):
    def __init__(self, text_encoder: BaseTextEncoder, kg: BiomedicalKG, temperature: float = 0.1):
        super().__init__()
        self.text_encoder = text_encoder
        self.kg = kg
        self.register_buffer("log_temperature", torch.tensor(float(temperature)).log())
        self.entity_ids = sorted(kg.entities)

    def score_all(self, question: str, device: torch.device | str) -> torch.Tensor:
        question_vec = self.text_encoder.encode([question], device=device).pooled[0]
        question_vec = F.normalize(question_vec.float(), dim=-1)
        temperature = self.log_temperature.exp().clamp_min(1e-3)
        scores = []
        # Score the complete inventory, with bounded encoder batches and no persistent cache.
        for start in range(0, len(self.entity_ids), 32):
            texts = [self.kg.entity(e).text for e in self.entity_ids[start : start + 32]]

            def score_chunk(q, texts=texts):
                vectors = self.text_encoder.encode(texts, device=device).pooled.float()
                return F.normalize(vectors, dim=-1) @ q / temperature

            if torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                scores.append(checkpoint(score_chunk, question_vec, use_reentrant=False))
            else:
                scores.append(score_chunk(question_vec))
        return torch.cat(scores)

    @torch.no_grad()
    def ground(
        self,
        question: str,
        top_n: int = 5,
        device: torch.device | str = "cpu",
    ) -> list[GroundingCandidate]:
        scores = self.score_all(question, device=device)
        probabilities = scores.softmax(dim=0)
        count = min(top_n, len(self.entity_ids))
        indices = torch.argsort(probabilities, descending=True, stable=True)[:count]
        values = probabilities[indices]
        return [
            GroundingCandidate(
                entity_id=self.entity_ids[index.item()],
                probability=float(value.item()),
                score=float(scores[index].item()),
            )
            for value, index in zip(values, indices)
        ]
