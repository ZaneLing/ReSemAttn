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
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())
        self.entity_ids = list(kg.entities)

    def score_all(self, question: str, device: torch.device | str) -> torch.Tensor:
        question_vec = self.text_encoder.encode([question], device=device).pooled[0]
        entity_texts = [self.kg.entity(entity_id).text for entity_id in self.entity_ids]
        entity_vecs = self.text_encoder.encode(entity_texts, device=device).pooled
        question_vec = F.normalize(question_vec, dim=-1)
        entity_vecs = F.normalize(entity_vecs, dim=-1)
        temperature = self.log_temperature.exp().clamp_min(1e-3)
        return entity_vecs @ question_vec / temperature

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
        values, indices = probabilities.topk(count)
        return [
            GroundingCandidate(
                entity_id=self.entity_ids[index.item()],
                probability=float(value.item()),
                score=float(scores[index].item()),
            )
            for value, index in zip(values, indices)
        ]
