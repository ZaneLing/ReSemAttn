from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F


def pointwise_bce_loss(scores: torch.Tensor, labels: torch.Tensor, positive_weight: float = 1.0) -> torch.Tensor:
    pos_weight = torch.tensor(float(positive_weight), device=scores.device)
    return F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=pos_weight)


def pairwise_margin_loss(scores: torch.Tensor, labels: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.new_tensor(0.0)
    diff = margin - pos.view(-1, 1) + neg.view(1, -1)
    return F.relu(diff).mean()


def multi_positive_listwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pos_mask = labels > 0.5
    if pos_mask.sum() == 0:
        return scores.new_tensor(0.0)
    log_num = torch.logsumexp(scores[pos_mask], dim=0)
    log_den = torch.logsumexp(scores, dim=0)
    return -(log_num - log_den)


def combined_loss(scores: torch.Tensor, labels: torch.Tensor, cfg: Dict) -> Dict[str, torch.Tensor]:
    tr = cfg.get("training", cfg)
    lans = pointwise_bce_loss(scores, labels, tr.get("positive_weight", 1.0))
    lpair = pairwise_margin_loss(scores, labels, tr.get("pair_margin", 0.25))
    llist = multi_positive_listwise_loss(scores, labels)
    loss = lans + tr.get("lambda_pair", 0.2) * lpair + tr.get("lambda_list", 0.1) * llist
    return {"loss": loss, "L_ans": lans.detach(), "L_pair": lpair.detach(), "L_list": llist.detach()}
