from __future__ import annotations

import torch
import torch.nn.functional as F


def pointwise_answer_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def pairwise_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    return F.relu(margin - positive_scores + negative_scores).mean()


def multi_positive_listwise_loss(scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 1 or positive_mask.ndim != 1:
        raise ValueError("scores and positive_mask must be rank-1 tensors.")
    if not positive_mask.any():
        raise ValueError("At least one positive candidate is required.")
    numerator = torch.logsumexp(scores[positive_mask], dim=0)
    denominator = torch.logsumexp(scores, dim=0)
    return denominator - numerator


def path_pair_loss(
    valid_scores: torch.Tensor, invalid_scores: torch.Tensor, margin: float = 0.2
) -> torch.Tensor:
    return pairwise_ranking_loss(valid_scores, invalid_scores, margin=margin)


def brier_calibration_loss(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return ((probabilities - labels.float()) ** 2).mean()


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: dict,
    positive_scores: torch.Tensor | None = None,
    negative_scores: torch.Tensor | None = None,
    valid_path_scores: torch.Tensor | None = None,
    invalid_path_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    pointwise = pointwise_answer_loss(logits, labels)
    total = float(config.get("pointwise_weight", 1.0)) * pointwise
    components = {"pointwise": float(pointwise.detach())}

    if positive_scores is not None and negative_scores is not None:
        pairwise = pairwise_ranking_loss(
            positive_scores,
            negative_scores,
            margin=float(config.get("pairwise_margin", 0.2)),
        )
        total = total + float(config.get("pairwise_weight", 0.5)) * pairwise
        components["pairwise"] = float(pairwise.detach())

    positive_mask = labels.bool()
    if positive_mask.any():
        listwise = multi_positive_listwise_loss(logits, positive_mask)
        total = total + float(config.get("listwise_weight", 0.5)) * listwise
        components["listwise"] = float(listwise.detach())

    if valid_path_scores is not None and invalid_path_scores is not None:
        path_loss = path_pair_loss(
            valid_path_scores, invalid_path_scores, margin=float(config.get("pairwise_margin", 0.2))
        )
        total = total + float(config.get("path_weight", 0.25)) * path_loss
        components["path"] = float(path_loss.detach())

    calibration = brier_calibration_loss(torch.sigmoid(logits), labels)
    total = total + float(config.get("calibration_weight", 0.1)) * calibration
    components["calibration"] = float(calibration.detach())
    components["total"] = float(total.detach())
    return total, components
