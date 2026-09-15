"""Semantic relations and incrementally stable near-orthogonal codebooks.

This module is deliberately independent of CLIP and CUDA.  It contains the
small, deterministic tensor operations required by the semantic TOPECL
variant, so their mathematical invariants can be tested on CPU before the
modules are connected to the training pipeline.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_text_bank(text_bank: Tensor, mask: Optional[Tensor]) -> Tensor:
    if text_bank.ndim != 3:
        raise ValueError(
            "text_bank must have shape [num_classes, num_prototypes, dim], "
            f"got {tuple(text_bank.shape)}"
        )
    if text_bank.shape[0] == 0 or text_bank.shape[1] == 0:
        raise ValueError("text_bank must contain at least one class and prototype")

    if mask is None:
        mask = torch.ones(
            text_bank.shape[:2], dtype=torch.bool, device=text_bank.device
        )
    elif mask.shape != text_bank.shape[:2]:
        raise ValueError(
            "mask must have shape [num_classes, num_prototypes], "
            f"got {tuple(mask.shape)}"
        )
    else:
        mask = mask.to(device=text_bank.device, dtype=torch.bool)

    if not torch.all(mask.any(dim=1)):
        raise ValueError("every class must contain at least one valid prototype")
    return mask


def mmr_select(
    candidate_embeddings: Tensor,
    anchor_embedding: Tensor,
    num_select: int,
    relevance_weight: float = 0.7,
    distractor_embeddings: Optional[Tensor] = None,
    interclass_penalty: float = 0.0,
) -> Tensor:
    """Select relevant, discriminative, and non-redundant candidates.

    Args:
        candidate_embeddings: Candidate text features with shape [N, D].
        anchor_embedding: The frozen base class-name feature with shape [D].
        num_select: Number of candidate indices to return.
        relevance_weight: Trade-off in [0, 1] between class relevance and
            redundancy avoidance.
        distractor_embeddings: Seen competing class anchors with shape [K, D].
        interclass_penalty: Weight in [0, 1] assigned to the maximum similarity
            with any competing class.  Zero exactly recovers the original
            within-class MMR rule.
    """
    if candidate_embeddings.ndim != 2:
        raise ValueError("candidate_embeddings must have shape [N, D]")
    if anchor_embedding.ndim != 1:
        raise ValueError("anchor_embedding must have shape [D]")
    if candidate_embeddings.shape[1] != anchor_embedding.shape[0]:
        raise ValueError("candidate and anchor dimensions do not match")
    if not 0 <= num_select <= candidate_embeddings.shape[0]:
        raise ValueError("num_select must be between 0 and the candidate count")
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("relevance_weight must be in [0, 1]")
    if not 0.0 <= interclass_penalty <= 1.0:
        raise ValueError("interclass_penalty must be in [0, 1]")
    if distractor_embeddings is not None:
        if distractor_embeddings.ndim != 2:
            raise ValueError("distractor_embeddings must have shape [K, D]")
        if distractor_embeddings.shape[1] != anchor_embedding.shape[0]:
            raise ValueError("distractor and anchor dimensions do not match")
    if interclass_penalty > 0 and (
        distractor_embeddings is None or distractor_embeddings.shape[0] == 0
    ):
        raise ValueError(
            "positive interclass_penalty requires at least one distractor"
        )
    if num_select == 0:
        return torch.empty(0, dtype=torch.long, device=candidate_embeddings.device)

    candidates = F.normalize(candidate_embeddings, dim=-1)
    anchor = F.normalize(anchor_embedding, dim=-1)
    relevance = candidates @ anchor
    if interclass_penalty > 0:
        distractors = F.normalize(
            distractor_embeddings.to(
                device=candidate_embeddings.device,
                dtype=candidate_embeddings.dtype,
            ),
            dim=-1,
        )
        confusability = (candidates @ distractors.T).amax(dim=1)
        relevance = relevance - interclass_penalty * confusability
    pairwise = candidates @ candidates.T

    selected: list[int] = []
    available = torch.ones(
        candidate_embeddings.shape[0],
        dtype=torch.bool,
        device=candidate_embeddings.device,
    )
    for _ in range(num_select):
        if selected:
            redundancy = pairwise[:, selected].amax(dim=1)
        else:
            redundancy = torch.zeros_like(relevance)
        score = (
            relevance_weight * relevance
            - (1.0 - relevance_weight) * redundancy
        )
        score = score.masked_fill(~available, -torch.inf)
        index = int(torch.argmax(score).item())
        selected.append(index)
        available[index] = False

    return torch.tensor(
        selected, dtype=torch.long, device=candidate_embeddings.device
    )


def aggregate_multi_prototype_logits(
    image_features: Tensor,
    text_bank: Tensor,
    mask: Optional[Tensor] = None,
    subprototype_weight: float = 0.5,
    pool_temperature: float = 0.07,
    logit_scale: Tensor | float = 1.0,
) -> Tensor:
    """Compute base-anchored, normalized-LSE multi-prototype logits.

    Prototype index zero is the base class-name anchor.  Remaining valid
    prototypes are pooled with normalized Log-Sum-Exp.  Classes without a
    subprototype safely fall back to their base score.
    """
    if image_features.ndim != 2:
        raise ValueError("image_features must have shape [batch, dim]")
    if image_features.shape[1] != text_bank.shape[-1]:
        raise ValueError("image and text feature dimensions do not match")
    if not 0.0 <= subprototype_weight <= 1.0:
        raise ValueError("subprototype_weight must be in [0, 1]")
    if pool_temperature <= 0:
        raise ValueError("pool_temperature must be positive")

    mask = _validate_text_bank(text_bank, mask)
    if not torch.all(mask[:, 0]):
        raise ValueError("prototype index zero must be a valid base anchor")

    image_features = F.normalize(image_features, dim=-1)
    text_bank = F.normalize(text_bank, dim=-1)
    similarities = torch.einsum("bd,cpd->bcp", image_features, text_bank)
    base_scores = similarities[:, :, 0]

    if text_bank.shape[1] == 1 or subprototype_weight == 0:
        return base_scores * logit_scale

    sub_mask = mask[:, 1:]
    counts = sub_mask.sum(dim=1)
    scaled = similarities[:, :, 1:] / pool_temperature
    scaled = scaled.masked_fill(~sub_mask.unsqueeze(0), -torch.inf)
    # Avoid an all--inf logsumexp even on the unused branch of torch.where.
    # Nonempty sets retain their exact original arithmetic.
    scaled = torch.where(counts[None, :, None] > 0, scaled, torch.zeros_like(scaled))
    pooled = pool_temperature * (
        torch.logsumexp(scaled, dim=-1)
        - counts.clamp_min(1).to(scaled.dtype).log().unsqueeze(0)
    )
    pooled = torch.where(counts.unsqueeze(0) > 0, pooled, base_scores)
    scores = (
        (1.0 - subprototype_weight) * base_scores
        + subprototype_weight * pooled
    )
    # Base-only fallback must also be bit-exact under half precision.
    scores = torch.where(counts.unsqueeze(0) > 0, scores, base_scores)
    return scores * logit_scale


def set_rbf_relation(
    text_bank: Tensor,
    mask: Optional[Tensor] = None,
    sigma: float = 0.5,
) -> Tensor:
    """Build a normalized PSD class-relation matrix from prototype sets."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    mask = _validate_text_bank(text_bank, mask)
    text_bank = F.normalize(text_bank, dim=-1)

    cosine = torch.einsum("ipd,jqd->ijpq", text_bank, text_bank)
    squared_distance = (2.0 - 2.0 * cosine).clamp_min(0.0)
    kernel = torch.exp(-squared_distance / (2.0 * sigma**2))

    pair_mask = (
        mask[:, None, :, None] & mask[None, :, None, :]
    )
    pair_count = pair_mask.sum(dim=(-1, -2)).clamp_min(1)
    set_kernel = (kernel * pair_mask).sum(dim=(-1, -2))
    set_kernel = set_kernel / pair_count.to(set_kernel.dtype)

    diagonal = torch.diagonal(set_kernel)
    relation = set_kernel / torch.sqrt(
        diagonal[:, None] * diagonal[None, :]
    )
    relation = 0.5 * (relation + relation.T)
    relation.fill_diagonal_(1.0)
    return relation


def semantic_target_gram(relation: Tensor, rho: float) -> Tensor:
    """Mix pure orthogonality with a PSD semantic relation."""
    _validate_square_symmetric(relation, "relation")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must satisfy 0 <= rho < 1")
    identity = torch.eye(
        relation.shape[0], dtype=relation.dtype, device=relation.device
    )
    gram = (1.0 - rho) * identity + rho * relation
    return 0.5 * (gram + gram.T)


def _validate_square_symmetric(matrix: Tensor, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    tolerance = 1e-6 if matrix.dtype == torch.float32 else 1e-10
    if not torch.allclose(matrix, matrix.T, atol=tolerance, rtol=tolerance):
        raise ValueError(f"{name} must be symmetric")


def _cholesky_factor(matrix: Tensor, name: str) -> Tensor:
    _validate_square_symmetric(matrix, name)
    factor, info = torch.linalg.cholesky_ex(0.5 * (matrix + matrix.T))
    if int(info.max().item()) != 0:
        minimum = float(torch.linalg.eigvalsh(matrix).min().item())
        raise ValueError(f"{name} must be positive definite, min eigenvalue={minimum}")
    return factor


def initial_codebook(target_gram: Tensor, code_dim: int) -> Tensor:
    """Factor the first task Gram matrix into a fixed ambient space."""
    class_count = target_gram.shape[0]
    if code_dim < class_count:
        raise ValueError(
            f"code_dim={code_dim} is smaller than class_count={class_count}"
        )
    factor = _cholesky_factor(target_gram, "target_gram")
    if code_dim == class_count:
        return factor
    padding = target_gram.new_zeros(class_count, code_dim - class_count)
    return torch.cat((factor, padding), dim=1)


def reserve_basis(class_codes: Tensor) -> Tensor:
    """Return an orthonormal row basis of the class-code null space."""
    if class_codes.ndim != 2:
        raise ValueError("class_codes must have shape [num_classes, code_dim]")
    class_count, code_dim = class_codes.shape
    if class_count > code_dim:
        raise ValueError("class count cannot exceed code dimension")
    if class_count == 0:
        return torch.eye(
            code_dim, dtype=class_codes.dtype, device=class_codes.device
        )
    if int(torch.linalg.matrix_rank(class_codes).item()) != class_count:
        raise ValueError("class_codes must have full row rank")

    orthogonal, _ = torch.linalg.qr(class_codes.T, mode="complete")
    return orthogonal[:, class_count:].T.contiguous()


def extend_codebook(
    old_codes: Tensor,
    cross_gram: Tensor,
    new_gram: Tensor,
) -> Tensor:
    """Append new class codes while preserving every old value exactly.

    Args:
        old_codes: Previous codes Q_old with shape [C_old, D].
        cross_gram: Target old-new block B with shape [C_old, C_new].
        new_gram: Target new-new block C with shape [C_new, C_new].
    """
    if old_codes.ndim != 2:
        raise ValueError("old_codes must have shape [old_classes, code_dim]")
    old_count, code_dim = old_codes.shape
    if cross_gram.ndim != 2 or cross_gram.shape[0] != old_count:
        raise ValueError("cross_gram must have shape [old_classes, new_classes]")
    new_count = cross_gram.shape[1]
    if new_gram.shape != (new_count, new_count):
        raise ValueError("new_gram shape does not match cross_gram")
    _validate_square_symmetric(new_gram, "new_gram")
    if old_count + new_count > code_dim:
        raise ValueError("not enough unused code dimensions for the new classes")
    if new_count == 0:
        return old_codes.clone()
    if old_count == 0:
        return initial_codebook(new_gram, code_dim)

    old_gram = old_codes @ old_codes.T
    solved_cross = torch.linalg.solve(old_gram, cross_gram)
    parallel = solved_cross.T @ old_codes

    schur = new_gram - cross_gram.T @ solved_cross
    schur = 0.5 * (schur + schur.T)
    residual_factor = _cholesky_factor(schur, "Schur complement")

    available = reserve_basis(old_codes)
    if available.shape[0] < new_count:
        raise ValueError("old codebook null space is too small for the new classes")
    new_directions = available[:new_count]
    new_codes = parallel + residual_factor @ new_directions
    return torch.cat((old_codes, new_codes), dim=0)


def full_classifier_bank(class_codes: Tensor) -> Tensor:
    """Append unused orthogonal directions used as extra negative classes."""
    return torch.cat((class_codes, reserve_basis(class_codes)), dim=0)


def gram_error(codes: Tensor, target_gram: Tensor) -> Tensor:
    """Normalized Frobenius error used by tests and runtime diagnostics."""
    if codes.shape[0] != target_gram.shape[0]:
        raise ValueError("code and target class counts do not match")
    return torch.linalg.matrix_norm(codes @ codes.T - target_gram) / max(
        target_gram.shape[0], 1
    )
