"""Isolated attribute residual with no gradients into native TOPECL."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class OnlineAttributeResidual(nn.Module):
    def __init__(self, dim=512, hidden=16, radius=0.1, attribute_strength=0.25,
                 attention_temperature=0.07):
        super().__init__()
        self.down = nn.Linear(dim, hidden)
        self.up = nn.Linear(hidden, dim)
        self.radius = radius
        self.attribute_strength = float(attribute_strength)
        self.attention_temperature = float(attention_temperature)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, semantic, bank, scale, *, attributes, bank_mask=None,
                adaptive_pooling=True, conditioned_gate=False):
        # This branch observes the native representation, never changes it.
        h = F.normalize(semantic.detach().float(), dim=-1)
        bank = F.normalize(bank.detach().to(h.device).float(), dim=-1)
        if bank.ndim != 3 or bank.shape[1] != 5 or bank.shape[2] != h.shape[1]:
            raise ValueError("expected class-name plus four attributes per class")
        names, attrs = bank[:, 0], bank[:, 1:]
        if attributes:
            # Fixed sample-dependent attribute selection, not LSE class scoring.
            attention = torch.einsum("bd,ckd->bck", h, attrs) / self.attention_temperature
            if bank_mask is None:
                attr_mask = torch.ones(attrs.shape[:2], dtype=torch.bool, device=h.device)
            else:
                bank_mask = bank_mask.detach().to(device=h.device, dtype=torch.bool)
                if bank_mask.shape != bank.shape[:2] or not bool(bank_mask[:, 0].all()):
                    raise ValueError("bank mask must match the bank and retain every class-name slot")
                attr_mask = bank_mask[:, 1:]
            # Matched banks may intentionally retain fewer than four attributes.
            # Avoid NaNs for name-only classes and give padded slots exactly zero weight.
            if adaptive_pooling:
                safe_attention = attention.masked_fill(
                    ~attr_mask[None], torch.finfo(attention.dtype).min
                )
                weights = torch.softmax(safe_attention, dim=-1) * attr_mask[None]
            else:
                weights = attr_mask[None].expand(len(h), -1, -1).float()
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            orthogonal_parts = attrs - (attrs * names[:, None]).sum(-1, keepdim=True) * names[:, None]
            context = F.normalize(
                names[None]
                + self.attribute_strength * torch.einsum("bck,ckd->bcd", weights, orthogonal_parts),
                dim=-1,
            )
        else:
            context = names[None].expand(len(h), -1, -1)
        hidden = F.gelu(self.down(h))
        if conditioned_gate:
            if not attributes:
                raise ValueError("conditioned_gate requires attribute prototypes")
            class_weights = torch.softmax(
                torch.einsum("bd,bcd->bc", h, context) / self.attention_temperature,
                dim=-1,
            )
            attribute_summary = torch.einsum("bc,bcd->bd", class_weights, context)
            interaction = F.normalize(h * attribute_summary, dim=-1)
            channel_gate = 2.0 * torch.sigmoid(self.down(interaction))
            hidden = hidden * channel_gate
        r = self.radius * torch.tanh(self.up(hidden)) / math.sqrt(h.shape[-1])
        # Subtract the identical no-residual path. Zero initialization is exactly
        # zero in FP32, without re-normalizing h on only one side of the subtraction.
        difference = F.normalize(h + r, dim=-1) - F.normalize(h, dim=-1)
        scale = torch.as_tensor(scale, device=h.device).detach().float()
        return scale * torch.einsum("bd,bcd->bc", difference, context.detach())


def residual_objective(native_fusion, delta, labels, kl_weight=0.0, temperature=2.0,
                       focal_gamma=0.0, infonce_weight=0.0, infonce_temperature=0.07,
                       focal_mix_weight=1.0):
    teacher = native_fusion.detach().float()
    candidate = teacher + delta
    ce_per_sample = F.cross_entropy(candidate, labels, reduction="none")
    ce = ce_per_sample.mean()
    focal = (((1.0 - torch.exp(-ce_per_sample)) ** focal_gamma) * ce_per_sample).mean() if focal_gamma > 0 else ce
    if not 0.0 <= focal_mix_weight <= 1.0:
        raise ValueError("focal_mix_weight must be in [0, 1]")
    classification = (1.0 - focal_mix_weight) * ce + focal_mix_weight * focal if focal_gamma > 0 else ce
    kl = F.kl_div(F.log_softmax(candidate / temperature, -1),
                  F.softmax(teacher / temperature, -1), reduction="batchmean") * temperature**2
    infonce = F.cross_entropy(delta / infonce_temperature, labels)
    return classification + kl_weight * kl + infonce_weight * infonce, {
        "ce": ce.detach(), "focal": focal.detach(), "kl": kl.detach(), "infonce": infonce.detach()
    }
