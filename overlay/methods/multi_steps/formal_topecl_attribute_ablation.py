"""Official TOPECL evaluation with compact attribute-residual ablations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.cuda.amp import autocast

from backbone.semantic_codebook import mmr_select
from methods.multi_steps.formal_eval import topecl_all_head_logits
from methods.multi_steps.formal_topecl import Formal_TOPECL
from methods.multi_steps.formal_topecl_attribute_residual import (
    Formal_TOPECL_Attribute_Residual,
    _CallbackGradScaler,
    add_special_args,
)
from utils.formal_metrics import MultiHeadMetricTracker
from utils.online_attribute_residual import residual_objective
from utils.semantic_prompts import build_candidate_prompts


BRANCH_SPECS = {
    "name_only": {
        "attributes": False, "adaptive_pooling": False, "conditioned_gate": False,
    },
    "naive": {
        "attributes": True, "adaptive_pooling": False, "conditioned_gate": False,
    },
    "full": {
        "attributes": True, "adaptive_pooling": True, "conditioned_gate": False,
    },
    "gated": {
        "attributes": True, "adaptive_pooling": True, "conditioned_gate": True,
    },
}
ORTH_WEIGHT_GRID = {
    "000": 0.0,
    "025": 0.25,
    "050": 0.5,
    "075": 0.75,
    "100": 1.0,
}
BASE_TRACKED_HEADS = (
    "native_text",
    "name_only_text",
    "naive_text",
    "full_text",
    "gated_text",
    "native_fusion",
    "name_only_fusion",
    "naive_fusion",
    "full_fusion",
    "gated_fusion",
)


class Formal_TOPECL_Attribute_Ablation(Formal_TOPECL_Attribute_Residual):
    """Share one native TOPECL trajectory across four residual variants."""

    def __init__(self, logger, config):
        super().__init__(logger, config)
        template = self.attribute_branch
        del self.attribute_branch
        self.attribute_branches = nn.ModuleDict({
            name: copy.deepcopy(template) for name in BRANCH_SPECS
        })
        self._naive_attribute_bank = torch.empty(0, 5, 512)
        self._naive_attribute_bank_device = None
        self._branch_optimizers = {}
        self._branch_schedulers = {}
        self._branch_epoch_losses = {name: [] for name in BRANCH_SPECS}
        fixed_orth_weight = getattr(config, "attribute_fixed_orth_weight", None)
        self._orth_weight_grid = (
            {
                f"{int(round(100 * float(fixed_orth_weight))):03d}":
                float(fixed_orth_weight)
            }
            if fixed_orth_weight is not None
            else ORTH_WEIGHT_GRID
        )
        tracked_heads = BASE_TRACKED_HEADS + tuple(
            f"{branch}_orth_{tag}_fusion"
            for branch in ("native", *BRANCH_SPECS)
            for tag in self._orth_weight_grid
        )
        self._attribute_tracker = MultiHeadMetricTracker(
            self._nb_tasks, heads=tracked_heads
        )
        self._tracked_heads = tracked_heads
        self._attribute_manifest_path = Path(self._logdir) / (
            f"attribute_ablation_metrics_seed{self._seed}.json"
        )

    def _append_attribute_bank(self) -> None:
        network = self._network_module(self._network)
        first_new = self._known_classes
        base = network.proto_textemb[first_new:self._total_classes].detach().float()
        full_bank = base.new_zeros(len(self.new_class_names), 5, 512)
        naive_bank = base.new_zeros(len(self.new_class_names), 5, 512)
        new_mask = torch.ones(
            len(self.new_class_names), 5, dtype=torch.bool, device=base.device
        )
        full_bank[:, 0] = base
        naive_bank[:, 0] = base

        selected_records = []
        self.text_enc.cuda()
        try:
            for local_index, class_name in enumerate(self.new_class_names):
                prompts = build_candidate_prompts(
                    class_name,
                    self.description_bank,
                    require_descriptions=self._config.attribute_require_descriptions,
                )
                if len(prompts) < 4:
                    raise ValueError(f"'{class_name}' has fewer than four descriptions")
                candidates = self._encode_text(prompts)
                full_indices = mmr_select(
                    candidates,
                    base[local_index],
                    4,
                    relevance_weight=self._config.attribute_mmr_lambda,
                ).cpu().tolist()
                naive_indices = list(range(4))
                full_bank[local_index, 1:] = candidates[full_indices]
                naive_bank[local_index, 1:] = candidates[naive_indices]
                selected_records.append({
                    "task": int(self._cur_task),
                    "class_id": int(first_new + local_index),
                    "class_name": class_name,
                    "naive_indices": naive_indices,
                    "full_indices": full_indices,
                    "naive_prompts": [prompts[index] for index in naive_indices],
                    "full_prompts": [prompts[index] for index in full_indices],
                })
        finally:
            self.text_enc.cpu()

        self._attribute_bank = torch.cat(
            (self._attribute_bank, full_bank.cpu()), dim=0
        )
        self._naive_attribute_bank = torch.cat(
            (self._naive_attribute_bank, naive_bank.cpu()), dim=0
        )
        self._attribute_mask = torch.cat(
            (self._attribute_mask, new_mask.cpu()), dim=0
        )
        self._attribute_selection.extend(selected_records)
        self._attribute_bank_device = self._attribute_bank.cuda()
        self._naive_attribute_bank_device = self._naive_attribute_bank.cuda()
        self._attribute_mask_device = self._attribute_mask.cuda()

        selection_path = Path(self._logdir) / (
            f"attribute_ablation_selection_seed{self._seed}.json"
        )
        selection_path.write_text(json.dumps({
            "description_path": str(
                Path(self._config.attribute_description_path).resolve()
            ),
            "mmr_lambda": float(self._config.attribute_mmr_lambda),
            "attributes_per_class": 4,
            "classes": self._attribute_selection,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        self._logger.info(
            "Frozen full and naive attribute banks ready: task={}, classes={}".format(
                self._cur_task, self._total_classes
            )
        )

    def _start_branch_phase(self, epochs: int, phase: str) -> None:
        self.attribute_branches.train()
        self._branch_optimizers = {
            name: optim.AdamW(
                branch.parameters(),
                lr=self._config.attribute_branch_lrate,
                weight_decay=0.0,
            )
            for name, branch in self.attribute_branches.items()
        }
        self._branch_schedulers = {
            name: optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            for name, optimizer in self._branch_optimizers.items()
        }
        self._branch_active = True
        self._branch_phase = phase
        self._branch_epoch_losses = {name: [] for name in BRANCH_SPECS}

    def _branch_inputs(self, name, count):
        spec = BRANCH_SPECS[name]
        bank = (
            self._naive_attribute_bank_device
            if name == "naive"
            else self._attribute_bank_device
        )
        return (
            bank[:count], spec["attributes"], spec["adaptive_pooling"],
            spec["conditioned_gate"],
        )

    def _branch_update(self, succeeded: bool) -> None:
        if not succeeded or not self._branch_active:
            self._pending_real = None
            self._pending_pseudo = None
            return
        batches = [
            batch for batch in (self._pending_real, self._pending_pseudo) if batch
        ]
        if not batches:
            return
        with torch.amp.autocast("cuda", enabled=False):
            for name, branch in self.attribute_branches.items():
                optimizer = self._branch_optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for semantic, text, orth, labels, _bank, mask, scale in batches:
                    count = min(self._total_classes, text.shape[1], orth.shape[1])
                    bank, attributes, adaptive_pooling, conditioned_gate = self._branch_inputs(
                        name, count
                    )
                    delta = branch(
                        semantic,
                        bank,
                        scale,
                        attributes=attributes,
                        bank_mask=mask[:count],
                        adaptive_pooling=adaptive_pooling,
                        conditioned_gate=conditioned_gate,
                    ) * self._config.attribute_inference_scale
                    objective, _ = residual_objective(
                        text[:, :count].float().detach(),
                        delta,
                        labels,
                        focal_gamma=self._config.attribute_focal_gamma,
                    )
                    losses.append(objective)
                total = sum(losses)
                total.backward()
                optimizer.step()
                self._branch_epoch_losses[name].append(
                    float(total.detach().item())
                )
        self._branch_steps += 1
        self._pending_real = None
        self._pending_pseudo = None

    def _train_model(
        self, model, train_loader, test_loader, optimizer, scheduler,
        task_id=None, epochs=100, note="", true_data=True, sync_data=True,
    ):
        phase = "stage1" if true_data else "stage2"
        self._start_branch_phase(epochs, phase)
        self.scaler = _CallbackGradScaler(self._branch_update)
        task_begin = sum(self._increment_steps[:task_id])
        task_end = task_begin + self._increment_steps[task_id]
        for epoch in range(epochs):
            model, _train_acc, train_losses = self._epoch_train(
                model, train_loader, optimizer, scheduler,
                task_begin=task_begin, task_end=task_end, task_id=task_id,
                true_data=true_data, sync_data=sync_data,
            )
            for branch_scheduler in self._branch_schedulers.values():
                branch_scheduler.step()
            means = {
                name: float(np.mean(values))
                for name, values in self._branch_epoch_losses.items()
            }
            self._branch_epoch_records.append({
                "task": int(task_id),
                "phase": phase,
                "epoch": int(epoch + 1),
                "native_loss": float(train_losses[1]),
                "branch_losses": means,
                "steps": int(self._branch_steps),
            })
            self._branch_epoch_losses = {name: [] for name in BRANCH_SPECS}
            self._logger.info(
                "Task {} {} epoch {}/{}: native={:.4f}, name={:.4f}, "
                "naive={:.4f}, full={:.4f}, gated={:.4f}".format(
                    task_id, phase, epoch + 1, epochs,
                    float(train_losses[1]), means["name_only"],
                    means["naive"], means["full"], means["gated"],
                )
            )
        self._branch_active = False
        self._branch_phase = None
        return model

    def _collect_ablation_predictions(self):
        network = self._network_module(self._network)
        network.eval()
        self.attribute_branches.eval()
        predictions = {head: [] for head in self._tracked_heads}
        targets_all = []
        for _, inputs, targets in self._test_loader:
            inputs = inputs.cuda()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                logits, semantic = topecl_all_head_logits(
                    network, inputs, self._formal_fusion_weight,
                    return_semantic_features=True,
                )
            native_text = logits["text"][:, :self._total_classes].float()
            native_orth = logits["orth"][:, :self._total_classes].float()
            native_fusion = logits["fusion"][:, :self._total_classes].float()
            outputs = {
                "native_text": native_text,
                "native_fusion": native_fusion,
            }
            for tag, weight in self._orth_weight_grid.items():
                outputs[f"native_orth_{tag}_fusion"] = (
                    native_text + weight * native_orth
                )
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                for name, branch in self.attribute_branches.items():
                    bank, attributes, adaptive_pooling, conditioned_gate = self._branch_inputs(
                        name, self._total_classes
                    )
                    delta = branch(
                        semantic,
                        bank,
                        network.logit_scale_textemb.exp(),
                        attributes=attributes,
                        bank_mask=self._attribute_mask_device[:self._total_classes],
                        adaptive_pooling=adaptive_pooling,
                        conditioned_gate=conditioned_gate,
                    ) * self._config.attribute_inference_scale
                    outputs[f"{name}_text"] = native_text + delta
                    outputs[f"{name}_fusion"] = native_fusion + delta
                    for tag, weight in self._orth_weight_grid.items():
                        outputs[f"{name}_orth_{tag}_fusion"] = (
                            native_text + weight * native_orth + delta
                        )
            for head, values in outputs.items():
                predictions[head].append(values.argmax(1).cpu().numpy())
            targets_all.append(np.asarray(targets))
        return (
            {head: np.concatenate(values) for head, values in predictions.items()},
            np.concatenate(targets_all),
        )

    def eval_task(self):
        Formal_TOPECL.eval_task(self)
        with self._preserve_rng():
            predictions, targets = self._collect_ablation_predictions()
        native_path = Path(self._logdir) / (
            f"formal_predictions_seed{self._seed}_task{self._cur_task}.npz"
        )
        with np.load(native_path) as native:
            if not np.array_equal(targets, native["targets"]):
                raise RuntimeError("attribute and native evaluation orders differ")
            if not np.array_equal(
                predictions["native_text"], native["text_predictions"]
            ):
                raise RuntimeError("native Text predictions differ")
            if not np.array_equal(
                predictions["native_fusion"], native["fusion_predictions"]
            ):
                raise RuntimeError("native Fusion predictions differ")

        stage = self._attribute_tracker.update(
            self._cur_task, predictions, targets, self._total_classes,
            self._increment_steps,
        )
        old_boundary = sum(self._increment_steps[:self._cur_task])
        masks = {
            "all": np.ones_like(targets, dtype=bool),
            "old": targets < old_boundary,
            "new": targets >= old_boundary,
        }
        relative = {}
        for name in BRANCH_SPECS:
            relative[name] = {
                head: {
                    split: self._change_counts(
                        predictions[f"native_{head}"],
                        predictions[f"{name}_{head}"],
                        targets,
                        mask,
                    )
                    for split, mask in masks.items()
                }
                for head in ("text", "fusion")
            }
        self._attribute_stages.append({
            "task_id": int(self._cur_task),
            "known_classes": int(self._total_classes),
            "heads": stage,
            "relative_to_native": relative,
        })
        summary = self._attribute_tracker.summary(self._cur_task)
        payload = {
            "run": {
                "dataset": self._dataset,
                "seed": int(self._seed),
                "increments": list(self._increment_steps),
                "groups": {
                    "native": "unchanged TOPECL",
                    "name_only": "equal-capacity class-name residual",
                    "naive": "first four attributes with uniform pooling",
                    "full": "MMR-selected attributes with sample-adaptive pooling",
                    "gated": "full attributes with attribute-conditioned channel gating",
                },
                "attributes_per_class": 4,
                "mmr_lambda": float(self._config.attribute_mmr_lambda),
                "attribute_strength": float(self._config.attribute_strength),
                "focal_gamma": float(self._config.attribute_focal_gamma),
                "inference_scale": float(self._config.attribute_inference_scale),
                "orth_weight_grid": self._orth_weight_grid,
                "branch_parameters": {
                    name: int(sum(p.numel() for p in branch.parameters()))
                    for name, branch in self.attribute_branches.items()
                },
                "branch_steps": int(self._branch_steps),
            },
            "stages": self._attribute_stages,
            "summary": summary,
            "complete": self._cur_task == self._nb_tasks - 1,
            "epoch_records": self._branch_epoch_records,
        }
        self._attribute_manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        prediction_path = Path(self._logdir) / (
            f"attribute_ablation_predictions_seed{self._seed}_task{self._cur_task}.npz"
        )
        np.savez_compressed(prediction_path, targets=targets, **predictions)
        self._logger.info(
            "ATTRIBUTE ABLATION fusion: native={:.2f}, name={:.2f}, "
            "naive={:.2f}, full={:.2f}, gated={:.2f}".format(
                stage["native_fusion"]["overall_accuracy"],
                stage["name_only_fusion"]["overall_accuracy"],
                stage["naive_fusion"]["overall_accuracy"],
                stage["full_fusion"]["overall_accuracy"],
                stage["gated_fusion"]["overall_accuracy"],
            )
        )
