"""Official TOPECL evaluation with the frozen attribute-residual branch."""

from __future__ import annotations

from argparse import ArgumentParser
from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast

from backbone.semantic_codebook import mmr_select
from methods.multi_steps.formal_eval import topecl_all_head_logits
from methods.multi_steps.formal_topecl import (
    Formal_TOPECL,
    add_special_args as add_formal_args,
)
from utils.formal_metrics import MultiHeadMetricTracker
from utils.online_attribute_residual import OnlineAttributeResidual, residual_objective
from utils.semantic_prompts import build_candidate_prompts, load_description_bank


TRACKED_HEADS = (
    "native_text",
    "attribute_text",
    "native_fusion",
    "attribute_fusion",
)


def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser = add_formal_args(parser)
    parser.add_argument("--attribute_description_path", type=str, help="")
    parser.add_argument("--attribute_require_descriptions", action="store_true", default=None)
    parser.add_argument("--attribute_num_subprototypes", type=int, help="")
    parser.add_argument("--attribute_mmr_lambda", type=float, help="")
    parser.add_argument("--attribute_residual_hidden", type=int, help="")
    parser.add_argument("--attribute_residual_radius", type=float, help="")
    parser.add_argument("--attribute_strength", type=float, help="")
    parser.add_argument("--attribute_initialization_seed", type=int, help="")
    parser.add_argument("--attribute_focal_gamma", type=float, help="")
    parser.add_argument("--attribute_inference_scale", type=float, help="")
    parser.add_argument("--attribute_branch_lrate", type=float, help="")
    parser.add_argument("--attribute_fixed_orth_weight", type=float, help="")
    return parser


class _CallbackGradScaler(GradScaler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def step(self, optimizer, *args, **kwargs):
        result = super().step(optimizer, *args, **kwargs)
        state = self._per_optimizer_states.get(id(optimizer), {})
        found_inf = state.get("found_inf_per_device", {})
        succeeded = bool(found_inf) and all(
            float(value.item()) == 0.0 for value in found_inf.values()
        )
        self.callback(succeeded)
        return result


class Formal_TOPECL_Attribute_Residual(Formal_TOPECL):
    """Train one isolated attribute residual beside an unchanged TOPECL run."""

    def __init__(self, logger, config):
        super().__init__(logger, config)
        if config.attribute_num_subprototypes != 4:
            raise ValueError("the locked method requires four attributes per class")
        if not 0.0 <= config.attribute_mmr_lambda <= 1.0:
            raise ValueError("attribute_mmr_lambda must be in [0, 1]")
        self.description_bank = load_description_bank(config.attribute_description_path)
        cuda_device = torch.cuda.current_device()
        with torch.random.fork_rng(devices=[cuda_device]):
            torch.manual_seed(config.attribute_initialization_seed)
            torch.cuda.manual_seed_all(config.attribute_initialization_seed)
            self.attribute_branch = OnlineAttributeResidual(
                dim=512,
                hidden=config.attribute_residual_hidden,
                radius=config.attribute_residual_radius,
                attribute_strength=(
                    0.25
                    if getattr(config, "attribute_strength", None) is None
                    else config.attribute_strength
                ),
            ).cuda()
        self._attribute_bank = torch.empty(0, 5, 512)
        self._attribute_mask = torch.empty(0, 5, dtype=torch.bool)
        self._attribute_bank_device = None
        self._attribute_mask_device = None
        self._attribute_selection = []
        self._branch_optimizer = None
        self._branch_scheduler = None
        self._pending_real = None
        self._pending_pseudo = None
        self._branch_active = False
        self._branch_phase = None
        self._branch_steps = 0
        self._branch_epoch_records = []
        self._attribute_tracker = MultiHeadMetricTracker(
            self._nb_tasks, heads=TRACKED_HEADS
        )
        self._attribute_manifest_path = Path(self._logdir) / (
            f"attribute_residual_metrics_seed{self._seed}.json"
        )
        self._attribute_stages = []

    @staticmethod
    def _network_module(model):
        return model.module if isinstance(model, nn.DataParallel) else model

    def _encode_text(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.text_tokenizer(prompts).cuda()
        with torch.no_grad(), autocast():
            features = self.text_enc(tokens)
        return features.detach().float()

    def _append_attribute_bank(self) -> None:
        network = self._network_module(self._network)
        first_new = self._known_classes
        base = network.proto_textemb[first_new:self._total_classes].detach().float()
        new_bank = base.new_zeros(len(self.new_class_names), 5, 512)
        new_mask = torch.zeros(
            len(self.new_class_names), 5, dtype=torch.bool, device=base.device
        )
        new_bank[:, 0] = base
        new_mask[:, 0] = True

        selected_records = []
        self.text_enc.cuda()
        try:
            for local_index, class_name in enumerate(self.new_class_names):
                prompts = build_candidate_prompts(
                    class_name,
                    self.description_bank,
                    require_descriptions=self._config.attribute_require_descriptions,
                )
                candidates = self._encode_text(prompts)
                indices = mmr_select(
                    candidates,
                    base[local_index],
                    4,
                    relevance_weight=self._config.attribute_mmr_lambda,
                )
                new_bank[local_index, 1:] = candidates[indices]
                new_mask[local_index, 1:] = True
                selected_records.append({
                    "task": int(self._cur_task),
                    "class_id": int(first_new + local_index),
                    "class_name": class_name,
                    "candidate_indices": indices.cpu().tolist(),
                    "prompts": [prompts[index] for index in indices.cpu().tolist()],
                })
        finally:
            self.text_enc.cpu()

        self._attribute_bank = torch.cat(
            (self._attribute_bank, new_bank.cpu()), dim=0
        )
        self._attribute_mask = torch.cat(
            (self._attribute_mask, new_mask.cpu()), dim=0
        )
        self._attribute_selection.extend(selected_records)
        self._attribute_bank_device = self._attribute_bank.cuda()
        self._attribute_mask_device = self._attribute_mask.cuda()
        selection_path = Path(self._logdir) / (
            f"attribute_selection_seed{self._seed}.json"
        )
        selection_path.write_text(json.dumps({
            "description_path": str(Path(self._config.attribute_description_path).resolve()),
            "mmr_lambda": float(self._config.attribute_mmr_lambda),
            "attributes_per_class": 4,
            "classes": self._attribute_selection,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        self._logger.info(
            "Frozen attribute bank ready: task={}, classes={}, slots=5".format(
                self._cur_task, self._total_classes
            )
        )

    def prepare_model(self, checkpoint=None):
        super().prepare_model(checkpoint)
        self._append_attribute_bank()
        network = self._network_module(self._network)
        expected = network.proto_textemb[:self._total_classes].detach().float().cpu()
        if not torch.allclose(self._attribute_bank[:, 0], expected, atol=1e-5, rtol=1e-5):
            raise RuntimeError("attribute class-name anchors differ from native TOPECL")

    def _start_branch_phase(self, epochs: int, phase: str) -> None:
        self.attribute_branch.train()
        self._branch_optimizer = optim.AdamW(
            self.attribute_branch.parameters(),
            lr=self._config.attribute_branch_lrate,
            weight_decay=0.0,
        )
        self._branch_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self._branch_optimizer, T_max=epochs
        )
        self._branch_active = True
        self._branch_phase = phase
        self._branch_epoch_loss = []

    def _branch_update(self, succeeded: bool) -> None:
        if not succeeded or not self._branch_active:
            self._pending_real = None
            self._pending_pseudo = None
            return
        batches = [batch for batch in (self._pending_real, self._pending_pseudo) if batch]
        if not batches:
            return
        self._branch_optimizer.zero_grad(set_to_none=True)
        losses = []
        with torch.amp.autocast("cuda", enabled=False):
            for semantic, text, orth, labels, bank, mask, scale in batches:
                delta = self.attribute_branch(
                    semantic, bank, scale, attributes=True, bank_mask=mask
                ) * self._config.attribute_inference_scale
                objective, _ = residual_objective(
                    text.float().detach(),
                    delta,
                    labels,
                    focal_gamma=self._config.attribute_focal_gamma,
                )
                losses.append(objective)
        total = sum(losses)
        total.backward()
        self._branch_optimizer.step()
        self._branch_steps += 1
        self._branch_epoch_loss.append(float(total.detach().item()))
        self._pending_real = None
        self._pending_pseudo = None

    def _forward_and_compute_loss(
        self, model, inputs, targets, task_begin, task_end, input_feature=False
    ):
        if input_feature:
            self._pending_pseudo = None
        else:
            self._pending_real = None
        if inputs.shape[0] == 0:
            return super()._forward_and_compute_loss(
                model, inputs, targets, task_begin, task_end, input_feature
            )

        network = self._network_module(model)
        captured = {}

        def capture(_module, _inputs, output):
            captured["semantic"] = output.detach()

        hook = network.adapter1.register_forward_hook(capture)
        try:
            result = super()._forward_and_compute_loss(
                model, inputs, targets, task_begin, task_end, input_feature
            )
        finally:
            hook.remove()
        _, _, _, text, orth, _ = result
        if text is not None and orth is not None:
            count = min(self._total_classes, text.shape[1], orth.shape[1])
            pending = (
                captured["semantic"].float(),
                text[:, :count].detach(),
                orth[:, :count].detach(),
                targets.detach(),
                self._attribute_bank_device[:count],
                self._attribute_mask_device[:count],
                network.logit_scale_textemb.exp().detach(),
            )
            if input_feature:
                self._pending_pseudo = pending
            else:
                self._pending_real = pending
        return result

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
            model, train_acc, train_losses = self._epoch_train(
                model, train_loader, optimizer, scheduler,
                task_begin=task_begin, task_end=task_end, task_id=task_id,
                true_data=true_data, sync_data=sync_data,
            )
            self._branch_scheduler.step()
            mean_loss = float(np.mean(self._branch_epoch_loss))
            self._branch_epoch_records.append({
                "task": int(task_id),
                "phase": phase,
                "epoch": int(epoch + 1),
                "mean_loss": mean_loss,
                "steps": int(self._branch_steps),
            })
            self._branch_epoch_loss = []
            self._logger.info(
                "Task {} {} epoch {}/{}: native_loss={:.4f}, attribute_focal={:.4f}".format(
                    task_id, phase, epoch + 1, epochs, float(train_losses[1]), mean_loss
                )
            )
        self._branch_active = False
        self._branch_phase = None
        return model

    def _collect_attribute_predictions(self):
        network = self._network_module(self._network)
        network.eval()
        self.attribute_branch.eval()
        predictions = {head: [] for head in TRACKED_HEADS}
        targets_all = []
        for _, inputs, targets in self._test_loader:
            inputs = inputs.cuda()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                logits, semantic = topecl_all_head_logits(
                    network, inputs, self._formal_fusion_weight,
                    return_semantic_features=True,
                )
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                delta = self.attribute_branch(
                    semantic,
                    self._attribute_bank_device[:self._total_classes],
                    network.logit_scale_textemb.exp(),
                    attributes=True,
                    bank_mask=self._attribute_mask_device[:self._total_classes],
                ) * self._config.attribute_inference_scale
                native_text = logits["text"][:, :self._total_classes].float()
                native_fusion = logits["fusion"][:, :self._total_classes].float()
                outputs = {
                    "native_text": native_text,
                    "attribute_text": native_text + delta,
                    "native_fusion": native_fusion,
                    "attribute_fusion": native_fusion + delta,
                }
            for head, values in outputs.items():
                predictions[head].append(values.argmax(1).cpu().numpy())
            targets_all.append(np.asarray(targets))
        return (
            {head: np.concatenate(values) for head, values in predictions.items()},
            np.concatenate(targets_all),
        )

    @contextmanager
    def _preserve_rng(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all()
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            torch.cuda.set_rng_state_all(cuda_state)

    @staticmethod
    def _change_counts(native, candidate, targets, mask):
        native_correct = native[mask] == targets[mask]
        candidate_correct = candidate[mask] == targets[mask]
        return {
            "samples": int(mask.sum()),
            "corrected": int(np.logical_and(~native_correct, candidate_correct).sum()),
            "harmed": int(np.logical_and(native_correct, ~candidate_correct).sum()),
        }

    def eval_task(self):
        super().eval_task()
        with self._preserve_rng():
            predictions, targets = self._collect_attribute_predictions()
        native_path = Path(self._logdir) / (
            f"formal_predictions_seed{self._seed}_task{self._cur_task}.npz"
        )
        with np.load(native_path) as native:
            if not np.array_equal(targets, native["targets"]):
                raise RuntimeError("attribute and native evaluation orders differ")
            if not np.array_equal(predictions["native_text"], native["text_predictions"]):
                raise RuntimeError("native Text predictions differ across evaluation paths")
            if not np.array_equal(predictions["native_fusion"], native["fusion_predictions"]):
                raise RuntimeError("native Fusion predictions differ across evaluation paths")

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
        relative = {
            "text": {
                name: self._change_counts(
                    predictions["native_text"], predictions["attribute_text"],
                    targets, mask,
                ) for name, mask in masks.items()
            },
            "fusion": {
                name: self._change_counts(
                    predictions["native_fusion"], predictions["attribute_fusion"],
                    targets, mask,
                ) for name, mask in masks.items()
            },
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
                "description_path": str(Path(self._config.attribute_description_path).resolve()),
                "attributes_per_class": 4,
                "mmr_lambda": float(self._config.attribute_mmr_lambda),
                "focal_gamma": float(self._config.attribute_focal_gamma),
                "inference_scale": float(self._config.attribute_inference_scale),
                "branch_parameters": int(sum(p.numel() for p in self.attribute_branch.parameters())),
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
            f"attribute_residual_predictions_seed{self._seed}_task{self._cur_task}.npz"
        )
        np.savez_compressed(prediction_path, targets=targets, **predictions)
        self._logger.info(
            "ATTRIBUTE RESIDUAL: text={:.2f} ({:+.2f}), fusion={:.2f} ({:+.2f})".format(
                stage["attribute_text"]["overall_accuracy"],
                stage["attribute_text"]["overall_accuracy"] - stage["native_text"]["overall_accuracy"],
                stage["attribute_fusion"]["overall_accuracy"],
                stage["attribute_fusion"]["overall_accuracy"] - stage["native_fusion"]["overall_accuracy"],
            )
        )
