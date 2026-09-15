"""Shared formal evaluation and checkpointing for TOPECL-family methods."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn
from torch.nn import functional as F
from scipy.stats import spearmanr

from backbone.semantic_codebook import aggregate_multi_prototype_logits, gram_error
from backbone.topecl_net import TOPECLNet
from utils.formal_metrics import HEAD_NAMES, MultiHeadMetricTracker
from utils.semantic_prompts import description_bank_sha256


def add_formal_eval_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--formal_fusion_weight", type=float, help="")
    parser.add_argument("--formal_checkpoint_policy", type=str, help="")
    parser.add_argument("--formal_require_clean_git", action="store_true", default=None)
    parser.add_argument(
        "--formal_require_validation_for_screening",
        action="store_true",
        default=None,
        help="Reject prefix screening runs unless task-end metrics use validation.",
    )
    parser.add_argument(
        "--screening_skip_epoch_diagnostics",
        action="store_true",
        default=None,
        help=(
            "Skip the two redundant end-of-epoch head evaluations used only for "
            "training logs. Formal three-head evaluation after every task is kept."
        ),
    )
    return parser


def topecl_all_head_logits(
    network: nn.Module,
    inputs: torch.Tensor,
    fusion_weight: float,
    return_semantic_features: bool = False,
):
    """Compute all classifier heads while sharing one CLIP image encoding."""
    with torch.no_grad():
        image_features = network.feature_extractor(inputs)
    semantic_features = network.adapter1(image_features)
    text_logits = network.linear_textemb(semantic_features)
    orth_features = network.adapter2(semantic_features)
    orth_logits = network.linear_orth(orth_features)[:, : network.knowncls]
    logits = {
        "text": text_logits,
        "orth": orth_logits,
        "fusion": text_logits + fusion_weight * orth_logits,
    }
    if return_semantic_features:
        return logits, semantic_features
    return logits


def _safe_pair_correlations(relation: np.ndarray, confusion: np.ndarray) -> dict:
    if relation.size == 0 or np.std(relation) == 0 or np.std(confusion) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(relation, confusion)[0, 1]),
        "spearman": float(spearmanr(relation, confusion).statistic),
    }


def relation_confusion_diagnostics(
    relation: torch.Tensor,
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    rho: float,
    top_k: int = 50,
) -> dict:
    """Compare semantic proximity with the model's observed class confusion."""
    class_count = relation.shape[0]
    matrix = np.zeros((class_count, class_count), dtype=np.float64)
    np.add.at(matrix, (targets, predictions), 1.0)
    counts = np.bincount(targets, minlength=class_count).clip(min=1)
    matrix = matrix / counts[:, None]
    symmetric = 0.5 * (matrix + matrix.T)
    upper = np.triu_indices(class_count, k=1)
    relation_values = relation.detach().float().cpu().numpy()[upper]
    confusion_values = symmetric[upper]
    order = np.argsort(-relation_values)
    pair_count = len(order)
    keep = min(top_k, max(1, math.ceil(0.1 * pair_count))) if pair_count else 0
    relation_top = set(order[:keep].tolist())
    confusion_top = set(np.argsort(-confusion_values)[:keep].tolist())
    top_pairs = []
    for flat_index in order[: min(20, pair_count)]:
        class_a = int(upper[0][flat_index])
        class_b = int(upper[1][flat_index])
        top_pairs.append({
            "class_a": class_names[class_a],
            "class_b": class_names[class_b],
            "relation": float(relation_values[flat_index]),
            "target_code_cosine": float(rho * relation_values[flat_index]),
            "symmetric_confusion_percent": float(
                100.0 * confusion_values[flat_index]
            ),
        })
    return {
        "correlation": _safe_pair_correlations(
            relation_values, confusion_values
        ),
        "top{}_overlap".format(keep): len(relation_top & confusion_top),
        "random_overlap_expectation": (
            float(keep * keep / pair_count) if pair_count else 0.0
        ),
        "top_pairs": top_pairs,
    }


def should_save_checkpoint(policy: str, seed: int, task: int, nb_tasks: int) -> bool:
    if policy == "none":
        return False
    if policy == "all":
        return True
    if policy == "final":
        return task == nb_tasks - 1
    if policy == "seed42_all_else_final":
        return seed == 42 or task == nb_tasks - 1
    raise ValueError(
        "formal_checkpoint_policy must be one of none, all, final, "
        "seed42_all_else_final"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_ready
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_state(repo_root: Path) -> dict:
    def run(*args: str, binary: bool = False):
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=not binary,
        ).stdout

    try:
        commit = run("rev-parse", "HEAD").strip()
        branch = run("branch", "--show-current").strip()
        status_lines = [
            line for line in run("status", "--porcelain", "--untracked-files=normal").splitlines()
            if line.strip()
        ]
        tracked_diff = run("diff", "--binary", "HEAD", binary=True)
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(status_lines),
            "changed_file_count": len(status_lines),
            "changed_files": status_lines,
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _artifact_hashes(config) -> dict:
    artifacts = {}
    selection_path = getattr(config, "semantic_selection_path", None)
    if selection_path:
        artifacts["visual_selection"] = {"path": str(Path(selection_path).resolve()),
                                         "sha256": _sha256_file(Path(selection_path))}
    config_path = getattr(config, "config", None)
    if config_path and Path(config_path).is_file():
        artifacts["config_yaml"] = {
            "path": str(Path(config_path).resolve()),
            "sha256": _sha256_file(Path(config_path)),
        }
    description_path = getattr(config, "semantic_description_path", None)
    if description_path and Path(description_path).is_file():
        artifacts["description_bank"] = {
            "path": str(Path(description_path).resolve()),
            "canonical_sha256": description_bank_sha256(description_path),
        }
    dataset_path = os.environ.get("CIFAR100_HF_ARCHIVE")
    if dataset_path and Path(dataset_path).is_file():
        artifacts["dataset_archive"] = {
            "path": str(Path(dataset_path).resolve()),
            "sha256": _sha256_file(Path(dataset_path)),
        }
    clip_root = os.environ.get("CLIP_DOWNLOAD_ROOT")
    if clip_root:
        clip_path = Path(clip_root) / "ViT-B-16.pt"
        if clip_path.is_file():
            artifacts["clip_checkpoint"] = {
                "path": str(clip_path.resolve()),
                "sha256": _sha256_file(clip_path),
            }
    return artifacts


def _open_set_metrics(confidence: np.ndarray, known_mask: np.ndarray) -> dict | None:
    binary = known_mask.astype(np.int64)
    if np.unique(binary).size < 2:
        return None
    fpr, tpr, _ = roc_curve(binary, confidence)
    valid = np.flatnonzero(tpr >= 0.95)
    return {
        "auc": float(np.around(roc_auc_score(binary, confidence) * 100.0, 2)),
        "fpr95": float(np.around(fpr[valid[0]] * 100.0, 2)) if valid.size else None,
        "average_precision": float(
            np.around(average_precision_score(binary, confidence) * 100.0, 2)
        ),
    }


class TOPECLFormalEvalMixin:
    """Mixin providing one-pass three-head evaluation and robust manifests."""

    def _train_model(self, *args, **kwargs):
        """Optionally omit screening-only diagnostics without changing training.

        Public TOPECL evaluates LCLS and ECLS at the end of both training phases.
        The formal wrapper immediately performs a shared three-head evaluation
        after the task, so those four full test-set passes are redundant in a
        time-limited screening run.  The flag is deliberately opt-in and is not
        enabled by any formal paper configuration.
        """
        skip = bool(
            getattr(self._config, "screening_skip_epoch_diagnostics", False)
        )
        previous = getattr(self, "_screening_skip_epoch_diagnostics_active", False)
        self._screening_skip_epoch_diagnostics_active = skip
        if skip:
            self._logger.info(
                "Screening mode: redundant end-of-epoch test passes are disabled; "
                "formal Text/Orth/Fusion evaluation after the task remains enabled."
            )
        try:
            return super()._train_model(*args, **kwargs)
        finally:
            self._screening_skip_epoch_diagnostics_active = previous

    def _epoch_test(self, *args, **kwargs):
        if getattr(self, "_screening_skip_epoch_diagnostics_active", False):
            if kwargs.get("ret_pred_target", False):
                raise RuntimeError(
                    "screening_skip_epoch_diagnostics cannot replace prediction "
                    "collection"
                )
            if kwargs.get("ret_task_acc", False):
                return float("nan"), float("nan")
            return float("nan")
        return super()._epoch_test(*args, **kwargs)

    def _init_formal_evaluation(self) -> None:
        fusion_weight = getattr(self._config, "formal_fusion_weight", None)
        if fusion_weight is None:
            fusion_weight = getattr(self._config, "semantic_fusion_weight", 1.0)
        self._formal_fusion_weight = float(fusion_weight)
        self._formal_checkpoint_policy = (
            getattr(self._config, "formal_checkpoint_policy", None) or "none"
        )
        # Validate at startup even when save_models is false.
        should_save_checkpoint(
            self._formal_checkpoint_policy,
            int(self._seed),
            0,
            self._nb_tasks,
        )
        self._formal_tracker = MultiHeadMetricTracker(self._nb_tasks)
        self._formal_manifest_path = Path(self._logdir) / (
            f"formal_metrics_seed{self._seed}.json"
        )
        self._formal_manifest_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self._config.get_parameters_dict()
        repo_root = Path(__file__).resolve().parents[2]
        git = _git_state(repo_root)
        if (
            getattr(self._config, "formal_require_clean_git", False)
            and git.get("dirty") is not False
        ):
            raise RuntimeError(
                "formal_require_clean_git is enabled but a clean git commit could "
                "not be verified; commit the experiment code before a formal run"
            )
        evaluation_source = (
            getattr(self._config, "evaluation_source", None) or "test"
        )
        if (
            getattr(
                self._config,
                "formal_require_validation_for_screening",
                False,
            )
            and getattr(self._config, "screening_task_limit", None)
            and evaluation_source != "validation"
        ):
            raise RuntimeError(
                "screening protocol requires evaluation_source=validation; "
                "the test split must remain untouched until configuration freeze"
            )
        self._formal_manifest = {
            "schema_version": 1,
            "run": {
                "created_at": datetime.now().astimezone().isoformat(),
                "seed": int(self._seed),
                "method": self._method,
                "semantic_variant": getattr(self._config, "semantic_variant", "baseline"),
                "dataset": self._dataset,
                "backbone": self._backbone,
                "task_count": self._nb_tasks,
                "source_task_count": int(
                    getattr(self._config, "source_task_count", self._nb_tasks)
                ),
                "executed_task_count": self._nb_tasks,
                "run_scope": (
                    "diagnostic_prefix"
                    if getattr(self._config, "screening_task_limit", None)
                    else "full_schedule"
                ),
                "evaluation_source": evaluation_source,
                "validation_split": getattr(
                    self._config, "validation_split", None
                ),
                "increments": list(self._increment_steps),
                "fusion_weight": self._formal_fusion_weight,
                "configuration": config_dict,
                "configuration_sha256": _canonical_hash(config_dict),
                "git": git,
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                },
                "artifacts": _artifact_hashes(self._config),
            },
            "stages": [],
            "summary": {},
            "complete": False,
        }
        if self._formal_manifest_path.is_file():
            previous = json.loads(self._formal_manifest_path.read_text(encoding="utf-8"))
            if previous.get("run", {}).get("configuration_sha256") == self._formal_manifest["run"]["configuration_sha256"]:
                self._formal_manifest = previous
                self._formal_tracker.restore(previous.get("summary", {}))

    def _geometry_metrics(self) -> dict:
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        if hasattr(network, "class_codes") and network.class_codes.numel():
            codes = network.class_codes[: network.knowncls].float()
            target = network.target_gram[: network.knowncls, : network.knowncls].float()
            relation = network.semantic_relation[: network.knowncls, : network.knowncls].float()
            off_mask = ~torch.eye(network.knowncls, dtype=torch.bool, device=relation.device)
            off_values = relation[off_mask]
            return {
                "type": "stable_semantic_codebook",
                "class_count": int(network.knowncls),
                "code_dim": int(network.code_dim),
                "old_code_drift": float(getattr(network, "last_codebook_old_drift", 0.0)),
                "normalized_gram_error": float(gram_error(codes, target).item()),
                "max_absolute_gram_error": float((codes @ codes.T - target).abs().max().item()),
                "relation_offdiag_mean": float(off_values.mean().item()) if off_values.numel() else 0.0,
                "relation_offdiag_max": float(off_values.max().item()) if off_values.numel() else 0.0,
            }
        proto = network.proto_orth.detach().float()
        identity = torch.eye(proto.shape[0], device=proto.device)
        error = proto @ proto.T - identity
        return {
            "type": "topecl_dynamic_random_orthogonal",
            "class_count": int(network.knowncls),
            "code_dim": int(proto.shape[1]),
            "normalized_gram_error": float(
                torch.linalg.matrix_norm(error).item() / max(proto.shape[0], 1)
            ),
            "max_absolute_gram_error": float(error.abs().max().item()),
        }

    def _new_prototype_diagnostic_state(self, network):
        enabled = bool(
            getattr(self._config, "semantic_record_diagnostics", False)
        )
        if (
            not enabled
            or not hasattr(network, "text_bank")
            or network.text_bank.ndim != 3
            or network.text_bank.shape[1] <= 1
        ):
            return None
        sub_count = int(network.text_bank.shape[1] - 1)
        class_count = int(network.knowncls)
        return {
            "sample_count": 0,
            "max_weight_sum": 0.0,
            "entropy_sum": 0.0,
            "normalized_entropy_sum": 0.0,
            "active_sample_count": 0,
            "effective_count_sum": 0.0,
            "winner_counts": np.zeros(sub_count, dtype=np.int64),
            "weight_sums": np.zeros(sub_count, dtype=np.float64),
            "all_winner_counts": np.zeros(sub_count + 1, dtype=np.int64),
            "class_counts": np.zeros(class_count, dtype=np.int64),
            "base_correct": np.zeros(class_count, dtype=np.int64),
            "mpsb_correct": np.zeros(class_count, dtype=np.int64),
        }

    @staticmethod
    def _update_prototype_diagnostic_state(
        state, network, semantic_features, text_logits, targets
    ) -> None:
        if state is None:
            return
        known_mask = targets < network.knowncls
        if not torch.any(known_mask):
            return
        features = F.normalize(semantic_features[known_mask].float(), dim=-1)
        targets = targets[known_mask].long()
        bank = F.normalize(
            network.text_bank[: network.knowncls].float().to(features.device),
            dim=-1,
        )
        similarities = torch.einsum("bd,cpd->bcp", features, bank)
        sub_mask = network.text_mask[: network.knowncls, 1:].to(features.device)
        scaled = similarities[:, :, 1:] / float(network.pool_temperature)
        scaled = scaled.masked_fill(~sub_mask.unsqueeze(0), -torch.inf)
        active_classes = sub_mask.any(dim=-1)
        # An empty attribute set means base-only, not softmax(-inf,...,-inf).
        scaled = torch.where(active_classes[None, :, None], scaled, torch.zeros_like(scaled))
        sub_weights = torch.softmax(scaled, dim=-1) * sub_mask.unsqueeze(0)
        row = torch.arange(len(targets), device=targets.device)
        true_weights = sub_weights[row, targets]
        entropy = -(
            true_weights * true_weights.clamp_min(1e-12).log()
        ).sum(dim=-1)
        # Recompute the class-name-only reference with public TOPECL's exact
        # F.linear implementation. The formerly used alpha=0 aggregation is
        # algebraically equivalent, but its einsum kernel can flip a near-tie
        # under AMP and therefore is not an operation-exact inference audit.
        if hasattr(network, "proto_textemb"):
            base_logits = TOPECLNet.linear_textemb(
                network, semantic_features[known_mask]
            )
        else:
            # Lightweight diagnostic test doubles expose only text_bank.
            base_proto = network.text_bank[: network.knowncls, 0]
            base_logits = network.logit_scale_textemb.exp() * F.linear(
                F.normalize(semantic_features[known_mask], p=2, dim=1),
                F.normalize(base_proto.to(semantic_features.device), p=2, dim=1),
            )
        base_predictions = base_logits.argmax(dim=-1)
        mpsb_predictions = text_logits[known_mask, : network.knowncls].argmax(dim=-1)
        true_mask = network.text_mask[:network.knowncls].to(features.device)[targets]
        true_all_winners = similarities[row, targets].masked_fill(~true_mask, -torch.inf).argmax(dim=-1)
        true_active = active_classes[targets]
        true_counts = sub_mask[targets].sum(-1)
        normalized_entropy = torch.where(true_counts > 1, entropy / true_counts.clamp_min(2).float().log(), torch.zeros_like(entropy))
        true_sub_winners = true_weights[true_active].argmax(dim=-1)

        state["sample_count"] += int(len(targets))
        state["max_weight_sum"] += float(
            true_weights.max(dim=-1).values.sum().item()
        )
        state["entropy_sum"] += float(entropy.sum().item())
        state["normalized_entropy_sum"] += float(normalized_entropy.sum().item())
        state["active_sample_count"] += int(true_active.sum().item())
        state["effective_count_sum"] += float(torch.exp(entropy)[true_active].sum().item())
        state["winner_counts"] += np.bincount(
            true_sub_winners.cpu().numpy(), minlength=len(state["winner_counts"])
        )
        state["weight_sums"] += (
            true_weights.sum(dim=0).detach().cpu().numpy()
        )
        state["all_winner_counts"] += np.bincount(
            true_all_winners.cpu().numpy(),
            minlength=len(state["all_winner_counts"]),
        )
        target_values = targets.cpu().numpy()
        state["class_counts"] += np.bincount(
            target_values, minlength=len(state["class_counts"])
        )
        np.add.at(
            state["base_correct"],
            target_values,
            (base_predictions == targets).cpu().numpy().astype(np.int64),
        )
        np.add.at(
            state["mpsb_correct"],
            target_values,
            (mpsb_predictions == targets).cpu().numpy().astype(np.int64),
        )

    def _finalize_prototype_diagnostic_state(self, state):
        if state is None or state["sample_count"] == 0:
            return None
        count = state["sample_count"]
        sub_count = len(state["winner_counts"])
        class_names = [
            self._idx_to_class.get(index, str(index))
            for index in range(len(state["class_counts"]))
        ]
        per_class = []
        for index, class_name in enumerate(class_names):
            class_count = max(int(state["class_counts"][index]), 1)
            base_accuracy = 100.0 * state["base_correct"][index] / class_count
            mpsb_accuracy = 100.0 * state["mpsb_correct"][index] / class_count
            per_class.append({
                "class_name": class_name,
                "base_accuracy": float(base_accuracy),
                "mpsb_accuracy": float(mpsb_accuracy),
                "delta": float(mpsb_accuracy - base_accuracy),
            })
        return {
            "sample_count": count,
            "attribute_active_sample_count": state["active_sample_count"],
            "aggregation_diagnostic_denominator": "all samples; base-only cases contribute zero attribute activation",
            "base_only_accuracy": float(
                100.0 * state["base_correct"].sum() / count
            ),
            "base_only_accuracy_computation": "public_topecl_f_linear_path",
            "mpsb_text_accuracy": float(
                100.0 * state["mpsb_correct"].sum() / count
            ),
            "mean_max_subprototype_weight": state["max_weight_sum"] / count,
            "mean_normalized_subprototype_entropy": (
                state["normalized_entropy_sum"] / count
            ),
            "effective_subprototype_count": (
                state["effective_count_sum"] / count
            ),
            "subprototype_winner_counts": state["winner_counts"].tolist(),
            "mean_weight_per_slot": (
                state["weight_sums"] / count
            ).tolist(),
            "all_prototype_winner_counts_base_then_subs": (
                state["all_winner_counts"].tolist()
            ),
            "most_harmed_classes": sorted(
                per_class, key=lambda item: item["delta"]
            )[:10],
            "most_helped_classes": sorted(
                per_class, key=lambda item: item["delta"], reverse=True
            )[:10],
        }

    def _collect_formal_predictions(self):
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        network.eval()
        loader = self._openset_test_loader if self._is_openset_test else self._test_loader
        predictions = {head: [] for head in HEAD_NAMES}
        confidences = {head: [] for head in HEAD_NAMES}
        targets_all = []
        prototype_state = self._new_prototype_diagnostic_state(network)
        for _, inputs, targets in loader:
            inputs = inputs.cuda()
            targets_device = torch.as_tensor(targets, device=inputs.device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                logits, semantic_features = topecl_all_head_logits(
                    network,
                    inputs,
                    self._formal_fusion_weight,
                    return_semantic_features=True,
                )
                self._update_prototype_diagnostic_state(
                    prototype_state,
                    network,
                    semantic_features,
                    logits["text"],
                    targets_device,
                )
            for head in HEAD_NAMES:
                probability = torch.softmax(logits[head][:, : self._total_classes], dim=-1)
                confidence, prediction = probability.max(dim=-1)
                predictions[head].append(prediction.cpu().numpy())
                confidences[head].append(confidence.cpu().numpy())
            targets_all.append(np.asarray(targets))
        return (
            {head: np.concatenate(values) for head, values in predictions.items()},
            {head: np.concatenate(values) for head, values in confidences.items()},
            np.concatenate(targets_all),
            self._finalize_prototype_diagnostic_state(prototype_state),
        )

    def _write_formal_manifest(self) -> None:
        temporary = self._formal_manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._formal_manifest, indent=2, ensure_ascii=False, default=_json_ready),
            encoding="utf-8",
        )
        os.replace(temporary, self._formal_manifest_path)

    def eval_task(self):
        predictions, confidences, targets, prototype_diagnostics = (
            self._collect_formal_predictions()
        )
        known_mask = targets < self._total_classes
        known_targets = targets[known_mask]
        known_predictions = {
            head: values[known_mask] for head, values in predictions.items()
        }
        stage_metrics = self._formal_tracker.update(
            self._cur_task,
            known_predictions,
            known_targets,
            self._total_classes,
            self._increment_steps,
        )
        open_set = {
            head: _open_set_metrics(confidences[head], known_mask)
            for head in HEAD_NAMES
        }
        stage_record = {
            "task_id": int(self._cur_task),
            "known_classes": int(self._total_classes),
            "evaluated_at": datetime.now().astimezone().isoformat(),
            "evaluation_source": (
                getattr(self._config, "evaluation_source", None) or "test"
            ),
            "heads": stage_metrics,
            "open_set": open_set,
            "geometry": self._geometry_metrics(),
        }
        if getattr(self._config, "semantic_record_diagnostics", False):
            semantic_diagnostics = {}
            if prototype_diagnostics is not None:
                semantic_diagnostics["prototype_aggregation"] = (
                    prototype_diagnostics
                )
            network = (
                self._network.module
                if isinstance(self._network, nn.DataParallel)
                else self._network
            )
            if hasattr(network, "semantic_relation") and network.semantic_relation.numel():
                class_names = [
                    self._idx_to_class.get(index, str(index))
                    for index in range(self._total_classes)
                ]
                relation = network.semantic_relation[
                    : self._total_classes, : self._total_classes
                ]
                rho = float(getattr(network, "semantic_rho", 0.0))
                semantic_diagnostics["relation_vs_actual_confusion"] = {
                    head: relation_confusion_diagnostics(
                        relation,
                        known_predictions[head],
                        known_targets,
                        class_names,
                        rho,
                    )
                    for head in ("orth", "fusion")
                }
            if semantic_diagnostics:
                stage_record["semantic_diagnostics"] = semantic_diagnostics
        stages = self._formal_manifest["stages"]
        stages[:] = [item for item in stages if item.get("task_id") != self._cur_task]
        stages.append(stage_record)
        stages.sort(key=lambda item: item["task_id"])
        self._formal_manifest["summary"] = self._formal_tracker.summary(self._cur_task)
        self._formal_manifest["complete"] = self._cur_task == self._nb_tasks - 1
        if self._formal_manifest["complete"]:
            self._formal_manifest["completed_at"] = datetime.now().astimezone().isoformat()
        else:
            self._formal_manifest.pop("completed_at", None)
        self._write_formal_manifest()

        for head in HEAD_NAMES:
            summary = self._formal_manifest["summary"][head]
            self._logger.info(
                "FORMAL {}: stage_acc={:.2f}, avg_stage_acc={:.2f}, "
                "BWF={:.2f}, forgetting={:.2f}".format(
                    head.upper(),
                    stage_metrics[head]["overall_accuracy"],
                    summary["average_stage_accuracy"],
                    summary["bwf"],
                    summary["average_forgetting"],
                )
            )
        self._logger.info(f"Formal metrics written to: {self._formal_manifest_path}")
        self._logger.visual_log(
            "test",
            {
                f"Formal_{head}_Acc": stage_metrics[head]["overall_accuracy"]
                for head in HEAD_NAMES
            },
            step=self._cur_task,
        )

        if self._save_pred_record:
            output = {"targets": targets}
            for head in HEAD_NAMES:
                output[f"{head}_predictions"] = predictions[head]
                output[f"{head}_confidence"] = confidences[head]
            np.savez_compressed(
                Path(self._logdir)
                / f"formal_predictions_seed{self._seed}_task{self._cur_task}.npz",
                **output,
            )

    @staticmethod
    def _cpu_value(value):
        return value.detach().cpu() if isinstance(value, torch.Tensor) else value

    def _save_formal_checkpoint(self) -> Path:
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        state_dict = {
            name: value.detach().cpu().clone()
            for name, value in network.state_dict().items()
        }
        save_dict = {
            "state_dict": state_dict,
            "config": self._config.get_parameters_dict(),
            "task_id": self._cur_task,
            "memory_class_means": (
                None if self._memory_bank is None else self._memory_bank.get_class_means()
            ),
            "class_means": self._cpu_value(getattr(self, "class_means", None)),
            "class_covs": self._cpu_value(getattr(self, "class_covs", None)),
            "radius": self._cpu_value(getattr(self, "radius", None)),
        }
        path = Path(self._logdir) / f"seed{self._seed}_task{self._cur_task}_checkpoint.pkl"
        torch.save(save_dict, path)
        self._logger.info(f"checkpoint saved at: {path}")
        return path

    def after_task(self):
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        network.after_train()
        self._known_classes = self._total_classes
        if self._save_models and should_save_checkpoint(
            self._formal_checkpoint_policy,
            int(self._seed),
            self._cur_task,
            self._nb_tasks,
        ):
            path = self._save_formal_checkpoint()
            for stage in self._formal_manifest["stages"]:
                if stage["task_id"] == self._cur_task:
                    stage["checkpoint"] = {
                        "path": str(path.resolve()),
                        "sha256": _sha256_file(path),
                    }
            self._write_formal_manifest()
