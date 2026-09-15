"""Pure metric tracking for reproducible multi-head continual evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


HEAD_NAMES = ("text", "orth", "fusion")


def _round(value: float) -> float:
    return float(np.around(value, decimals=2))


def calculate_stage_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    total_classes: int,
    increments: Sequence[int],
) -> dict:
    """Return overall, per-task, and macro class accuracy in percent."""
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if predictions.shape != targets.shape or targets.size == 0:
        raise ValueError("predictions and targets must be non-empty equal-length arrays")

    overall = _round(float((predictions == targets).mean() * 100.0))
    per_task = []
    begin = 0
    for increment in increments:
        end = begin + int(increment)
        if begin >= total_classes:
            break
        mask = (targets >= begin) & (targets < min(end, total_classes))
        if not np.any(mask):
            raise ValueError(f"no evaluation samples found for class range [{begin}, {end})")
        per_task.append(_round(float((predictions[mask] == targets[mask]).mean() * 100.0)))
        begin = end

    class_accuracies = []
    for class_id in np.unique(targets):
        mask = targets == class_id
        class_accuracies.append(float((predictions[mask] == class_id).mean() * 100.0))
    return {
        "overall_accuracy": overall,
        "per_task_accuracy": per_task,
        "class_average_accuracy": _round(float(np.mean(class_accuracies))),
        "sample_count": int(targets.size),
    }


def backward_forgetting(matrix: np.ndarray, stage: int) -> float:
    """TOPECL BWF averaged over every later stage of each old task."""
    if stage <= 0:
        return 0.0
    values = [
        float(np.mean(matrix[index, index + 1 : stage + 1] - matrix[index, index]))
        for index in range(stage)
    ]
    return _round(float(np.mean(values)))


def average_forgetting(matrix: np.ndarray, stage: int) -> float:
    """TOPECL forgetting: pre-final historical best minus final accuracy."""
    if stage <= 0:
        return 0.0
    values = [
        float(matrix[index, index:stage].max() - matrix[index, stage])
        for index in range(stage)
    ]
    return _round(float(np.mean(values)))


@dataclass
class MultiHeadMetricTracker:
    """Maintain one accuracy curve and task-stage matrix for every head."""

    nb_tasks: int
    heads: tuple[str, ...] = HEAD_NAMES
    overall_curves: dict[str, list[float]] = field(init=False)
    task_matrices: dict[str, np.ndarray] = field(init=False)
    class_average_curves: dict[str, list[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.overall_curves = {head: [] for head in self.heads}
        self.class_average_curves = {head: [] for head in self.heads}
        self.task_matrices = {
            head: np.zeros((self.nb_tasks, self.nb_tasks), dtype=np.float64)
            for head in self.heads
        }

    def restore(self, summary: Mapping[str, object]) -> None:
        for head in self.heads:
            head_summary = summary.get(head)
            if not isinstance(head_summary, Mapping):
                continue
            self.overall_curves[head] = [
                float(value) for value in head_summary.get("stage_accuracy_curve", [])
            ]
            self.class_average_curves[head] = [
                float(value)
                for value in head_summary.get("class_average_accuracy_curve", [])
            ]
            matrix = np.asarray(head_summary.get("task_accuracy_matrix", []), dtype=float)
            if matrix.ndim == 2:
                rows = min(matrix.shape[0], self.nb_tasks)
                cols = min(matrix.shape[1], self.nb_tasks)
                self.task_matrices[head][:rows, :cols] = matrix[:rows, :cols]

    def update(
        self,
        stage: int,
        predictions: Mapping[str, np.ndarray],
        targets: np.ndarray,
        total_classes: int,
        increments: Sequence[int],
    ) -> dict[str, dict]:
        if not 0 <= stage < self.nb_tasks:
            raise ValueError("stage is outside the configured task range")
        result = {}
        for head in self.heads:
            metrics = calculate_stage_accuracy(
                predictions[head], targets, total_classes, increments
            )
            curve = self.overall_curves[head]
            class_curve = self.class_average_curves[head]
            if len(curve) == stage:
                curve.append(metrics["overall_accuracy"])
                class_curve.append(metrics["class_average_accuracy"])
            elif len(curve) > stage:
                curve[stage] = metrics["overall_accuracy"]
                class_curve[stage] = metrics["class_average_accuracy"]
            else:
                raise ValueError("metric stages must be recorded in order")
            per_task = metrics["per_task_accuracy"]
            self.task_matrices[head][: len(per_task), stage] = per_task
            result[head] = metrics
        return result

    def summary(self, stage: int) -> dict[str, dict]:
        result = {}
        for head in self.heads:
            curve = self.overall_curves[head][: stage + 1]
            class_curve = self.class_average_curves[head][: stage + 1]
            matrix = self.task_matrices[head]
            result[head] = {
                "stage_accuracy_curve": curve,
                "average_stage_accuracy": _round(float(np.mean(curve))),
                "final_accuracy": curve[-1],
                "class_average_accuracy_curve": class_curve,
                "final_class_average_accuracy": class_curve[-1],
                "bwf": backward_forgetting(matrix, stage),
                "average_forgetting": average_forgetting(matrix, stage),
                "task_accuracy_matrix": matrix[: stage + 1, : stage + 1].tolist(),
            }
        return result
