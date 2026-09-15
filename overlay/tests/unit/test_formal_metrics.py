import unittest
import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from methods.multi_steps.formal_eval import (
    _open_set_metrics,
    should_save_checkpoint,
    topecl_all_head_logits,
)
from utils.formal_metrics import (
    MultiHeadMetricTracker,
    average_forgetting,
    backward_forgetting,
    calculate_stage_accuracy,
)
from tools.summarize_formal_results import aggregate
from tools.validate_formal_manifest import validate_manifest


class DummyTOPECLNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Identity()
        self.adapter1 = nn.Identity()
        self.adapter2 = nn.Identity()
        self.knowncls = 2

    @staticmethod
    def linear_textemb(features):
        return features[:, :2]

    @staticmethod
    def linear_orth(features):
        return 2.0 * features[:, :2]


class FormalMetricTests(unittest.TestCase):
    def test_retention_metrics_match_topecl_reference_formulas(self):
        matrix = np.array([
            [80.0, 70.0, 75.0],
            [0.0, 60.0, 65.0],
            [0.0, 0.0, 90.0],
        ])
        self.assertEqual(backward_forgetting(matrix, 2), -1.25)
        self.assertEqual(average_forgetting(matrix, 2), 0.0)

    def test_stage_accuracy_is_overall_per_task_and_per_class(self):
        result = calculate_stage_accuracy(
            np.array([0, 0, 2, 3]),
            np.array([0, 1, 2, 3]),
            total_classes=4,
            increments=[2, 2, 2],
        )
        self.assertEqual(result["overall_accuracy"], 75.0)
        self.assertEqual(result["per_task_accuracy"], [50.0, 100.0])
        self.assertEqual(result["class_average_accuracy"], 75.0)

    def test_tracker_reports_topecl_bwf_and_forgetting(self):
        tracker = MultiHeadMetricTracker(nb_tasks=3)
        same = lambda values: {head: np.asarray(values) for head in tracker.heads}
        tracker.update(0, same([0, 1]), np.array([0, 1]), 2, [2, 2, 2])
        tracker.update(
            1,
            same([0, 0, 2, 3]),
            np.array([0, 1, 2, 3]),
            4,
            [2, 2, 2],
        )
        tracker.update(
            2,
            same([1, 0, 2, 3, 4, 5]),
            np.array([0, 1, 2, 3, 4, 5]),
            6,
            [2, 2, 2],
        )
        summary = tracker.summary(2)["text"]
        self.assertEqual(summary["stage_accuracy_curve"], [100.0, 75.0, 66.67])
        self.assertEqual(summary["bwf"], -37.5)
        self.assertEqual(summary["average_forgetting"], 50.0)

        restored = MultiHeadMetricTracker(nb_tasks=3)
        restored.restore(tracker.summary(2))
        self.assertEqual(restored.summary(2), tracker.summary(2))

    def test_all_heads_share_features_and_fusion_is_exact(self):
        network = DummyTOPECLNetwork()
        inputs = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        logits = topecl_all_head_logits(network, inputs, fusion_weight=0.5)
        torch.testing.assert_close(logits["text"], inputs)
        torch.testing.assert_close(logits["orth"], 2.0 * inputs)
        torch.testing.assert_close(logits["fusion"], 2.0 * inputs)

    def test_checkpoint_policy_is_explicit(self):
        self.assertTrue(should_save_checkpoint("all", 100, 2, 10))
        self.assertFalse(should_save_checkpoint("final", 42, 8, 10))
        self.assertTrue(should_save_checkpoint("final", 42, 9, 10))
        self.assertTrue(should_save_checkpoint("seed42_all_else_final", 42, 3, 10))
        self.assertFalse(should_save_checkpoint("seed42_all_else_final", 100, 3, 10))
        self.assertTrue(should_save_checkpoint("seed42_all_else_final", 100, 9, 10))
        with self.assertRaises(ValueError):
            should_save_checkpoint("sometimes", 42, 0, 10)

    def test_open_set_metrics_use_known_class_confidence(self):
        metrics = _open_set_metrics(
            np.array([0.95, 0.85, 0.20, 0.10]),
            np.array([True, True, False, False]),
        )
        self.assertEqual(metrics["auc"], 100.0)
        self.assertEqual(metrics["fpr95"], 0.0)
        self.assertEqual(metrics["average_precision"], 100.0)

    @staticmethod
    def _manifest(seed, accuracy):
        head_stage = {
            "overall_accuracy": accuracy,
            "per_task_accuracy": [accuracy],
            "class_average_accuracy": accuracy,
            "sample_count": 10,
        }
        head_summary = {
            "stage_accuracy_curve": [accuracy],
            "average_stage_accuracy": accuracy,
            "final_accuracy": accuracy,
            "class_average_accuracy_curve": [accuracy],
            "final_class_average_accuracy": accuracy,
            "bwf": 0.0,
            "average_forgetting": 0.0,
            "task_accuracy_matrix": [[accuracy]],
        }
        return {
            "schema_version": 1,
            "run": {
                "seed": seed,
                "method": "semantic_topecl",
                "semantic_variant": "full",
                "dataset": "cifar100",
                "backbone": "openai_ViT-B/16",
                "task_count": 1,
                "increments": [100],
                "fusion_weight": 1.0,
                "configuration": {},
                "configuration_sha256": hashlib.sha256(b"{}").hexdigest(),
                "git": {"dirty": False, "commit": "test-commit"},
                "artifacts": {},
            },
            "stages": [
                {
                    "task_id": 0,
                    "heads": {head: dict(head_stage) for head in ("text", "orth", "fusion")},
                    "geometry": {"type": "topecl_dynamic_random_orthogonal"},
                }
            ],
            "summary": {
                head: dict(head_summary) for head in ("text", "orth", "fusion")
            },
            "complete": True,
        }

    def test_manifest_validation_and_seed_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed, accuracy in ((42, 80.0), (100, 82.0)):
                path = Path(directory) / f"seed{seed}.json"
                path.write_text(
                    json.dumps(self._manifest(seed, accuracy)), encoding="utf-8"
                )
                paths.append(path)
                result = validate_manifest(
                    path, require_complete=True, require_clean_git=True
                )
                self.assertEqual(result["errors"], [])
            result = aggregate(paths)
            metric = result[0]["heads"]["fusion"]["final_accuracy"]
            self.assertEqual(metric["mean"], 81.0)
            self.assertEqual(metric["std"], 1.41)


if __name__ == "__main__":
    unittest.main()
