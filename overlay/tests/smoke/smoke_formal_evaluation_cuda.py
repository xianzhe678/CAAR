"""Two-stage CUDA smoke test for the formal metrics manifest."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from methods.multi_steps.formal_eval import TOPECLFormalEvalMixin


class Config:
    formal_fusion_weight = 1.0
    formal_checkpoint_policy = "all"
    formal_require_clean_git = False
    semantic_variant = "smoke"
    config = None
    semantic_description_path = None

    def get_parameters_dict(self):
        return {
            "formal_fusion_weight": self.formal_fusion_weight,
            "formal_checkpoint_policy": self.formal_checkpoint_policy,
            "formal_require_clean_git": self.formal_require_clean_git,
            "semantic_variant": self.semantic_variant,
        }


class Logger:
    def info(self, message):
        print(message)

    def visual_log(self, *args, **kwargs):
        pass


class DummyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Identity()
        self.adapter1 = nn.Identity()
        self.adapter2 = nn.Identity()
        self.knowncls = 2
        self.proto_orth = nn.Parameter(torch.eye(4), requires_grad=False)

    def linear_textemb(self, features):
        return 8.0 * features[:, : self.knowncls]

    def linear_orth(self, features):
        return 6.0 * features[:, : self.knowncls]

    def after_train(self):
        pass


class Learner(TOPECLFormalEvalMixin):
    pass


def loader(class_count: int) -> DataLoader:
    features = torch.eye(4)[:class_count]
    targets = torch.arange(class_count)
    indexes = torch.arange(class_count)
    return DataLoader(TensorDataset(indexes, features, targets), batch_size=2)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    with tempfile.TemporaryDirectory() as directory:
        learner = Learner()
        learner._config = Config()
        learner._seed = 42
        learner._nb_tasks = 2
        learner._increment_steps = [2, 2]
        learner._logdir = directory
        learner._method = "formal_smoke"
        learner._dataset = "synthetic"
        learner._backbone = "identity"
        learner._logger = Logger()
        learner._is_openset_test = False
        learner._save_pred_record = False
        learner._save_models = True
        learner._memory_bank = None
        learner._network = DummyNetwork().cuda()
        learner._init_formal_evaluation()

        learner._cur_task = 0
        learner._known_classes = 0
        learner._total_classes = 2
        learner._test_loader = loader(2)
        learner.eval_task()
        learner.after_task()
        assert learner._network.proto_orth.is_cuda

        learner._cur_task = 1
        learner._total_classes = 4
        learner._network.knowncls = 4
        learner._test_loader = loader(4)
        learner.eval_task()
        learner.after_task()
        assert learner._network.proto_orth.is_cuda

        manifest_path = Path(directory) / "formal_metrics_seed42.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["complete"] is True
        assert len(manifest["stages"]) == 2
        assert len(list(Path(directory).glob("seed42_task*_checkpoint.pkl"))) == 2
        for head in ("text", "orth", "fusion"):
            assert manifest["summary"][head]["final_accuracy"] == 100.0
        print("Formal evaluation CUDA smoke test passed:", manifest_path.name)


if __name__ == "__main__":
    main()
