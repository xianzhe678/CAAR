"""Verify the one-loader class-statistics path against public TOPECL."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from methods.multi_steps.topecl import TOPECL


class TinyDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(123)
        self.inputs = torch.randn(28, 8, generator=generator)
        self.targets = np.repeat(np.arange(4), 7)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return index, self.inputs[index], int(self.targets[index])


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Linear(8, 6, bias=False)
        self.adapter1 = nn.Linear(6, 4, bias=False)


class Logger:
    def info(self, *_args, **_kwargs):
        pass


def state(network, dataset):
    return SimpleNamespace(
        class_means=torch.empty(0),
        class_covs=torch.empty(0),
        radius=0.0,
        _known_classes=0,
        _total_classes=4,
        _cur_task=0,
        _network=network,
        _sampler_dataset=dataset,
        _batch_size=3,
        _num_workers=0,
        _logger=Logger(),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.manual_seed(7)
    network = TinyNetwork().cuda()
    dataset = TinyDataset()
    initial_rng_state = torch.random.get_rng_state()

    legacy = state(network, dataset)
    TOPECL._calculate_mean_convs_legacy(legacy)
    legacy_next_random = torch.rand(4)

    torch.random.set_rng_state(initial_rng_state)
    fast = state(network, dataset)
    TOPECL._calculate_mean_convs_aligned_loader(fast)
    fast_next_random = torch.rand(4)

    # CUDA FP16 kernels can differ by one rounding unit across separately
    # constructed loaders even though their samples and batch boundaries match.
    torch.testing.assert_close(
        fast.class_means, legacy.class_means, rtol=1e-3, atol=2e-4
    )
    torch.testing.assert_close(
        fast.class_covs, legacy.class_covs, rtol=2e-3, atol=2e-4
    )
    if not np.isclose(fast.radius, legacy.radius, rtol=1e-3, atol=2e-4):
        raise AssertionError(f"radius mismatch: {fast.radius} != {legacy.radius}")
    torch.testing.assert_close(fast_next_random, legacy_next_random, rtol=0, atol=0)
    print("Fast class-statistics CUDA equivalence smoke test passed")


if __name__ == "__main__":
    main()
