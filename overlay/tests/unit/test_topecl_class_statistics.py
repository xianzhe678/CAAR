import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from methods.multi_steps.topecl import (
    _preserve_legacy_loader_rng,
    class_aligned_batches,
)


class ClassAlignedBatchTests(unittest.TestCase):
    def test_batches_are_class_pure_complete_and_ordered(self):
        targets = np.asarray([2, 2, 2, 0, 0, 1, 1, 1, 1])
        batches = class_aligned_batches(targets, batch_size=2)
        self.assertEqual(batches, [[3, 4], [5, 6], [7, 8], [0, 1], [2]])
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(len(targets))))
        for batch in batches:
            self.assertEqual(len(set(targets[batch].tolist())), 1)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            class_aligned_batches(np.asarray([0, 1]), batch_size=0)
        with self.assertRaises(ValueError):
            class_aligned_batches(np.asarray([[0, 1]]), batch_size=2)

    def test_fused_loader_preserves_legacy_parent_rng_position(self):
        dataset = TensorDataset(torch.arange(2))
        initial_state = torch.random.get_rng_state()

        for _ in range(4):
            next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
        expected = torch.rand(4)

        torch.random.set_rng_state(initial_state)
        next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
        _preserve_legacy_loader_rng(4)
        actual = torch.rand(4)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rng_preserver_rejects_empty_class_set(self):
        with self.assertRaises(ValueError):
            _preserve_legacy_loader_rng(0)


if __name__ == "__main__":
    unittest.main()
