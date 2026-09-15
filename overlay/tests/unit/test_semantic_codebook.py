import unittest

import torch

from backbone.semantic_codebook import (
    extend_codebook,
    full_classifier_bank,
    gram_error,
    initial_codebook,
    reserve_basis,
    semantic_target_gram,
    set_rbf_relation,
)


class SemanticCodebookTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(19)
        self.bank = torch.randn(8, 4, 16, generator=generator, dtype=torch.float64)
        self.relation = set_rbf_relation(self.bank, sigma=0.6)
        self.gram = semantic_target_gram(self.relation, rho=0.2)
        self.code_dim = 12

    def test_initial_codebook_realizes_target_gram(self):
        codes = initial_codebook(self.gram[:3, :3], self.code_dim)
        self.assertEqual(codes.shape, (3, self.code_dim))
        torch.testing.assert_close(
            codes @ codes.T,
            self.gram[:3, :3],
            atol=1e-10,
            rtol=1e-9,
        )

    def test_incremental_completion_preserves_old_codes_exactly(self):
        codes = initial_codebook(self.gram[:3, :3], self.code_dim)
        old_snapshot = codes.clone()
        codes = extend_codebook(
            codes,
            self.gram[:3, 3:5],
            self.gram[3:5, 3:5],
        )
        self.assertTrue(torch.equal(codes[:3], old_snapshot))
        torch.testing.assert_close(
            codes @ codes.T,
            self.gram[:5, :5],
            atol=1e-10,
            rtol=1e-9,
        )

        old_snapshot = codes.clone()
        codes = extend_codebook(
            codes,
            self.gram[:5, 5:8],
            self.gram[5:8, 5:8],
        )
        self.assertTrue(torch.equal(codes[:5], old_snapshot))
        torch.testing.assert_close(
            codes @ codes.T,
            self.gram,
            atol=1e-10,
            rtol=1e-9,
        )
        self.assertLess(float(gram_error(codes, self.gram)), 1e-10)

    def test_reserve_basis_is_orthonormal_and_orthogonal_to_classes(self):
        codes = initial_codebook(self.gram[:5, :5], self.code_dim)
        reserve = reserve_basis(codes)
        self.assertEqual(reserve.shape, (self.code_dim - 5, self.code_dim))
        torch.testing.assert_close(
            reserve @ codes.T,
            torch.zeros(self.code_dim - 5, 5, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-9,
        )
        torch.testing.assert_close(
            reserve @ reserve.T,
            torch.eye(self.code_dim - 5, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-9,
        )
        classifier = full_classifier_bank(codes)
        expected = torch.block_diag(
            self.gram[:5, :5],
            torch.eye(self.code_dim - 5, dtype=torch.float64),
        )
        torch.testing.assert_close(
            classifier @ classifier.T,
            expected,
            atol=1e-10,
            rtol=1e-9,
        )

    def test_reserve_basis_at_full_capacity_is_empty(self):
        codes = initial_codebook(self.gram, 8)
        reserve = reserve_basis(codes)
        self.assertEqual(reserve.shape, (0, 8))

    def test_rho_zero_degenerates_to_pure_orthogonality(self):
        identity_gram = semantic_target_gram(self.relation, rho=0.0)
        codes = initial_codebook(identity_gram[:3, :3], self.code_dim)
        codes = extend_codebook(
            codes, identity_gram[:3, 3:5], identity_gram[3:5, 3:5]
        )
        codes = extend_codebook(
            codes, identity_gram[:5, 5:8], identity_gram[5:8, 5:8]
        )
        torch.testing.assert_close(
            codes @ codes.T,
            torch.eye(8, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-9,
        )
        classifier = full_classifier_bank(codes)
        torch.testing.assert_close(
            classifier @ classifier.T,
            torch.eye(self.code_dim, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-9,
        )

    def test_capacity_and_shape_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            initial_codebook(self.gram, 7)
        old = initial_codebook(self.gram[:3, :3], 4)
        with self.assertRaises(ValueError):
            extend_codebook(old, self.gram[:3, 3:5], self.gram[3:5, 3:5])
        with self.assertRaises(ValueError):
            extend_codebook(old, torch.zeros(2, 1), torch.eye(1))


if __name__ == "__main__":
    unittest.main()

