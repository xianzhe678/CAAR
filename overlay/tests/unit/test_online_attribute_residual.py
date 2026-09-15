import copy
import pytest
import torch

from utils.online_attribute_residual import OnlineAttributeResidual, residual_objective


@pytest.mark.parametrize("attributes", [False, True])
def test_zero_exact_and_gradient_isolated(attributes):
    torch.manual_seed(35)
    module = OnlineAttributeResidual()
    assert sum(p.numel() for p in module.parameters()) == 16912
    h = torch.randn(8, 512, requires_grad=True)
    bank = torch.randn(10, 5, 512, requires_grad=True)
    scale = torch.tensor(14., requires_grad=True)
    native = torch.randn(8, 10, requires_grad=True)
    delta = module(h, bank, scale, attributes=attributes)
    assert torch.count_nonzero(delta) == 0
    assert torch.equal(native.detach() + delta, native.detach())
    loss, _ = residual_objective(native, delta, torch.arange(8), 1.)
    loss.backward()
    assert module.up.weight.grad.norm() > 0
    assert all(x.grad is None for x in (h, bank, scale, native))


def test_identical_attributes_recover_name_control_and_nonzero_correction():
    torch.manual_seed(35)
    module = OnlineAttributeResidual()
    torch.nn.init.normal_(module.up.weight, std=.03)
    h = torch.randn(8, 512)
    bank = torch.randn(10, 1, 512).expand(-1, 5, -1).clone()
    name = module(h, bank, 14., attributes=False)
    attr = module(h, bank, 14., attributes=True)
    torch.testing.assert_close(name, attr, atol=2e-6, rtol=1e-4)
    assert name.norm() > 0


def test_forward_consumes_no_rng_and_arms_can_share_initialization():
    torch.manual_seed(35)
    module = OnlineAttributeResidual()
    other = copy.deepcopy(module)
    h, bank = torch.randn(8, 512), torch.randn(10, 5, 512)
    state = torch.random.get_rng_state().clone()
    module(h, bank, 14., attributes=True)
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(p, q) for p, q in zip(module.parameters(), other.parameters()))


def test_focal_and_infonce_losses_are_finite_and_train_residual():
    native = torch.randn(8, 10)
    labels = torch.arange(8)
    for kwargs in ({"focal_gamma": 2.0}, {"infonce_weight": 0.1}):
        delta = torch.zeros_like(native, requires_grad=True)
        loss, parts = residual_objective(native, delta, labels, **kwargs)
        assert torch.isfinite(loss)
        assert {"ce", "focal", "kl", "infonce"} == set(parts)
        loss.backward()
        assert delta.grad.norm() > 0


def test_masked_attribute_bank_ignores_padding_and_name_only_class():
    torch.manual_seed(41)
    module = OnlineAttributeResidual()
    torch.nn.init.normal_(module.up.weight, std=.03)
    h = torch.randn(3, 512)
    bank = torch.randn(2, 5, 512)
    mask = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]])
    first = module(h, bank, 14., attributes=True, bank_mask=mask)
    bank[:, 2:] = torch.randn_like(bank[:, 2:]) * 1000
    second = module(h, bank, 14., attributes=True, bank_mask=mask)
    torch.testing.assert_close(first, second)
    name = module(h, bank, 14., attributes=False)
    torch.testing.assert_close(first[:, 1], name[:, 1], atol=2e-6, rtol=1e-4)


def test_half_ce_half_focal_is_the_mean_of_endpoint_losses():
    native = torch.randn(8, 10)
    delta = torch.randn(8, 10, requires_grad=True)
    labels = torch.arange(8)
    ce, _ = residual_objective(native, delta, labels, focal_gamma=0.0)
    focal, _ = residual_objective(native, delta, labels, focal_gamma=2.0)
    hybrid, _ = residual_objective(native, delta, labels, focal_gamma=2.0, focal_mix_weight=0.5)
    torch.testing.assert_close(hybrid, 0.5 * (ce + focal))


def test_attribute_conditioned_gate_is_capacity_matched_and_attribute_dependent():
    torch.manual_seed(56)
    module = OnlineAttributeResidual()
    assert sum(p.numel() for p in module.parameters()) == 16912
    h = torch.randn(4, 512)
    bank = torch.randn(6, 5, 512)
    zero = module(h, bank, 14.0, attributes=True, conditioned_gate=True)
    assert torch.count_nonzero(zero) == 0

    torch.nn.init.normal_(module.up.weight, std=0.03)
    plain = module(h, bank, 14.0, attributes=True)
    gated = module(h, bank, 14.0, attributes=True, conditioned_gate=True)
    assert gated.shape == plain.shape == (4, 6)
    assert not torch.equal(gated, plain)
