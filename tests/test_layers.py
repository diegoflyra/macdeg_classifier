"""Parâmetros, forward e gradientes das camadas funcionais, só com tensores sintéticos."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from macdeg import layers


@pytest.mark.parametrize("manual,reference,shape", [
    (lambda: layers.Conv2d(3, 4, 3, stride=2, padding=1),
     lambda: nn.Conv2d(3, 4, 3, stride=2, padding=1), (2, 3, 11, 13)),
    (lambda: layers.Conv2d(4, 4, 7, padding=3, groups=4),
     lambda: nn.Conv2d(4, 4, 7, padding=3, groups=4), (2, 4, 9, 11)),
    (lambda: layers.Linear(5, 3), lambda: nn.Linear(5, 3), (2, 7, 5)),
])
def test_explicit_parameters_match_forward_and_backward(manual, reference, shape):
    model, expected = manual().double(), reference().double()
    assert set(dict(model.named_parameters())) == {"weight", "bias"}
    assert isinstance(model.weight, nn.Parameter) and isinstance(model.bias, nn.Parameter)
    model.load_state_dict(expected.state_dict(), strict=True)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    actual, target = model(x), expected(expected_x)
    torch.testing.assert_close(actual, target)
    actual.square().mean().backward()
    target.square().mean().backward()
    torch.testing.assert_close(x.grad, expected_x.grad)
    torch.testing.assert_close(model.weight.grad, expected.weight.grad)
    torch.testing.assert_close(model.bias.grad, expected.bias.grad)


@pytest.mark.parametrize("manual,reference", [
    (lambda: layers.Conv2d(3, 4, 3), lambda: nn.Conv2d(3, 4, 3)),
    (lambda: layers.Linear(5, 3), lambda: nn.Linear(5, 3)),
])
def test_default_initialization_preserves_previous_seed_behavior(manual, reference):
    torch.manual_seed(123)
    model = manual()
    torch.manual_seed(123)
    expected = reference()
    assert torch.equal(model.weight, expected.weight)
    assert torch.equal(model.bias, expected.bias)


@pytest.mark.parametrize("manual,reference", [
    (lambda: layers.ReLU(), lambda: nn.ReLU()),
    (lambda: layers.MaxPool2d(3, stride=2), lambda: nn.MaxPool2d(3, stride=2)),
    (lambda: layers.AdaptiveAvgPool2d((6, 6)), lambda: nn.AdaptiveAvgPool2d((6, 6))),
])
def test_functional_activation_and_pooling_match_values_and_gradients(manual, reference):
    x = torch.randn(2, 3, 9, 11, dtype=torch.float64, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    actual, target = manual()(x), reference()(expected_x)
    torch.testing.assert_close(actual, target)
    actual.square().sum().backward()
    target.square().sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad)


def test_functional_dropout_respects_training_and_evaluation():
    model, expected = layers.Dropout(0.5), nn.Dropout(0.5)
    x = torch.ones(128)
    torch.manual_seed(4)
    actual = model.train()(x)
    torch.manual_seed(4)
    target = expected.train()(x)
    assert torch.equal(actual, target)
    assert torch.count_nonzero(actual) > 0 and torch.count_nonzero(actual) < len(x)
    assert torch.equal(model.eval()(x), x)


@pytest.mark.parametrize("kind,args,shape", [
    (layers.Conv2d, (3, 4, 3), (1, 3, 9, 9)),
    (layers.Linear, (5, 3), (2, 5)),
])
def test_bias_can_be_omitted(kind, args, shape):
    model = kind(*args, bias=False)
    assert set(dict(model.named_parameters())) == {"weight"}
    model(torch.randn(shape)).sum().backward()
    assert model.weight.grad is not None


def test_layer_norm_matches_channels_last_values_and_gradients():
    model, reference = layers.LayerNorm(5).double(), nn.LayerNorm(5, eps=1e-6).double()
    x = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    target_x = x.detach().clone().requires_grad_()
    reference.load_state_dict(model.state_dict(), strict=True)
    actual, target = model(x), reference(target_x)
    torch.testing.assert_close(actual, target)
    actual.square().sum().backward()
    target.square().sum().backward()
    torch.testing.assert_close(x.grad, target_x.grad)
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    torch.testing.assert_close(model.bias.grad, reference.bias.grad)


def test_layer_norm_2d_normalizes_channels_at_each_position():
    model = layers.LayerNorm2d(5).double()
    reference = nn.LayerNorm(5, eps=1e-6).double()
    x = torch.randn(2, 5, 7, 9, dtype=torch.float64)
    expected = reference(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("probability", [0., 0.3, 1.])
def test_stochastic_depth_matches_row_masks_and_disables_in_eval(probability):
    from torchvision.ops import stochastic_depth
    model = layers.StochasticDepth(probability)
    x = torch.ones(32, 3, 4, 4)
    torch.manual_seed(5)
    actual = model(x)
    torch.manual_seed(5)
    expected = stochastic_depth(x, probability, "row", training=True)
    assert torch.equal(actual, expected)
    assert torch.equal(model.eval()(x), x)


def test_functional_gelu_matches_reference():
    x = torch.linspace(-5, 5, 101, dtype=torch.float64)
    torch.testing.assert_close(layers.GELU()(x), nn.GELU()(x))
