"""predict refaz em fp32 o lote cujas notas saem não finitas (estouro do fp16 na GPU)."""
import numpy as np
import torch
from torch import nn

from macdeg.train import predict


class OverflowOnce(nn.Module):
    """Devolve inf na primeira chamada (como o fp16 estourado) e o valor certo depois."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        logits = images.flatten(1).mean(dim=1, keepdim=True)
        if self.calls == 1:
            logits = logits.clone()
            logits[0] = torch.inf
        return logits


class Finite(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return images.flatten(1).mean(dim=1, keepdim=True)


def _loader():
    images = torch.zeros(4, 3, 8, 8)
    return [(images, torch.zeros(4))]


def test_predict_redoes_nonfinite_batch_in_fp32():
    model = OverflowOnce()
    probs = predict(model, _loader(), torch.device("cpu"))
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs, 0.5)
    assert model.calls == 2


def test_predict_finite_batch_runs_once():
    model = Finite()
    probs = predict(model, _loader(), torch.device("cpu"))
    np.testing.assert_allclose(probs, 0.5)
    assert model.calls == 1
