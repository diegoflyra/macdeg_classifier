"""Camadas com parâmetros explícitos e operações funcionais do PyTorch.

Os tensores aprendidos são nn.Parameter. Os kernels de convolução, pooling e
produto linear continuam sendo os do PyTorch, com GPU e autograd disponíveis.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _initialize(weight: nn.Parameter, bias: nn.Parameter | None, fan_in: int) -> None:
    """Mesmo padrão inicial de Conv2d/Linear, antes de carregar os pesos ImageNet."""
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)


class Conv2d(nn.Module):
    """Convolução com tensor de filtros declarado por nós; kernel/stride/padding inteiros."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True, groups: int = 1) -> None:
        super().__init__()
        if groups <= 0 or in_channels % groups or out_channels % groups:
            raise ValueError("groups deve ser positivo e dividir os canais de entrada e saída.")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        _initialize(self.weight, self.bias, (in_channels // groups) * kernel_size * kernel_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.conv2d(images, self.weight, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)


class Linear(nn.Module):
    """Transformação linear com pesos (saídas, entradas) e viés explícitos."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        _initialize(self.weight, self.bias, in_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.weight, self.bias)


class ReLU(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(features, inplace=self.inplace)


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(features, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(features, self.output_size)


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("p deve estar entre 0 e 1.")
        self.p = p

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.dropout(features, p=self.p, training=self.training)


class LayerNorm(nn.Module):
    """Normalização da última dimensão, com escala e viés próprios."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = (channels,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(features, self.normalized_shape, self.weight, self.bias, self.eps)


class LayerNorm2d(LayerNorm):
    """LayerNorm sobre canais de uma imagem NCHW, sem estatísticas acumuladas."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class GELU(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(features, approximate="none")


class Permute(nn.Module):
    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.permute(*self.dims)


class Flatten(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.flatten(features, start_dim=1)


class StochasticDepth(nn.Module):
    """Descarta o ramo residual por amostra, com reescala; desligado em eval()."""

    def __init__(self, p: float) -> None:
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("p deve estar entre 0 e 1.")
        self.p = p

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return features
        survival = 1 - self.p
        shape = (features.shape[0],) + (1,) * (features.ndim - 1)
        mask = torch.empty(shape, dtype=features.dtype, device=features.device).bernoulli_(survival)
        if survival > 0:
            mask.div_(survival)
        return features * mask
