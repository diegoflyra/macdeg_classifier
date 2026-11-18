"""ConvNeXt manual, compatível com os pesos ImageNet-1K do torchvision.

Referências: https://arxiv.org/abs/2201.03545 e
https://docs.pytorch.org/vision/stable/_modules/torchvision/models/convnext.html
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from macdeg import layers

CONVNEXT_TINY_IMAGENET_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
CONVNEXT_IMAGENET_URLS = {
    "convnext_tiny": CONVNEXT_TINY_IMAGENET_URL,
    "convnext_small": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
    "convnext_base": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
    "convnext_large": "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
}


class ConvNeXtBlock(nn.Module):
    """Depthwise 7×7, LayerNorm, MLP expandido 4×, LayerScale e residual."""

    def __init__(self, channels: int, layer_scale: float, stochastic_depth_prob: float) -> None:
        super().__init__()
        # Os índices acompanham o checkpoint oficial; Permute não tem parâmetros.
        self.block = nn.ModuleList([
            layers.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),
            layers.Permute((0, 2, 3, 1)),
            layers.LayerNorm(channels, eps=1e-6),
            layers.Linear(channels, 4 * channels),
            layers.GELU(),
            layers.Linear(4 * channels, channels),
            layers.Permute((0, 3, 1, 2)),
        ])
        self.layer_scale = nn.Parameter(torch.full((channels, 1, 1), layer_scale))
        self.stochastic_depth = layers.StochasticDepth(stochastic_depth_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.block[0](inputs)
        x = self.block[1](x)
        x = self.block[2](x)
        x = self.block[3](x)
        x = self.block[4](x)
        x = self.block[5](x)
        x = self.block[6](x)
        x = self.layer_scale * x
        x = self.stochastic_depth(x)
        return inputs + x


class ConvNeXt(nn.Module):
    variant = "convnext_tiny"
    depths = (3, 3, 9, 3)
    widths = (96, 192, 384, 768)
    default_stochastic_depth = 0.1

    def __init__(self, num_classes: int = 1000, stochastic_depth_prob: float | None = None,
                 layer_scale: float = 1e-6, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        if stochastic_depth_prob is None:
            stochastic_depth_prob = self.default_stochastic_depth
        if not math.isfinite(stochastic_depth_prob) or not 0 <= stochastic_depth_prob <= 1:
            raise ValueError("stochastic_depth_prob deve estar entre 0 e 1 e ser finito.")
        if type(gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing deve ser true ou false.")
        self.frozen_stages = 0
        self.gradient_checkpointing = gradient_checkpointing
        # Stem: redução espacial de quatro vezes.
        features = [nn.ModuleList([
            layers.Conv2d(3, self.widths[0], kernel_size=4, stride=4),
            layers.LayerNorm2d(self.widths[0], eps=1e-6),
        ])]
        block_index, total_blocks = 0, sum(self.depths)
        for stage_index, (channels, depth) in enumerate(zip(self.widths, self.depths)):
            stage = []
            for _ in range(depth):
                probability = stochastic_depth_prob * block_index / (total_blocks - 1)
                stage.append(ConvNeXtBlock(channels, layer_scale, probability))
                block_index += 1
            features.append(nn.ModuleList(stage))
            if stage_index < len(self.widths) - 1:
                features.append(nn.ModuleList([
                    layers.LayerNorm2d(channels, eps=1e-6),
                    layers.Conv2d(channels, self.widths[stage_index + 1], kernel_size=2, stride=2),
                ]))
        self.features = nn.ModuleList(features)
        self.avgpool = layers.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.ModuleList([
            layers.LayerNorm2d(self.widths[-1], eps=1e-6),
            layers.Flatten(),
            layers.Linear(self.widths[-1], num_classes),
        ])
        # Mesmo padrão do construtor oficial; o checkpoint substitui esses valores.
        for module in self.modules():
            if isinstance(module, (layers.Conv2d, layers.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features[0][0](images)
        x = self.features[0][1](x)
        # Estágios 1–4, intercalados com normalização e redução espacial.
        for feature_index in (1, 3, 5, 7):
            for block in self.features[feature_index]:
                if (self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                        and feature_index >= 2 * self.frozen_stages):
                    # Recalcula ativações no backward; mantém o sorteio do stochastic depth.
                    x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
                else:
                    x = block(x)
            if feature_index < 7:
                x = self.features[feature_index + 1][0](x)
                x = self.features[feature_index + 1][1](x)
        x = self.avgpool(x)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        return self.classifier[2](x)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # O extrator congelado não ganha ruído de stochastic depth ao treinar.
            for feature in self.features[:2 * self.frozen_stages]:
                feature.eval()
        return self


class ConvNeXtTiny(ConvNeXt):
    """18 blocos; mantém o construtor usado nos primeiros experimentos."""


class ConvNeXtSmall(ConvNeXt):
    variant = "convnext_small"
    depths = (3, 3, 27, 3)
    default_stochastic_depth = 0.4


class ConvNeXtBase(ConvNeXt):
    variant = "convnext_base"
    depths = (3, 3, 27, 3)
    widths = (128, 256, 512, 1024)
    default_stochastic_depth = 0.5


class ConvNeXtLarge(ConvNeXtBase):
    variant = "convnext_large"
    widths = (192, 384, 768, 1536)


CONVNEXT_MODELS = {
    "convnext_tiny": ConvNeXtTiny,
    "convnext_small": ConvNeXtSmall,
    "convnext_base": ConvNeXtBase,
    "convnext_large": ConvNeXtLarge,
}


def load_convnext_imagenet_weights(model: ConvNeXt) -> None:
    """Carrega o state_dict oficial inteiro, antes de substituir a saída."""
    state = torch.hub.load_state_dict_from_url(
        CONVNEXT_IMAGENET_URLS[model.variant], map_location="cpu", progress=True,
        check_hash=True, weights_only=True,
    )
    model.load_state_dict(state, strict=True)


def load_convnext_tiny_imagenet_weights(model: ConvNeXtTiny) -> None:
    """Compatibilidade com os primeiros experimentos da Tiny."""
    load_convnext_imagenet_weights(model)


def freeze_convnext_stages(model: ConvNeXt, freeze_first_stages: int) -> None:
    """N primeiros estágios e stem congelados; a transição ao próximo fica treinável."""
    if type(freeze_first_stages) is not int or not 0 <= freeze_first_stages <= 4:
        raise ValueError("freeze_first_stages deve ser um inteiro entre 0 e 4.")
    for index, feature in enumerate(model.features):
        feature.requires_grad_(index >= 2 * freeze_first_stages)
    model.classifier.requires_grad_(True)
    model.frozen_stages = freeze_first_stages
    model.train(model.training)
