"""Rede híbrida de DMRI com uma vista global e recortes em resolução nativa."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from macdeg import layers


HYBRID_MODEL = "hibrida_dmri"
HYBRID_BACKBONES = {HYBRID_MODEL: "resnet18", "hibrida_dmri_resnet50": "resnet50"}
NUM_CROPS = 3


class MultiscaleResidual(nn.Module):
    """Dois campos receptivos (3×3 e 5×5) com projeção e soma residual."""

    def __init__(self) -> None:
        super().__init__()
        self.short_branch = nn.Sequential(layers.Conv2d(128, 64, 3, padding=1), layers.ReLU())
        self.long_branch = nn.Sequential(
            layers.Conv2d(128, 64, 3, padding=1), layers.ReLU(),
            layers.Conv2d(64, 64, 3, padding=1), layers.ReLU(),
        )
        self.project = layers.Conv2d(128, 128, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branches = torch.cat((self.short_branch(features), self.long_branch(features)), dim=1)
        return features + self.alpha * self.project(branches)


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local = layers.Conv2d(128, 64, 1)
        self.context = layers.Conv2d(512, 64, 1)
        self.score = layers.Conv2d(64, 1, 1)

    def forward(self, local: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        context = F.interpolate(self.context(deep).float(), size=local.shape[-2:],
                                mode="bilinear", align_corners=False)
        return self.score(F.relu(self.local(local) + context)).squeeze(1)


def _extract_crops(images: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Seleciona três janelas sem sobreposição, em ordem de escore e sem reamostrar pixels."""
    side = images.shape[-1]
    crop_side = side * 3 // 16  # 192 px a 1024, mesma fração no --teste.
    rows, cols = scores.shape[-2:]
    # Centro de cada célula na vista nativa. Perto da borda, desloca a janela inteira.
    y = (2 * torch.arange(rows, device=images.device) + 1) * side // (2 * rows) - crop_side // 2
    x = (2 * torch.arange(cols, device=images.device) + 1) * side // (2 * cols) - crop_side // 2
    yy, xx = torch.meshgrid(y.clamp(0, side - crop_side), x.clamp(0, side - crop_side), indexing="ij")
    yy, xx = yy.flatten(), xx.flatten()

    # A decisão discreta não participa do backward; o mapa recebe gradiente pela média ponderada
    # e, quando habilitada, pela perda de localização.
    remaining = scores.detach().flatten(1)
    chosen_y, chosen_x = [], []
    for _ in range(NUM_CROPS):
        index = remaining.argmax(dim=1)
        top, left = yy[index], xx[index]
        chosen_y.append(top)
        chosen_x.append(left)
        overlaps = ((yy - top[:, None]).abs() < crop_side) & ((xx - left[:, None]).abs() < crop_side)
        remaining = remaining.masked_fill(overlaps, -torch.inf)

    offsets = torch.arange(crop_side, device=images.device)
    ys = torch.stack(chosen_y, dim=1)[:, :, None, None] + offsets[None, None, :, None]
    xs = torch.stack(chosen_x, dim=1)[:, :, None, None] + offsets[None, None, None, :]
    batch = torch.arange(len(images), device=images.device)[:, None, None, None]
    return images.permute(0, 2, 3, 1)[batch, ys, xs].permute(0, 1, 4, 2, 3)


class HybridDMRI(nn.Module):
    def __init__(self, pretrained: bool, outputs: int = 1, freeze_first_stages: int = 0, *,
                 backbone_name: str = "resnet18") -> None:
        super().__init__()
        from torchvision import models

        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            # A ResNet-50 mantém seus canais nativos entre estágios. As projeções
            # mantêm as larguras do bloco multiescala, da atenção e da cabeça.
            self.local_adapter = layers.Conv2d(512, 128, 1, bias=False)
            self.deep_adapter = layers.Conv2d(2048, 512, 1, bias=False)
        elif backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.local_adapter = nn.Identity()
            self.deep_adapter = nn.Identity()
        else:
            raise ValueError(f"Backbone da híbrida não suportado: {backbone_name!r}.")
        self.backbone.fc = nn.Identity()
        self.frozen_stages = freeze_first_stages
        self.multiscale = MultiscaleResidual()
        self.attention = SpatialAttention()
        self.crop_encoder = nn.Sequential(
            layers.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), layers.ReLU(),
            layers.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), layers.ReLU(),
            layers.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), layers.ReLU(),
        )
        self.classifier = nn.Sequential(layers.Linear(512 + 128 + 128, 128), layers.ReLU(), layers.Linear(128, outputs))
        for module in self._frozen_modules():
            module.requires_grad_(False)
        self.train(self.training)

    def _frozen_modules(self) -> tuple[nn.Module, ...]:
        if self.frozen_stages == 0:
            return ()
        backbone = self.backbone
        stages = (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        return (backbone.conv1, backbone.bn1, *stages[:self.frozen_stages])

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self._frozen_modules():
            module.eval()  # BatchNorm congelada mantém as estatísticas do ImageNet.
        return self

    def forward(
        self, images: torch.Tensor, *, return_localization: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2] != images.shape[-1] or images.shape[-1] < 32:
            raise ValueError("A híbrida requer imagens RGB quadradas de pelo menos 32 pixels.")

        x = F.interpolate(images, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True)
        backbone = self.backbone
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
        x = backbone.layer2(backbone.layer1(x))
        local = self.multiscale(self.local_adapter(x))
        deep = self.deep_adapter(backbone.layer4(backbone.layer3(x)))
        scores = self.attention(local, deep)

        weights = scores.flatten(1).float().softmax(dim=1).to(local.dtype)
        local_summary = torch.bmm(local.flatten(2), weights.unsqueeze(2)).squeeze(2)
        crops = _extract_crops(images, scores)
        encoded = self.crop_encoder(crops.flatten(0, 1)).mean(dim=(-2, -1))
        crop_summary = encoded.reshape(len(images), NUM_CROPS, -1).mean(dim=1)
        logits = self.classifier(torch.cat((deep.mean(dim=(-2, -1)), local_summary, crop_summary), dim=1))
        if return_localization:
            localization = scores.flatten(1).float().topk(NUM_CROPS, dim=1).values.mean(dim=1)
            return logits, localization
        return logits
