"""Redes em duas resoluções sobre a EfficientNetV2-S. Entrada: a retinografia em 1024 px.

- `hibrida_v2s`: o bloco proposto pelo Gustavo (branch `rede-hibrida`, docs/hibrida_dmri.md), com a V2-S
  no lugar da ResNet-18. A vista global (512 px) passa pela V2-S; da saída do estágio 2 (64 x 64, cada
  posição resume 8 px) sai um ramo multiescala residual (ideia da Inception-ResNet); uma atenção espacial
  combina esse mapa com o mapa profundo (16 x 16) e escolhe 3 janelas de 192 px na imagem em 1024 px,
  onde uma drusa média tem 5 a 10 px (contra 2,5 a 5 px em 512). Os recortes passam por uma CÓPIA
  pré-treinada dos estágios 0-3 da V2-S (no desenho original, uma CNN pequena começava do zero). A
  cabeça junta o resumo global, o multiescala ponderado pela atenção e a média dos recortes (MIL: o
  rótulo é do olho, não do recorte). Referências: Shen et al. (GMIC), Ilse et al. (attention MIL),
  Szegedy et al. (Inception-ResNet).
- `v2s_macula`: vista global em 512 px + recorte central de 448 px em resolução nativa (~6 mm, a grade
  do AREDS; a fóvea fica perto do centro nas nossas bases), cada um com a sua V2-S pré-treinada. Sem
  atenção: testa só "ver a mácula em resolução nativa ajuda?". Diferente do `crop: 0.6` da rodada 3, que
  substituía a imagem inteira e reduzia a região para 512 px.

Saída: logits (saída 0 = DMRI; depois as auxiliares, como nas outras redes). Com
`return_localization=True`, a híbrida também devolve o logit de localização (média dos 3 maiores escores
da atenção), usado pela perda auxiliar `aux_localization`.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

V2S = "tf_efficientnetv2_s.in21k_ft_in1k"
# Nome do modelo -> rede do timm cujos pesos e normalização ele usa.
BACKBONE = {"hibrida_v2s": V2S, "v2s_macula": V2S}
NUM_CROPS = 3
CROP_FRACTION = 3 / 16       # 192 px em 1024 (a mesma fração em imagens menores, nos testes)
MACULA_FRACTION = 7 / 16     # 448 px em 1024


def _v2s(pretrained: bool) -> nn.Module:
    import timm
    return timm.create_model(V2S, pretrained=pretrained, num_classes=0)


def _half(images: torch.Tensor) -> torch.Tensor:
    """Vista global: metade do lado, com antialias (como a redução do cache para 512 px)."""
    return F.interpolate(images, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True)


class MultiscaleResidual(nn.Module):
    """Dois campos receptivos (3x3 e 5x5) concatenados, projeção 1x1 e soma residual com alfa treinável
    (começa em 0,1: o bloco novo entra quase desligado e não estraga o mapa pré-treinado)."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        half = channels // 2
        self.short_branch = nn.Sequential(nn.Conv2d(channels, half, 3, padding=1), nn.ReLU())
        self.long_branch = nn.Sequential(nn.Conv2d(channels, half, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(half, half, 3, padding=1), nn.ReLU())
        self.project = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branches = torch.cat((self.short_branch(features), self.long_branch(features)), dim=1)
        return features + self.alpha * self.project(branches)


class SpatialAttention(nn.Module):
    """Escore por posição do mapa local, usando o contexto do mapa profundo ampliado até a mesma grade."""

    def __init__(self, local_channels: int, deep_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.local = nn.Conv2d(local_channels, hidden, 1)
        self.context = nn.Conv2d(deep_channels, hidden, 1)
        self.score = nn.Conv2d(hidden, 1, 1)

    def forward(self, local: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        context = F.interpolate(self.context(deep).float(), size=local.shape[-2:], mode="bilinear",
                                align_corners=False).to(local.dtype)
        return self.score(F.relu(self.local(local) + context)).squeeze(1)


def extract_crops(images: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Três janelas sem sobreposição, em ordem de escore, copiando os pixels nativos (sem reamostrar).
    Devolve (recortes [B, 3, C, s, s], cantos superiores esquerdos [B, 3, 2] em pixels). A escolha é
    discreta: não passa gradiente às coordenadas (o mapa aprende pela média ponderada e pela perda de
    localização). Código do Gustavo (branch rede-hibrida)."""
    side = images.shape[-1]
    crop_side = int(side * CROP_FRACTION)
    rows, cols = scores.shape[-2:]
    y = (2 * torch.arange(rows, device=images.device) + 1) * side // (2 * rows) - crop_side // 2
    x = (2 * torch.arange(cols, device=images.device) + 1) * side // (2 * cols) - crop_side // 2
    yy, xx = torch.meshgrid(y.clamp(0, side - crop_side), x.clamp(0, side - crop_side), indexing="ij")
    yy, xx = yy.flatten(), xx.flatten()
    remaining = scores.detach().float().flatten(1)
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
    crops = images.permute(0, 2, 3, 1)[batch, ys, xs].permute(0, 1, 4, 2, 3)
    return crops, torch.stack((torch.stack(chosen_y, 1), torch.stack(chosen_x, 1)), dim=2)


class HybridV2S(nn.Module):
    def __init__(self, pretrained: bool, outputs: int = 1) -> None:
        super().__init__()
        self.backbone = _v2s(pretrained)
        # Recortes: cópia dos estágios 0-3 (saída de 128 canais), com os pesos pré-treinados.
        self.crop_encoder = nn.Sequential(copy.deepcopy(self.backbone.conv_stem), copy.deepcopy(self.backbone.bn1),
                                          *copy.deepcopy(self.backbone.blocks[:4]))
        self.local_adapter = nn.Conv2d(64, 128, 1)
        self.multiscale = MultiscaleResidual(128)
        self.attention = SpatialAttention(128, self.backbone.num_features)
        self.classifier = nn.Sequential(nn.Linear(self.backbone.num_features + 128 + 128, 128), nn.ReLU(),
                                        nn.Linear(128, outputs))
        self.last_attention = None     # escores da última chamada (explicabilidade)
        self.last_crops = None         # cantos dos recortes da última chamada

    def deep_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.backbone
        x = b.bn1(b.conv_stem(_half(images)))
        x = b.blocks[2](b.blocks[1](b.blocks[0](x)))
        local = x
        for block in b.blocks[3:]:
            x = block(x)
        return local, b.bn2(b.conv_head(x))

    def forward(self, images: torch.Tensor, *, return_localization: bool = False):
        local, deep = self.deep_features(images)
        local = self.multiscale(self.local_adapter(local))
        scores = self.attention(local, deep)
        weights = scores.flatten(1).float().softmax(dim=1).to(local.dtype)
        # Média ponderada pela atenção. Produto e soma, não torch.bmm: o gradiente do bmm usa um kernel
        # Triton compilado na hora em algumas versões do PyTorch, que falha sem compilador C.
        local_summary = (local.flatten(2) * weights.unsqueeze(1)).sum(dim=2)
        crops, corners = extract_crops(images, scores)
        encoded = self.crop_encoder(crops.flatten(0, 1)).mean(dim=(-2, -1))
        crop_summary = encoded.reshape(len(images), NUM_CROPS, -1).mean(dim=1)
        self.last_attention, self.last_crops = scores.detach(), corners.detach()
        logits = self.classifier(torch.cat((deep.mean(dim=(-2, -1)), local_summary, crop_summary), dim=1))
        if return_localization:
            return logits, scores.flatten(1).float().topk(NUM_CROPS, dim=1).values.mean(dim=1)
        return logits


class MaculaV2S(nn.Module):
    def __init__(self, pretrained: bool, outputs: int = 1) -> None:
        super().__init__()
        self.backbone = _v2s(pretrained)            # vista global, 512 px
        self.macula = _v2s(pretrained)              # recorte central, resolução nativa
        self.classifier = nn.Linear(2 * self.backbone.num_features, outputs)

    @staticmethod
    def center(images: torch.Tensor) -> torch.Tensor:
        side = images.shape[-1]
        crop = int(side * MACULA_FRACTION)
        top = (side - crop) // 2
        return images[..., top:top + crop, top:top + crop]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        whole = self.backbone(_half(images))                 # num_classes=0: vetor de 1280 (média espacial)
        macula = self.macula(self.center(images))
        return self.classifier(torch.cat((whole, macula), dim=1))


def build(name: str, pretrained: bool, outputs: int) -> nn.Module:
    if name == "hibrida_v2s":
        return HybridV2S(pretrained, outputs)
    if name == "v2s_macula":
        return MaculaV2S(pretrained, outputs)
    raise ValueError(f"rede em duas resoluções desconhecida: {name!r}")
