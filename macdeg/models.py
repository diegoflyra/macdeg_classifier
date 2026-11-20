"""AlexNet escrita camada por camada, com carregamento separado dos pesos ImageNet.

Compatível com a arquitetura e os nomes de parâmetros do torchvision:
https://docs.pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html
"""
from __future__ import annotations

import torch
from torch import nn


ALEXNET_IMAGENET_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"


class AlexNet(nn.Module):
    """Cinco convoluções e três lineares; aceita os pesos IMAGENET1K_V1.

    Os índices de features e classifier são mantidos para carregar também
    checkpoints existentes do projeto. A saída original tem 1000 classes;
    macdeg.train troca essa camada depois de carregar os pesos pré-treinados.
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Bloco 1: características iniciais, com redução espacial pelo stride.
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Bloco 2.
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Bloco 3.
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Blocos 4 e 5: ficam treináveis no fine-tuning parcial.
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = self.avgpool(features)
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)


def load_alexnet_imagenet_weights(model: AlexNet) -> None:
    """Baixa ou reutiliza o checkpoint oficial e exige correspondência completa.

    Deve ser chamado com a saída original de 1000 classes, antes de substituir
    a última camada por um logit. Uma falha no download ou na compatibilidade
    interrompe o carregamento; não há fallback silencioso para pesos aleatórios.
    """
    state = torch.hub.load_state_dict_from_url(
        ALEXNET_IMAGENET_URL, map_location="cpu", progress=True,
        check_hash=True, weights_only=True,
    )
    model.load_state_dict(state, strict=True)


def freeze_alexnet_features(model: nn.Module, freeze_first_convs: int) -> None:
    """Congela as primeiras N convoluções e deixa o restante treinável.

    Contamos convoluções, não índices do Sequential, pois features também
    contém ReLU e pooling. O classificador inteiro continua treinável.
    Não há BatchNorm nem dropout em features na AlexNet do torchvision.
    """
    convolutions = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]
    if type(freeze_first_convs) is not int or not 0 <= freeze_first_convs <= len(convolutions):
        raise ValueError(f"freeze_first_convs deve ser um inteiro entre 0 e {len(convolutions)}.")
    for index, layer in enumerate(convolutions):
        layer.requires_grad_(index >= freeze_first_convs)
    model.classifier.requires_grad_(True)
