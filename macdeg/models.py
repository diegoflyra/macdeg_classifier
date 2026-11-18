"""AlexNet escrita camada por camada, com carregamento separado dos pesos ImageNet.

Compatível com a arquitetura e os nomes de parâmetros do torchvision:
https://docs.pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html
"""
from __future__ import annotations

import torch
from torch import nn

from macdeg import layers


ALEXNET_IMAGENET_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"


class AlexNet(nn.Module):
    """Cinco convoluções e três lineares; aceita os pesos IMAGENET1K_V1.

    Os índices de features e classifier são mantidos para carregar também
    checkpoints existentes do projeto. A saída original tem 1000 classes;
    macdeg.train troca essa camada depois de carregar os pesos pré-treinados.
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.ModuleList([
            # Bloco 1: características iniciais, com redução espacial pelo stride.
            layers.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            layers.ReLU(inplace=True),
            layers.MaxPool2d(kernel_size=3, stride=2),
            # Bloco 2.
            layers.Conv2d(64, 192, kernel_size=5, padding=2),
            layers.ReLU(inplace=True),
            layers.MaxPool2d(kernel_size=3, stride=2),
            # Bloco 3.
            layers.Conv2d(192, 384, kernel_size=3, padding=1),
            layers.ReLU(inplace=True),
            # Blocos 4 e 5: ficam treináveis no fine-tuning parcial.
            layers.Conv2d(384, 256, kernel_size=3, padding=1),
            layers.ReLU(inplace=True),
            layers.Conv2d(256, 256, kernel_size=3, padding=1),
            layers.ReLU(inplace=True),
            layers.MaxPool2d(kernel_size=3, stride=2),
        ])
        self.avgpool = layers.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.ModuleList([
            layers.Dropout(p=dropout),
            layers.Linear(256 * 6 * 6, 4096),
            layers.ReLU(inplace=True),
            layers.Dropout(p=dropout),
            layers.Linear(4096, 4096),
            layers.ReLU(inplace=True),
            layers.Linear(4096, num_classes),
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Bloco 1: convolução, ReLU e max pooling.
        x = self.features[0](images)
        x = self.features[1](x)
        x = self.features[2](x)

        # Bloco 2: convolução, ReLU e max pooling.
        x = self.features[3](x)
        x = self.features[4](x)
        x = self.features[5](x)

        # Blocos 3, 4 e 5; o último termina com max pooling.
        x = self.features[6](x)
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        # Duas densas ocultas, com dropout, e a saída.
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)
        return self.classifier[6](x)


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

    Contamos convoluções, não índices da lista, pois features também
    contém ReLU e pooling. O classificador inteiro continua treinável.
    Não há BatchNorm nem dropout em features na AlexNet do torchvision.
    """
    convolutions = [layer for layer in model.features if isinstance(layer, layers.Conv2d)]
    if type(freeze_first_convs) is not int or not 0 <= freeze_first_convs <= len(convolutions):
        raise ValueError(f"freeze_first_convs deve ser um inteiro entre 0 e {len(convolutions)}.")
    for index, layer in enumerate(convolutions):
        layer.requires_grad_(index >= freeze_first_convs)
    model.classifier.requires_grad_(True)
