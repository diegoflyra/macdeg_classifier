"""Congelamento das convoluções superficiais da AlexNet do torchvision."""
from __future__ import annotations

from torch import nn


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
