import pytest
import torch
from torch.nn import functional as F

from macdeg.hybrid import HYBRID_MODEL, _extract_crops


def test_crops_copy_native_pixels_and_skip_overlapping_peaks():
    side, crop_side = 128, 24
    image = torch.arange(3 * side * side).reshape(1, 3, side, side)
    scores = torch.zeros(1, 8, 8)
    scores[0, 0, 0], scores[0, 1, 1], scores[0, 0, -1], scores[0, -1, 0] = 10, 9, 8, 7

    crops = _extract_crops(image, scores)

    assert crops.shape == (1, 3, 3, crop_side, crop_side)
    torch.testing.assert_close(crops[:, 0], image[:, :, :crop_side, :crop_side])
    torch.testing.assert_close(crops[:, 1], image[:, :, :crop_side, -crop_side:])
    torch.testing.assert_close(crops[:, 2], image[:, :, -crop_side:, :crop_side])


@pytest.mark.parametrize(("name", "backbone_name", "weights_enum", "weights_version"), [
    (HYBRID_MODEL, "resnet18", "ResNet18_Weights", "IMAGENET1K_V1"),
    ("hibrida_dmri_resnet50", "resnet50", "ResNet50_Weights", "IMAGENET1K_V2"),
])
def test_hybrid_preserves_frozen_backbone_and_trains_all_new_modules(
    monkeypatch, name, backbone_name, weights_enum, weights_version
):
    from torchvision import models
    from macdeg.train import build_model, build_optimizer

    backbone = getattr(models, backbone_name)(weights=None)
    imagenet_weights = getattr(getattr(models, weights_enum), weights_version)

    def load_backbone(*, weights):
        assert weights is imagenet_weights
        return backbone

    monkeypatch.setattr(models, backbone_name, load_backbone)
    model = build_model(name, pretrained=True, freeze_first_stages=2, head_init="xavier").train()
    assert model.backbone is backbone
    assert model.multiscale.alpha.item() == pytest.approx(0.1)
    assert not backbone.bn1.training and not backbone.layer2.training
    assert backbone.layer3.training and model.crop_encoder.training
    frozen_mean = backbone.layer2[0].bn1.running_mean.clone()

    config = {"model": name, "lr": 1e-5, "head_lr": 1e-4, "exclude_norm_bias_from_weight_decay": True}
    optimizer = build_optimizer(model, config)
    rates = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
    for parameter_name, parameter in model.named_parameters():
        expected = config["lr"] if parameter_name.startswith("backbone.") else config["head_lr"]
        assert rates.get(id(parameter)) == (expected if parameter.requires_grad else None)

    images = torch.randn(2, 3, 128, 128)
    logits, localization = model(images, return_localization=True)
    assert logits.shape == (2, 1) and localization.shape == (2,)
    labels = torch.tensor([0., 1.])
    F.binary_cross_entropy_with_logits(localization, labels).backward(retain_graph=True)
    assert model.attention.score.weight.grad is not None
    assert model.crop_encoder[0].weight.grad is None
    F.binary_cross_entropy_with_logits(logits[:, 0], labels).backward()
    assert model.crop_encoder[0].weight.grad is not None
    assert model.multiscale.alpha.grad is not None
    assert backbone.layer2[0].conv1.weight.grad is None
    if backbone_name == "resnet50":
        assert model.local_adapter.weight.grad.shape == (128, 512, 1, 1)
        assert model.deep_adapter.weight.grad.shape == (512, 2048, 1, 1)
    torch.testing.assert_close(backbone.layer2[0].bn1.running_mean, frozen_mean)


@pytest.mark.parametrize("name", [HYBRID_MODEL, "hibrida_dmri_resnet50"])
def test_hybrid_inference_uses_both_resolutions_and_reloads_weights(name):
    from macdeg.train import build_model

    trained = build_model(name, pretrained=False, freeze_first_stages=2).eval()
    restored = build_model(name, pretrained=False).eval()
    restored.load_state_dict(trained.state_dict(), strict=True)
    shapes = []

    def record_shape(module, args):
        shapes.append(tuple(args[0].shape))

    restored.backbone.conv1.register_forward_pre_hook(record_shape)
    restored.crop_encoder.register_forward_pre_hook(record_shape)
    images = torch.randn(1, 3, 1024, 1024)
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), trained(images))
    assert shapes == [(1, 3, 512, 512), (3, 3, 192, 192)]
