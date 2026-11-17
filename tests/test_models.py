"""Fine-tuning da AlexNet com tensores sintéticos e sem baixar pesos."""
from __future__ import annotations

import io

import pytest
import torch
import yaml
from torch import nn

from macdeg.train import build_model, build_optimizer


@pytest.mark.parametrize("pretrained", [False, True])
def test_alexnet_selects_imagenet_weights_and_keeps_hidden_layers(monkeypatch, pretrained):
    import torchvision.models as tvm
    original, captured = tvm.alexnet, {}

    def without_download(*, weights):
        model = original(weights=None)
        captured.update(weights=weights, first=model.features[0], hidden=model.classifier[4],
                        output=model.classifier[-1])
        return model

    monkeypatch.setattr(tvm, "alexnet", without_download)
    model = build_model("alexnet", pretrained=pretrained, freeze_first_convs=3)
    assert captured["weights"] == (tvm.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
    assert model.features[0] is captured["first"] and model.classifier[4] is captured["hidden"]
    assert model.classifier[-1] is not captured["output"]
    assert model.classifier[-1].out_features == 1


@pytest.mark.parametrize("size", [128, 224, 512])
def test_alexnet_binary_output_at_smoke_and_research_resolutions(size):
    model = build_model("alexnet", pretrained=False, freeze_first_convs=3).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, size, size))
    assert logits.shape == (1, 1) and torch.isfinite(logits).all()


@pytest.mark.parametrize("count", [0, 3, 5])
def test_alexnet_freezes_only_the_requested_convolutions(count):
    model = build_model("alexnet", pretrained=False, freeze_first_convs=count)
    model.train()  # chamar train() não pode desfazer o congelamento
    convolutions = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]
    assert len(convolutions) == 5
    for index, layer in enumerate(convolutions):
        assert all(p.requires_grad == (index >= count) for p in layer.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


@pytest.mark.parametrize("count", [-1, 6, 1.5, True, "3", None])
def test_invalid_freeze_count_is_rejected(count):
    with pytest.raises(ValueError, match="inteiro entre 0 e 5"):
        build_model("alexnet", pretrained=False, freeze_first_convs=count)


def test_freezing_is_not_silently_ignored_for_other_models():
    with pytest.raises(ValueError, match="model: alexnet"):
        build_model("resnet18", pretrained=False, freeze_first_convs=3)


def test_optimizer_updates_deep_layers_and_preserves_frozen_weights():
    torch.manual_seed(0)
    model = build_model("alexnet", pretrained=False, freeze_first_convs=3).train()
    frozen_before = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    deep_before = model.features[10].weight.detach().clone()
    head_before = model.classifier[-1].weight.detach().clone()
    optimizer = build_optimizer(model, {"lr": 3e-4})
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters() if p.requires_grad}
    logits = model(torch.randn(2, 3, 128, 128)).squeeze(1)
    loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.features[10].weight.grad is not None
    optimizer.step()
    for name, p in model.named_parameters():
        if name in frozen_before:
            assert p.grad is None and torch.equal(p, frozen_before[name])
    assert not torch.equal(model.features[10].weight, deep_before)
    assert not torch.equal(model.classifier[-1].weight, head_before)


def test_xavier_initializes_only_the_new_output():
    torch.manual_seed(7)
    original = build_model("alexnet", pretrained=False)
    first = original.features[0].weight.detach().clone()
    hidden_row = original.classifier[4].weight[0].detach().clone()
    output = original.classifier[-1].weight.detach().clone()
    del original
    torch.manual_seed(7)
    model = build_model("alexnet", pretrained=False, head_init="xavier")
    assert torch.equal(model.features[0].weight, first)
    assert torch.equal(model.classifier[4].weight[0], hidden_row)
    assert not torch.equal(model.classifier[-1].weight, output)
    assert torch.count_nonzero(model.classifier[-1].bias) == 0


@pytest.mark.parametrize("name,init", [("alexnet", "invalid"), ("resnet18", "xavier")])
def test_unsupported_head_initialization_is_rejected(name, init):
    with pytest.raises(ValueError, match="head_init"):
        build_model(name, pretrained=False, head_init=init)


def test_separate_output_rate_excludes_frozen_weights_and_keeps_scheduler_ratio():
    model = build_model("alexnet", pretrained=False, freeze_first_convs=3)
    config = {"model": "alexnet", "lr": 1e-5, "head_lr": 1e-4}
    optimizer = build_optimizer(model, config)
    assert len(optimizer.param_groups) == 2
    base_ids, head_ids = [{id(p) for p in group["params"]} for group in optimizer.param_groups]
    assert head_ids == {id(p) for p in model.classifier[-1].parameters()}
    assert not base_ids & head_ids
    assert base_ids | head_ids == {id(p) for p in model.parameters() if p.requires_grad}
    assert id(model.classifier[4].weight) in base_ids
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        base_lr, head_lr = scheduler.get_last_lr()
        assert head_lr / base_lr == pytest.approx(10)
    resumed = build_optimizer(model, config)
    resumed.load_state_dict(optimizer.state_dict())
    assert [g["lr"] for g in resumed.param_groups] == pytest.approx(scheduler.get_last_lr())


@pytest.mark.parametrize("model,rate", [("resnet18", 1e-4), ("alexnet", -1), ("alexnet", float("nan"))])
def test_invalid_separate_output_rate_is_rejected(model, rate):
    with pytest.raises(ValueError, match="head_lr"):
        build_optimizer(nn.Linear(2, 1), {"model": model, "lr": 1e-5, "head_lr": rate})


def test_finetuned_checkpoint_reloads_without_imagenet_download():
    model = build_model("alexnet", pretrained=False, freeze_first_convs=3).eval()
    images = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        expected = model(images)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    del model
    buffer.seek(0)
    # Inferência não depende de requires_grad; o hub pode usar o construtor padrão.
    restored = build_model("alexnet", pretrained=False).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(images), expected)


def test_finetuning_config_uses_the_current_base_and_one_change():
    """005: base vigente (rodada 3) + AlexNet; a única mudança contra gustavo-01 é o congelamento."""
    from macdeg.cluster import PROJECT, expand
    baseline = yaml.safe_load((PROJECT / "configs/train/baseline.yaml").read_text(encoding="utf-8"))
    path = PROJECT / "configs/experimentos/005_alexnet_congelamento.yaml"
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    configs = expand(experiment)
    assert len(configs) == 1
    config = configs[0]["config"]
    assert config["model"] == "alexnet" and config["pretrained"] is True
    assert config["freeze_first_convs"] == 3
    assert {k for k in baseline if config[k] != baseline[k]} == {"name", "model", "augment"}
    assert config["image_size"] == 512 and config["aux_dr"] == 0.5
    assert experiment["referencia"].endswith("/gustavo-01")


@pytest.mark.parametrize("filename,key,values", [
    ("006_alexnet_taxas.yaml", "head_lr", [None, 1e-4]),
    ("007_alexnet_inicializacao.yaml", "head_init", [None, "xavier"]),
])
def test_finetuning_comparisons_change_one_option(filename, key, values):
    from macdeg.cluster import PROJECT, expand
    experiment = yaml.safe_load((PROJECT / "configs/experimentos" / filename).read_text(encoding="utf-8"))
    first, second = [c["config"] for c in expand(experiment)]
    assert {k for k in first if first[k] != second[k]} == {"name", key}
    assert [first[key], second[key]] == values
    assert first["lr"] == second["lr"] == 1e-5
    assert first["pretrained"] and second["pretrained"]
