"""ConvNeXt manual: pesos compatíveis, congelamento e treino com tensores sintéticos."""
from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn

from macdeg import layers
from macdeg.convnext import (
    CONVNEXT_TINY_IMAGENET_URL, CONVNEXT_IMAGENET_URLS, CONVNEXT_MODELS,
    ConvNeXt, ConvNeXtTiny, freeze_convnext_stages,
    load_convnext_tiny_imagenet_weights, load_convnext_imagenet_weights,
)
from macdeg.train import build_model, build_optimizer


@pytest.mark.parametrize("size", [32, 224])
def test_convnext_accepts_official_checkpoint_and_matches_outputs(size):
    import torchvision.models as tvm
    reference = tvm.convnext_tiny(weights=None).eval()
    model = ConvNeXtTiny().eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    assert sum(p.numel() for p in model.parameters()) == 28_589_128
    images = torch.randn(1, 3, size, size)
    with torch.no_grad():
        torch.testing.assert_close(model(images), reference(images))


@pytest.mark.parametrize("size", [128, 512])
def test_convnext_binary_output_at_pipeline_resolutions(size):
    model = build_model("convnext_tiny", pretrained=False, freeze_first_stages=2, head_init="xavier").eval()
    assert isinstance(model.classifier[-1], layers.Linear)
    assert model.classifier[-1].out_features == 1
    assert not any(isinstance(m, (nn.Conv2d, nn.Linear, nn.LayerNorm, nn.Sequential)) for m in model.modules())
    with torch.no_grad():
        logits = model(torch.randn(1, 3, size, size))
    assert logits.shape == (1, 1) and torch.isfinite(logits).all()


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_convnext_freezing_survives_train_and_disables_frozen_stage_noise(count):
    model = build_model("convnext_tiny", pretrained=False, freeze_first_stages=count)
    model.train().train()
    for index, feature in enumerate(model.features):
        expected = index >= 2 * count
        assert all(p.requires_grad == expected for p in feature.parameters())
        assert feature.training == expected
        assert all(m.training == expected for m in feature.modules() if isinstance(m, layers.StochasticDepth))
    assert model.classifier.training and all(p.requires_grad for p in model.classifier.parameters())
    model.eval()
    assert not any(module.training for module in model.modules())


@pytest.mark.parametrize("count", [-1, 5, True, 1.5])
def test_convnext_rejects_invalid_freeze_counts(count):
    with pytest.raises(ValueError, match="freeze_first_stages"):
        build_model("convnext_tiny", pretrained=False, freeze_first_stages=count)


def test_stage_freezing_is_not_silently_ignored_for_other_models():
    with pytest.raises(ValueError, match="freeze_first_stages"):
        build_model("alexnet", pretrained=False, freeze_first_stages=2)


def test_convnext_weights_are_selected_before_replacing_the_output(monkeypatch):
    import macdeg.convnext as convnext
    calls = []

    def offline(model):
        assert model.classifier[-1].out_features == 1000
        calls.append(True)
        with torch.no_grad():
            model.features[0][0].weight.fill_(0.125)
            model.classifier[0].bias.fill_(0.25)

    monkeypatch.setattr(convnext, "load_convnext_tiny_imagenet_weights", offline)
    model = build_model("convnext_tiny", pretrained=True, freeze_first_stages=2, head_init="xavier")
    assert calls == [True]
    assert torch.all(model.features[0][0].weight == 0.125)
    assert torch.all(model.classifier[0].bias == 0.25)
    assert torch.count_nonzero(model.classifier[-1].bias) == 0


def test_convnext_loader_uses_official_url_and_strict_state_loading(monkeypatch):
    model = ConvNeXtTiny()
    checkpoint = model.state_dict()
    checkpoint["features.0.0.weight"] = torch.full_like(checkpoint["features.0.0.weight"], 0.125)
    checkpoint["classifier.2.bias"] = torch.full_like(checkpoint["classifier.2.bias"], 0.25)
    calls = []

    def offline(url, **kwargs):
        calls.append((url, kwargs))
        return checkpoint

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", offline)
    load_convnext_tiny_imagenet_weights(model)
    assert calls == [(CONVNEXT_TINY_IMAGENET_URL, {"map_location": "cpu", "progress": True,
                                               "check_hash": True, "weights_only": True})]
    assert torch.all(model.features[0][0].weight == 0.125)
    assert torch.all(model.classifier[-1].bias == 0.25)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: {})
    with pytest.raises(RuntimeError):
        load_convnext_tiny_imagenet_weights(model)


def test_convnext_download_failure_does_not_fall_back_to_random_weights(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("download indisponível no teste")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", unavailable)
    with pytest.raises(OSError, match="download indisponível"):
        build_model("convnext_tiny", pretrained=True)


def test_convnext_backward_and_optimizer_preserve_frozen_parameters():
    torch.manual_seed(0)
    # Sem stochastic depth neste teste para medir uma atualização sem sorteio de ramos.
    model = ConvNeXtTiny(num_classes=1, stochastic_depth_prob=0)
    freeze_convnext_stages(model, 2)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    scale = model.features[7][0].layer_scale
    before_scale = scale.detach().clone()
    before_head = model.classifier[-1].weight.detach().clone()
    optimizer = build_optimizer(model, {"model": "convnext_tiny", "lr": 1e-5, "head_lr": 1e-4})
    base_ids, head_ids = [{id(p) for p in g["params"]} for g in optimizer.param_groups]
    assert head_ids == {id(p) for p in model.classifier[-1].parameters()}
    assert id(model.classifier[0].weight) in base_ids
    assert not base_ids & head_ids
    assert base_ids | head_ids == {id(p) for p in model.parameters() if p.requires_grad}
    loss = nn.BCEWithLogitsLoss()(model(torch.randn(2, 3, 32, 32)).squeeze(1), torch.tensor([0., 1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert scale.grad is not None and torch.isfinite(scale.grad).all()
    optimizer.step()
    for name, p in model.named_parameters():
        if name in frozen:
            assert p.grad is None and torch.equal(p, frozen[name])
    assert not torch.equal(scale, before_scale)
    assert not torch.equal(model.classifier[-1].weight, before_head)


def test_convnext_checkpoint_reloads_for_hub_inference():
    model = build_model("convnext_tiny", pretrained=False, freeze_first_stages=2).eval()
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = model(images)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    del model
    buffer.seek(0)
    restored = build_model("convnext_tiny", pretrained=False).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(images), expected)


def test_convnext_experiments_preserve_recipe_and_isolate_freezing():
    from macdeg.cluster import PROJECT, expand
    experiment = yaml.safe_load((PROJECT / "configs/experimentos/005_convnext_tiny_finetuning.yaml").read_text(encoding="utf-8"))
    configs = expand(experiment)
    assert len(configs) == 1
    config = configs[0]["config"]
    assert config["pretrained"] and config["freeze_first_stages"] == 2
    assert config["image_size"] == 512 and config["augment"] is None
    assert config["lr"] == 1e-5 and config["head_lr"] == 1e-4
    comparison = yaml.safe_load((PROJECT / "configs/experimentos/006_convnext_congelamento.yaml").read_text(encoding="utf-8"))
    first, second = [item["config"] for item in expand(comparison)]
    assert {key for key in first if first[key] != second[key]} == {"name", "freeze_first_stages"}
    assert [first["freeze_first_stages"], second["freeze_first_stages"]] == [0, 2]


@pytest.mark.parametrize("name,count", [("convnext_small", 50_223_688), ("convnext_base", 88_591_464)])
def test_larger_variants_match_official_checkpoints_and_outputs(name, count):
    import torchvision.models as tvm
    reference = getattr(tvm, name)(weights=None).eval()
    model = CONVNEXT_MODELS[name]().eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    assert sum(p.numel() for p in model.parameters()) == count
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(images), reference(images))


def test_large_matches_official_checkpoint_layout_and_binary_head():
    import torchvision.models as tvm
    with torch.device("meta"):
        reference = tvm.convnext_large(weights=None)
        model = CONVNEXT_MODELS["convnext_large"]()
        model.load_state_dict(reference.state_dict(), strict=True)
        assert sum(p.numel() for p in model.parameters()) == 197_767_336
        assert CONVNEXT_IMAGENET_URLS["convnext_large"] == tvm.ConvNeXt_Large_Weights.IMAGENET1K_V1.url
        binary = build_model("convnext_large", pretrained=False, freeze_first_stages=2, head_init="xavier").eval()
        assert sum(p.numel() for p in binary.parameters()) == 196_231_873
        assert binary(torch.empty(2, 3, 32, 32)).shape == (2, 1)
    optimizer = build_optimizer(binary, {"model": "convnext_large", "lr": 1e-5, "head_lr": 1e-4})
    assert optimizer.param_groups[-1]["lr"] == 1e-4


@pytest.mark.parametrize("name", CONVNEXT_MODELS)
def test_each_variant_downloads_its_own_checkpoint(name, monkeypatch):
    calls = []

    class Receiver:
        variant = name

        def load_state_dict(self, state, strict):
            assert state is marker and strict is True

    marker = object()

    def download(url, **kwargs):
        calls.append((url, kwargs))
        return marker

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", download)
    load_convnext_imagenet_weights(Receiver())
    assert calls == [(CONVNEXT_IMAGENET_URLS[name], {"map_location": "cpu", "progress": True,
                                                   "check_hash": True, "weights_only": True})]


class MiniConvNeXt(ConvNeXt):
    """Mesmo fluxo, com tensores pequenos para testar gradientes e checkpointing."""
    depths = (1, 1, 1, 1)
    widths = (4, 8, 16, 32)


@pytest.mark.parametrize("frozen", [0, 2])
def test_checkpointing_preserves_outputs_gradients_and_random_masks(frozen):
    torch.manual_seed(11)
    model = MiniConvNeXt(num_classes=1, stochastic_depth_prob=0.3)
    freeze_convnext_stages(model, frozen)
    images = torch.randn(2, 3, 32, 32)
    torch.manual_seed(17)
    expected = model(images)
    expected.square().sum().backward()
    gradients = {n: p.grad.clone() if p.grad is not None else None for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing = True
    torch.manual_seed(17)
    actual = model(images)
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    for name, parameter in model.named_parameters():
        if gradients[name] is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradients[name])


def test_checkpointing_only_runs_on_trainable_blocks_in_training(monkeypatch):
    import macdeg.convnext as convnext
    original, calls = convnext.checkpoint, []

    def recording(block, inputs, **kwargs):
        assert kwargs == {"use_reentrant": False, "preserve_rng_state": True}
        calls.append(block)
        return original(block, inputs, **kwargs)

    monkeypatch.setattr(convnext, "checkpoint", recording)
    model = MiniConvNeXt(num_classes=1, gradient_checkpointing=True)
    freeze_convnext_stages(model, 2)
    images = torch.randn(2, 3, 32, 32)
    model(images).sum().backward()
    assert calls == [model.features[5][0], model.features[7][0]]
    calls.clear()
    with torch.no_grad():
        model(images)
    model.eval()(images)
    assert calls == []


def test_weight_decay_groups_cover_trainable_parameters_once_with_correct_rates():
    model = MiniConvNeXt(num_classes=1)
    freeze_convnext_stages(model, 2)
    optimizer = build_optimizer(model, {"model": "convnext_tiny", "lr": 1e-5, "head_lr": 1e-4,
                                        "weight_decay": 0.01, "exclude_norm_bias_from_weight_decay": True})
    memberships = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            assert id(parameter) not in memberships
            memberships[id(parameter)] = group
    assert set(memberships) == {id(p) for p in model.parameters() if p.requires_grad}
    head_ids = {id(p) for p in model.classifier[-1].parameters()}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = memberships[id(parameter)]
        excluded = parameter.ndim <= 1 or name.endswith(".bias") or name.endswith(".layer_scale")
        assert group["weight_decay"] == (0 if excluded else 0.01)
        assert group["lr"] == (1e-4 if id(parameter) in head_ids else 1e-5)
    resumed = build_optimizer(model, {"model": "convnext_tiny", "lr": 1e-5, "head_lr": 1e-4,
                                      "weight_decay": 0.01, "exclude_norm_bias_from_weight_decay": True})
    resumed.load_state_dict(optimizer.state_dict())
    assert [g["lr"] for g in resumed.param_groups] == [g["lr"] for g in optimizer.param_groups]


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_stochastic_depth_is_rejected(value):
    with pytest.raises(ValueError, match="stochastic_depth_prob"):
        MiniConvNeXt(stochastic_depth_prob=value)


@pytest.mark.parametrize("value", [0, -1, 3, 32, True, 2.5])
def test_invalid_microbatch_is_rejected(value):
    from macdeg.train import micro_batch
    with pytest.raises(ValueError, match="micro_batch"):
        micro_batch({"batch_size": 16, "micro_batch": value}, torch.device("cpu"))


def test_microbatch_is_fixed_across_cpu_and_gpu_and_preserves_effective_batch():
    from macdeg.train import micro_batch
    config = {"batch_size": 16, "micro_batch": 2}
    assert micro_batch(config, torch.device("cpu")) == micro_batch(config, torch.device("cuda")) == 2


def test_convnext_cluster_expansion_has_expected_configurations_and_cost():
    from macdeg.cluster import PROJECT, expand
    for filename, count in [("007_convnext_arquiteturas.yaml", 3), ("008_convnext_ajustes.yaml", 6)]:
        experiment = yaml.safe_load((PROJECT / "configs/experimentos" / filename).read_text(encoding="utf-8"))
        configs = expand(experiment)
        assert len(configs) == count
        assert all(c["config"]["augment"] is None and c["config"]["image_size"] == 512 for c in configs)
        assert all(c["config"]["micro_batch"] == 2 and c["config"]["batch_size"] == 16 for c in configs)
        first = configs[0]["config"]
        for candidate in configs[1:]:
            assert len({k for k in first if first[k] != candidate["config"][k]} - {"name"}) == 1


def test_dr_targets_follow_manifest_tokens_and_ignore_dmri_and_drusas():
    from macdeg.train import dr_target

    frame = pd.DataFrame({"label": [0, 0, 0, 0, 1, -1],
                          "disease": ["glaucoma|retinopatia_diabetica", "retinopatia_hipertensiva", None,
                                      "retinopatia_diabetica_provavel", "retinopatia_diabetica", None]})
    assert dr_target(frame).tolist() == [1., 0., 0., 0., -1., -1.]


@pytest.mark.parametrize("targets", [[1., -1., 0.], [-1., -1., -1.]])
def test_dr_loss_matches_round_two_mask_and_gradients(targets):
    from macdeg.train import dr_loss

    target = torch.tensor(targets)
    logits = torch.tensor([-0.8, 3., 0.4], requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    known = target >= 0
    expected = nn.functional.binary_cross_entropy_with_logits(reference_logits[known], target[known]) if known.any() \
        else reference_logits.sum() * 0
    actual = dr_loss(logits, target)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


@pytest.mark.parametrize("weight", [None, 0.5])
def test_hub_roundtrip_returns_only_dmri_with_and_without_aux_head(tmp_path, monkeypatch, weight):
    import huggingface_hub
    from macdeg import hub
    from macdeg.cluster import task_dir
    from macdeg.train import outputs_for, predict

    name = "convnext_tiny"
    monkeypatch.setitem(CONVNEXT_MODELS, name, MiniConvNeXt)
    config = {"model": name, "image_size": 32, "aux_dr": weight}
    outputs = outputs_for(config)
    model = build_model(name, pretrained=False, outputs=outputs)
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.copy_(torch.tensor([-2., 8.][:outputs]))
    batch = tmp_path / "batch"
    task = {"configuracao": name, "outer": 0, "inner": None}
    folder = task_dir(batch, task)
    folder.mkdir(parents=True)
    torch.save(model.state_dict(), folder / "best.pt")
    lote = {"nome": "teste-final", "commit": "teste", "assinatura_dados": "teste", "lote_de_origem": str(tmp_path),
            "configuracoes": [{"nome": name, "config": config, "mudancas": {}}], "tarefas": [task]}
    (batch / "lote.json").write_text(json.dumps(lote), encoding="utf-8")
    snapshot = tmp_path / "snapshot"

    class OfflineAPI:
        def whoami(self):
            return {"name": "teste"}

        def create_repo(self, repo, *, private, exist_ok):
            assert private and exist_ok

        def upload_folder(self, *, repo_id, folder_path, commit_message):
            shutil.copytree(folder_path, snapshot)

    monkeypatch.setattr(hub, "_api", OfflineAPI)
    monkeypatch.setattr(hub, "secret", lambda key: None)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: snapshot / "missing.json")
    hub.upload_final(batch, name, {"limiar": 0.5})
    card_path = Path(snapshot) / name / "config.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["saidas"] == outputs
    if weight is None:  # Cartões antigos não registravam o número de saídas.
        del card["saidas"]
        card_path.write_text(json.dumps(card), encoding="utf-8")
    ensemble = hub.load_ensemble("teste/macdeg_classifier", name, device="cpu")
    assert ensemble.models[0].classifier[-1].out_features == outputs
    image = np.full((32, 32, 3), 127, np.uint8)
    expected = torch.sigmoid(torch.tensor(-2.)).item()
    assert ensemble.predict(image) == pytest.approx(expected)
    images = torch.zeros(2, 3, 32, 32)
    assert predict(model, [(images, torch.zeros(2))], torch.device("cpu")).tolist() == pytest.approx([expected] * 2)
