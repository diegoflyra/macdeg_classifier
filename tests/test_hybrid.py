"""Redes em duas resoluções (macdeg.hybrid) e explicabilidade (macdeg.explain)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F


def test_crops_copy_native_pixels_and_skip_overlapping_peaks():
    from macdeg.hybrid import extract_crops
    side, crop_side = 128, 24
    image = torch.arange(3 * side * side, dtype=torch.float32).reshape(1, 3, side, side)
    scores = torch.zeros(1, 8, 8)
    scores[0, 0, 0], scores[0, 1, 1], scores[0, 0, -1], scores[0, -1, 0] = 10, 9, 8, 7
    crops, corners = extract_crops(image, scores)
    assert crops.shape == (1, 3, 3, crop_side, crop_side)
    torch.testing.assert_close(crops[:, 0], image[:, :, :crop_side, :crop_side])     # o pico 9 se sobrepõe
    torch.testing.assert_close(crops[:, 1], image[:, :, :crop_side, -crop_side:])
    torch.testing.assert_close(crops[:, 2], image[:, :, -crop_side:, :crop_side])
    assert corners[0, 1].tolist() == [0, side - crop_side]


def test_hybrid_v2s_uses_both_resolutions_pretrained_crop_encoder_and_localization():
    from macdeg.train import build_model, normalization
    model = build_model("hibrida_v2s", pretrained=False, outputs=2).train()
    for a, b in zip(model.crop_encoder[2:].parameters(), model.backbone.blocks[:4].parameters()):
        torch.testing.assert_close(a, b)                     # cópia dos estágios 0-3, pesos independentes
        assert a.data_ptr() != b.data_ptr()
    assert normalization("hibrida_v2s") == normalization("tf_efficientnetv2_s.in21k_ft_in1k")
    shapes = []
    model.backbone.conv_stem.register_forward_pre_hook(lambda m, a: shapes.append(tuple(a[0].shape)))
    model.crop_encoder.register_forward_pre_hook(lambda m, a: shapes.append(tuple(a[0].shape)))
    logits, localization = model(torch.randn(2, 3, 128, 128), return_localization=True)
    assert logits.shape == (2, 2) and localization.shape == (2,)
    assert shapes == [(2, 3, 64, 64), (6, 3, 24, 24)]       # vista global na metade; recortes nativos
    F.binary_cross_entropy_with_logits(localization, torch.tensor([0., 1.])).backward(retain_graph=True)
    assert model.attention.score.weight.grad is not None
    F.binary_cross_entropy_with_logits(logits[:, 0], torch.tensor([0., 1.])).backward()
    assert model.multiscale.alpha.grad is not None and model.crop_encoder[0].weight.grad is not None


def test_macula_v2s_sees_the_center_at_native_resolution():
    from macdeg.hybrid import MaculaV2S
    from macdeg.train import build_model
    model = build_model("v2s_macula", pretrained=False, outputs=1).eval()
    shapes = []
    model.backbone.conv_stem.register_forward_pre_hook(lambda m, a: shapes.append(tuple(a[0].shape)))
    model.macula.conv_stem.register_forward_pre_hook(lambda m, a: shapes.append(tuple(a[0].shape)))
    images = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        assert model(images).shape == (1, 1)
    assert shapes == [(1, 3, 64, 64), (1, 3, 56, 56)]
    torch.testing.assert_close(MaculaV2S.center(images), images[..., 36:92, 36:92])


def test_grad_cam_maps_sum_to_one_and_stats_find_the_macula():
    from macdeg.explain import MAP_SIDE, cam_layers, grad_cam, map_stats
    from macdeg.train import build_model
    model = build_model("resnet18", pretrained=False, outputs=1).eval()
    images = torch.randn(2, 3, 64, 64)
    layer = cam_layers(model, images)["dmri"]
    maps = grad_cam(model, layer, images, 0)
    assert maps.shape == (2, MAP_SIDE, MAP_SIDE)
    assert np.allclose(maps.astype(np.float32).reshape(2, -1).sum(1), 1, atol=1e-2) or maps.max() == 0
    center = np.zeros((1, MAP_SIDE, MAP_SIDE), np.float32)
    center[0, 15:17, 15:17] = 0.25
    corner = np.zeros((1, MAP_SIDE, MAP_SIDE), np.float32)
    corner[0, 0, 0] = 1
    stats = map_stats(np.concatenate([center, corner]))
    assert stats["massa_macula"].tolist() == [pytest.approx(1), 0]
    assert stats["massa_fora_retina"].tolist() == [0, pytest.approx(1)]


def test_hybrid_trains_with_localization_and_writes_explanations(tiny_dataset, tmp_path, monkeypatch):
    from macdeg import train
    from macdeg.explain import FILE, summarize
    monkeypatch.setenv("MACDEG_CACHE", str(tmp_path / "cache"))
    config = {"name": "x", "dataset": str(tiny_dataset), "model": "hibrida_v2s", "pretrained": False,
              "image_size": 128, "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0, "device": "cpu",
              "augment": None, "aux_dr": 0.5, "aux_localization": 0.25, "cache": False, "limit": 40}
    out = tmp_path / "tarefa"
    assert train.run(config, 0, 1, 0, out) is not None
    maps = dict(np.load(out / FILE, allow_pickle=False))
    test = pd.read_csv(out / "test_predictions.csv")
    assert list(maps["sample_id"]) == list(test["sample_id"])
    assert set(maps) == {"sample_id", "dmri", "atencao"}
    summary = summarize([maps], test.assign(pred=(test["prob"] > 0.5).astype(int)), None)
    assert {r["grupo"] for r in summary["dmri"]} >= {"normal"}
    with pytest.raises(ValueError, match="aux_localization"):
        train.run({**config, "model": "resnet18"}, 0, 1, 0, tmp_path / "outra")


def test_hybrid_trains_with_drusas_and_localization_together(tiny_dataset, tmp_path, monkeypatch):
    """Híbrida + saída de drusas + perda de localização (candidata a modelo final): as drusas de treino
    (rótulo -1) ficam fora da perda principal e da de localização; as do teste externo são pontuadas."""
    from macdeg import train
    from macdeg.data.drusas import FOLDS_FILE, assign_folds
    monkeypatch.setenv("MACDEG_CACHE", str(tmp_path / "cache"))
    manifest = pd.read_csv(tiny_dataset / "manifest.csv", dtype={"uid": str})
    splits = pd.read_csv(tiny_dataset / "splits_nested_seed0.csv", dtype={"uid": str})
    extras = pd.read_csv(tiny_dataset / "extras_drusas.csv", dtype={"uid": str})
    links = extras[["uid"]].assign(patient_id=[f"p{i}" for i in range(len(extras))], age=np.arange(len(extras)) * 6 + 45)
    assign_folds(extras, links, manifest, splits).to_csv(tiny_dataset / FOLDS_FILE, index=False)
    config = {"name": "x", "dataset": str(tiny_dataset), "model": "hibrida_v2s", "pretrained": False,
              "image_size": 128, "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0, "device": "cpu",
              "augment": None, "aux_dr": 0.5, "aux_drusas": 0.5, "aux_localization": 0.25, "cache": False}
    assert train.outputs_for(config) == 3
    out = tmp_path / "tarefa"
    assert train.run(config, 0, 1, 0, out) is not None
    scored = pd.read_csv(out / "extra_predictions.csv")
    assert scored["prob"].between(0, 1).all() and scored["prob_drusas"].between(0, 1).all()
    assert np.isfinite(pd.read_csv(out / "test_predictions.csv")["prob"]).all()
