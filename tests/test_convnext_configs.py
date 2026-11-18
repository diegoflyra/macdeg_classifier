"""Receitas de cluster: fatores independentes e custo explícito, sem dados de pacientes."""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def test_cluster_recipe_preserves_the_data_and_evaluation_inputs():
    baseline = read("configs/train/baseline.yaml")
    config = read("configs/train/convnext_cluster.yaml")
    for key in ("dataset", "splits", "sources", "other_diseases", "image_size", "batch_size",
                "balance", "augment", "shuffle_labels"):
        assert config[key] == baseline[key]
    assert config["pretrained"] is True and config["augment"] is None
    assert config["gradient_checkpointing"] is True
    assert config["batch_size"] % config["micro_batch"] == 0
    assert config["micro_batch"] == 2
    assert config["head_lr"] == 10 * config["lr"]
    assert config["exclude_norm_bias_from_weight_decay"] is True
    assert config["max_grad_norm"] > 0


@pytest.mark.parametrize("filename,configurations,models", [
    ("007_convnext_arquiteturas.yaml", 3, ["convnext_tiny", "convnext_small", "convnext_base"]),
    ("008_convnext_ajustes.yaml", 6, None),
    ("009_convnext_large.yaml", 2, ["convnext_base", "convnext_large"]),
])
def test_cluster_experiments_change_one_factor_and_bound_the_training_count(filename, configurations, models):
    experiment = read("configs/experimentos/" + filename)
    assert experiment["base"] == "configs/train/convnext_cluster.yaml"
    assert experiment["referencia"] is None
    if models is not None:
        assert experiment["grade"] == {"model": models}
        assert len(models) == configurations
        assert experiment["variacoes"] == []
    else:
        assert experiment["grade"] == {}
        changes = experiment["variacoes"]
        assert len(changes) + 1 == configurations
        assert all(len(change) == 1 for change in changes)
        assert len({tuple(change.items()) for change in changes}) == len(changes)
        assert {next(iter(change)) for change in changes} == {
            "freeze_first_stages", "lr", "weight_decay", "stochastic_depth_prob",
        }


def test_convnext_augmentation_batches_share_the_round_two_recipe():
    base = "configs/train/convnext_cluster.yaml"
    isolated = read("configs/experimentos/010_convnext_augmentation.yaml")
    architectures = read("configs/experimentos/011_convnext_arquiteturas_augmentation.yaml")
    assert isolated["base"] == architectures["base"] == base
    assert isolated["referencia"] is None and architectures["referencia"] is None
    assert read(base)["augment"] is None and isolated["grade"] == {}
    assert isolated["variacoes"] == [{"augment": {
        "crop_scale": [0.85, 1.0], "hflip": True, "rotate_deg": 15,
        "rotate_interpolation": "bilinear", "color": 0.1,
    }}]
    assert architectures["grade"] == {
        "model": ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"],
        "augment": [isolated["variacoes"][0]["augment"]],
    }
    assert architectures["variacoes"] == []


def test_aux_dr_batches_reuse_augmentation_and_weight_from_round_two():
    augmentation = read("configs/experimentos/011_convnext_arquiteturas_augmentation.yaml")
    paired = read("configs/experimentos/012_convnext_aux_dr.yaml")
    architectures = read("configs/experimentos/013_convnext_arquiteturas_aux_dr.yaml")
    base = augmentation["base"]
    assert paired["base"] == architectures["base"] == base
    assert read(base).get("aux_dr") is None
    assert paired["grade"] == {"augment": augmentation["grade"]["augment"], "aux_dr": [None, 0.5]}
    assert architectures["grade"] == {**augmentation["grade"], "aux_dr": [0.5]}
    for experiment in (paired, architectures):
        assert experiment["variacoes"] == [] and experiment["referencia"] is None
