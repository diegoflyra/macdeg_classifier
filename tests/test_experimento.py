"""Ponta a ponta, com dados sintéticos: lançamento (modo de teste, local) -> tarefas -> análise."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from macdeg.data.splits import assign_nested


@pytest.fixture()
def tiny_dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(150):
        source = ["a", "b", "rfmid2"][i % 3]
        detail = ["dmri", "normal", "normal", "outra_doenca"][(i // 3) % 4]
        uid = f"{i:020x}"
        image = np.full((64, 64, 3), 40, np.uint8)
        cv2.circle(image, (32, 32), 28, (60, 90, 160), -1)
        if detail == "dmri":  # "drusas" claras: o modelo tem algo a aprender
            for _ in range(6):
                cv2.circle(image, tuple(int(v) for v in rng.integers(20, 44, 2)), 2, (200, 220, 240), -1)
        cv2.imwrite(str(root / "images" / f"{uid}.jpg"), image)
        rows.append({"uid": uid, "source": source, "label": int(detail == "dmri"), "label_detail": detail,
                     "disease": "glaucoma" if detail == "outra_doenca" else None,
                     "disease_group": "glaucoma" if detail == "outra_doenca" else None,
                     "patient_id": None, "group_id": f"g{i}", "age": 50 + 20 * (detail == "dmri") + i % 7,
                     "image": f"images/{uid}.jpg"})
    manifest = pd.DataFrame(rows).sort_values("uid").reset_index(drop=True)
    manifest["sample_id"] = [f"img_{i:05d}" for i in range(len(manifest))]
    manifest.to_csv(root / "manifest.csv", index=False)
    assign_nested(manifest, seed=0).to_csv(root / "splits_nested_seed0.csv", index=False)
    drusas = manifest.head(10)[["uid", "source", "image"]].assign(sample_id=[f"dru_{i:05d}" for i in range(10)],
                                                                   group_id=[f"d{i}" for i in range(10)])
    drusas.to_csv(root / "extras_drusas.csv", index=False)
    return root


def test_launch_tasks_and_analysis_end_to_end(tiny_dataset, tmp_path, monkeypatch):
    import macdeg.cluster as cluster
    from macdeg.analysis import analyze
    for key in ("GITHUB_TOKEN", "WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cluster, "RUNS", tmp_path / "runs")
    base = {"name": "x", "dataset": str(tiny_dataset), "splits": "splits_nested_seed0.csv", "model": "resnet18",
            "pretrained": False, "image_size": 64, "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0,
            "balance": "pos_weight", "device": "cpu", "augment": None}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    experiment = {"nome": "e2e", "base": str(tmp_path / "base.yaml"), "grade": {"lr": [1e-3, 1e-2]},
                  "variacoes": [], "referencia": None}
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(experiment))

    batch = cluster.prepare(tmp_path / "exp.yaml", str(tiny_dataset), test=True)
    lote = json.loads((batch / "lote.json").read_text())
    assert [c["nome"] for c in lote["configuracoes"]] == ["e2e-00", "e2e-01"]
    assert len(lote["tarefas"]) == 2 * len(cluster.TEST_OUTERS) * 4
    assert (batch / "codigo" / "README.md").exists()        # código fixado no commit (só o que está commitado)
    cluster.run_local(batch, str(tiny_dataset))
    result = analyze(batch, str(tiny_dataset), push=False)
    assert result["tarefas_faltando"] == []
    assert result["referencia"] == "e2e-00"
    assert result["configuracoes"]["e2e-01"]["veredito"]["veredito"] in {"ganhou", "empatou", "perdeu"}
    assert set(result["aninhado"]["escolhida_por_dobra"]) == set(cluster.TEST_OUTERS)
    tipos = {d["tipo"] for d in result["configuracoes"]["e2e-00"]["distribuicao_notas"]}
    assert {"dmri", "normal", "outra", "drusas"} <= tipos
    assert result["configuracoes"]["e2e-00"]["regua_idade_auroc"] is not None

    out = batch / "analise"
    assert {"RESUMO.md", "relatorio.json", "ranking.csv", "historico.csv", "lote.json"} <= {p.name for p in out.iterdir()}
    for path in out.glob("previsoes_externas_*.csv"):     # só IDs nossos saem do cluster
        preds = pd.read_csv(path)
        assert preds["sample_id"].str.match(r"^img_\d{5}$").all() and "uid" not in preds
    assert "tarefas" not in json.loads((out / "lote.json").read_text())

    # Tarefa repetida não treina de novo; conjunto alterado é recusado
    assert cluster.run_task(batch, 0, str(tiny_dataset), "disabled") is None
    manifest = pd.read_csv(tiny_dataset / "manifest.csv")
    manifest.loc[0, "label"] = 1 - manifest.loc[0, "label"]
    manifest.to_csv(tiny_dataset / "manifest.csv", index=False)
    with pytest.raises(SystemExit):
        cluster.run_task(batch, 1, str(tiny_dataset), "disabled")


def test_logs_are_not_pushed_for_test_batches(tmp_path):
    from macdeg.analysis import push_logs
    batch = tmp_path / "lote"
    (batch / "logs").mkdir(parents=True)
    (batch / "logs" / "tarefa_0.out").write_text("AUROC 0.8\n")
    (batch / "lote.json").write_text(json.dumps({"nome": "x", "teste": True}))
    assert push_logs(batch) is False          # lote de teste: nada vai para o GitHub


@pytest.mark.parametrize(("filename", "model_name", "loss_weights"), [
    ("014_hibrida_dmri.yaml", "hibrida_dmri", [0.0, 0.25]),
    ("015_hibrida_resnet50.yaml", "hibrida_dmri_resnet50", [0.25]),
])
def test_hybrid_pilot_trains_and_analyzes(tiny_dataset, tmp_path, monkeypatch, filename, model_name, loss_weights):
    from macdeg import cluster
    from macdeg.analysis import analyze

    monkeypatch.setattr(cluster, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(cluster, "TEST_OVERRIDES", {
        **cluster.TEST_OVERRIDES, "pretrained": False, "device": "cpu", "batch_size": 4,
    })
    experiment = cluster.PROJECT / "configs/experimentos" / filename
    batch = cluster.prepare(experiment, str(tiny_dataset), test=True, pilot=True)
    lote = json.loads((batch / "lote.json").read_text())
    assert [c["config"]["aux_localization"] for c in lote["configuracoes"]] == loss_weights
    assert {c["config"]["model"] for c in lote["configuracoes"]} == {model_name}
    assert len(lote["tarefas"]) == len(loss_weights)

    cluster.run_local(batch, str(tiny_dataset), gpus=[])

    for task in lote["tarefas"]:
        folder = cluster.task_dir(batch, task)
        assert (folder / "metrics.json").is_file()
        assert pd.read_csv(folder / "test_predictions.csv")["prob"].between(0, 1).all()
    report = analyze(batch, str(tiny_dataset), push=False)
    assert report["tarefas_faltando"] == []
    assert report["referencia"] == f"{model_name}-00"


def test_rodar_runs_a_whole_test_experiment(tiny_dataset, tmp_path, monkeypatch):
    """O rodar.py (o que o cluster/experimento.sh chama) roda tarefas e análise num processo só."""
    import sys
    import macdeg.cluster as cluster
    import rodar
    for key in ("GITHUB_TOKEN", "WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cluster, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(rodar, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(rodar, "load_dotenv", lambda *a, **k: None)   # o teste não usa as chaves reais
    pointer = tmp_path / "macdeg_1.lote"
    monkeypatch.setenv("MACDEG_JOB_LOTE", str(pointer))
    base = {"name": "x", "dataset": str(tiny_dataset), "model": "resnet18", "pretrained": False, "image_size": 64,
            "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0, "device": "cpu", "augment": None}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({"nome": "r", "base": str(tmp_path / "base.yaml")}))
    monkeypatch.setattr(sys, "argv", ["rodar.py", str(tmp_path / "exp.yaml"), "--teste", "--dados", str(tiny_dataset)])
    rodar.main()
    batch = Path((tmp_path / "runs" / "ultimo_lote.txt").read_text())
    assert Path(pointer.read_text(encoding="utf-8")) == batch
    assert (batch / "analise" / "RESUMO.md").exists() and (batch / "logs").is_dir()
    lote = json.loads((batch / "lote.json").read_text())
    assert all((cluster.task_dir(batch, t) / "metrics.json").exists() for t in lote["tarefas"])
    from macdeg import analysis

    def analyze_twice(*args, **kwargs):
        pytest.fail("um segundo job tentou analisar um lote já analisado")

    monkeypatch.setattr(analysis, "analyze", analyze_twice)
    monkeypatch.setattr(sys, "argv", ["rodar.py", "--retomar", str(batch), "--do-fim", "--dados", str(tiny_dataset)])
    rodar.main()


def test_single_gpu_jobs_share_tasks_from_opposite_ends(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from macdeg import cluster, train

    tasks = [{"id": i, "configuracao": "shared-00", "outer": 0, "inner": i} for i in range(4)]
    lote = {"tarefas": tasks, "splits": "splits.csv", "assinatura_dados": "test",
            "configuracoes": [{"nome": "shared-00", "config": {"dataset": str(tmp_path)}}]}
    (tmp_path / "lote.json").write_text(json.dumps(lote), encoding="utf-8")
    monkeypatch.setattr(cluster, "dataset_signature", lambda *args: "test")
    first_started, finish_first = Event(), Event()
    trained = []

    def train_once(config, outer, inner, *, out, **kwargs):
        trained.append(inner)
        if inner == 0:
            first_started.set()
            assert finish_first.wait(timeout=10)
        (out / "metrics.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(train, "run", train_once)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(cluster.run_local, tmp_path, gpus=["0"])
        try:
            assert first_started.wait(timeout=10)
            cluster.run_local(tmp_path, gpus=["0"], from_end=True)
            assert trained == [0, 3, 2, 1]
            assert cluster.pending_tasks(tmp_path, lote) == [0]
        finally:
            finish_first.set()
        first.result(timeout=10)
    assert trained == [0, 3, 2, 1]
    assert cluster.pending_tasks(tmp_path, lote) == []


def test_resumed_job_waits_for_all_tasks_before_analysis(tmp_path, monkeypatch):
    import sys

    import rodar
    from macdeg import analysis, cluster

    batch = tmp_path / "lote"
    batch.mkdir()
    task = {"id": 0, "configuracao": "shared-00", "outer": 0, "inner": 0}
    lote = {"tarefas": [task], "configuracoes": [{}], "teste": True}
    (batch / "lote.json").write_text(json.dumps(lote), encoding="utf-8")
    pointer = tmp_path / "macdeg_2.lote"
    monkeypatch.setenv("MACDEG_JOB_LOTE", str(pointer))
    monkeypatch.setattr(sys, "argv", ["rodar.py", "--retomar", str(batch), "--do-fim"])
    monkeypatch.setattr(rodar, "RUNS", tmp_path)
    monkeypatch.setattr(rodar, "load_dotenv", lambda: None)
    monkeypatch.setattr(rodar, "check", lambda *args: None)
    directions, analyses = [], []
    monkeypatch.setattr(rodar, "run_local", lambda *args, from_end: directions.append(from_end))
    monkeypatch.setattr(analysis, "analyze", lambda batch, *args, push: analyses.append((batch, push)))

    rodar.main()
    assert analyses == []
    assert Path(pointer.read_text(encoding="utf-8")) == batch

    out = cluster.task_dir(batch, task)
    out.mkdir(parents=True)
    (out / "metrics.json").write_text("{}", encoding="utf-8")
    rodar.main()
    rodar.main()
    assert directions == [True, True, True]
    assert analyses == [(batch, False)]


def test_experiment_script_is_valid_bash():
    import subprocess
    from macdeg.ambiente import PROJECT
    script = PROJECT / "cluster" / "experimento.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    text = script.read_text()
    assert "--partition=short-simple" in text and "--qos=simple" in text and "#SBATCH --gres=gpu:1" in text
    assert "--error=logs/%x_%j.err" in text and "python rodar.py" in text and "trap 'rm -rf" in text


def test_pilot_is_one_real_training_and_analysis_handles_it(tiny_dataset, tmp_path, monkeypatch):
    import macdeg.cluster as cluster
    from macdeg.analysis import analyze
    for key in ("GITHUB_TOKEN", "WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cluster, "RUNS", tmp_path / "runs")
    base = {"name": "x", "dataset": str(tiny_dataset), "model": "resnet18", "pretrained": False, "image_size": 64,
            "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0, "device": "cpu", "augment": None}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({"nome": "p", "base": str(tmp_path / "base.yaml")}))
    batch = cluster.prepare(tmp_path / "exp.yaml", str(tiny_dataset), pilot=True)
    lote = json.loads((batch / "lote.json").read_text())
    assert lote["piloto"] and lote["tarefas"] == [{"id": 0, "configuracao": "p-00", "outer": 0, "inner": 0}]
    assert "epochs" not in cluster.TEST_OVERRIDES or lote["configuracoes"][0]["config"]["epochs"] == 1  # sem cortes de teste
    cluster.run_local(batch, str(tiny_dataset))
    result = analyze(batch, str(tiny_dataset), push=False)
    assert result["piloto"] and result["configuracoes"]["p-00"]["dobras_externas"] == [0]
    resources = result["recursos_piloto"]["p-00"]
    assert resources["gpu"] == "cpu" and resources["epochs_run"] == 1
    assert resources["task_seconds"] >= resources["seconds_per_epoch"] >= 0
    assert resources["peak_reserved_gpu_gb"] is None and resources["gpu_total_gb"] is None
    summary = (batch / "analise" / "RESUMO.md").read_text(encoding="utf-8")
    assert "PILOTO" in summary and "Tempo e memória do piloto" in summary


def test_image_cache_gives_the_same_pixels_as_reading_the_jpeg(tiny_dataset, tmp_path, monkeypatch):
    """O cache só acelera: o tensor que chega ao modelo é idêntico ao da leitura direta do JPEG."""
    import torch
    from macdeg import cache
    from macdeg.train import FundusDataset, transforms
    monkeypatch.setenv("MACDEG_CACHE", str(tmp_path / "cache"))
    manifest = pd.read_csv(tiny_dataset / "manifest.csv")
    # imagem maior que o tamanho do treino, como no conjunto real (1024 -> 512)
    big = cv2.resize(cv2.imread(str(tiny_dataset / manifest.loc[0, "image"])), (100, 100))
    cv2.imwrite(str(tiny_dataset / manifest.loc[0, "image"]), big)
    config = {"image_size": 48, "augment": None}
    first = cache.build(tiny_dataset, list(manifest["image"][:20]), 48, workers=2)
    assert len(first[1]) == 20
    again = cache.build(tiny_dataset, list(manifest["image"][:10]), 48)           # reaproveita, não refaz
    assert again == first
    plain = FundusDataset(manifest, tiny_dataset, transforms(config, False))
    cached = FundusDataset(manifest, tiny_dataset, transforms(config, False), first)
    for i in (0, 5, 19, 30):                                   # 30 está fora do cache: lê o JPEG
        assert torch.equal(plain[i][0], cached[i][0])
    assert cache.build(tiny_dataset, list(manifest["image"][:21]), 48) == first      # 1 faltando: vem do JPEG
    grown = cache.build(tiny_dataset, list(manifest["image"][15:40]), 48, workers=1)   # muitas faltando: amplia
    assert len(grown[1]) == 40 and grown[0] != first[0] and first[0].exists()


def test_tasks_are_split_between_two_gpus(tiny_dataset, tmp_path, monkeypatch):
    """Duas "GPUs" (aqui, dois processos na CPU) dividem as tarefas; o lote termina inteiro."""
    import macdeg.cluster as cluster
    for key in ("GITHUB_TOKEN", "WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cluster, "RUNS", tmp_path / "runs")
    monkeypatch.setenv("MACDEG_CACHE", str(tmp_path / "cache"))
    base = {"name": "x", "dataset": str(tiny_dataset), "model": "resnet18", "pretrained": False, "image_size": 64,
            "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0, "device": "cpu", "augment": None}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({"nome": "g", "base": str(tmp_path / "base.yaml")}))
    batch = cluster.prepare(tmp_path / "exp.yaml", str(tiny_dataset), test=True)
    cluster.run_local(batch, str(tiny_dataset), gpus=["", ""])
    lote = json.loads((batch / "lote.json").read_text())
    assert all((cluster.task_dir(batch, t) / "metrics.json").exists() for t in lote["tarefas"])
    history = pd.read_csv(cluster.task_dir(batch, lote["tarefas"][0]) / "history.csv")
    assert {"train_seconds", "data_wait_seconds"} <= set(history.columns)


def test_alexnet_finetuning_runs_the_existing_training_pipeline(tiny_dataset, tmp_path, monkeypatch):
    """Uma atualização real e inferência; seleção ImageNet com download substituído."""
    import macdeg.models as models
    from macdeg.train import PREDICTION_COLUMNS, run
    weights_requested = []

    def without_download(model):
        assert model.classifier[-1].out_features == 1000
        weights_requested.append(True)

    monkeypatch.setattr(models, "load_alexnet_imagenet_weights", without_download)
    monkeypatch.setenv("MACDEG_CACHE", str(tmp_path / "cache"))
    config = {"name": "alexnet_finetuning_smoke", "dataset": str(tiny_dataset), "model": "alexnet",
              "pretrained": True, "freeze_first_convs": 3,
              "image_size": 128, "batch_size": 4, "epochs": 1, "max_steps": 1,
              "limit": 24, "lr": 1e-5, "head_lr": 1e-4, "head_init": "xavier",
              "num_workers": 0, "balance": "pos_weight", "device": "cpu",
              "augment": None}
    target = tmp_path / "alexnet"
    metrics = run(config, outer=0, inner=0, seed=0, out=target)
    assert weights_requested == [True]
    assert metrics["epochs_run"] == 1
    assert metrics["frozen_parameters"] > 0 and metrics["trainable_parameters"] > 0
    history = pd.read_csv(target / "history.csv")
    assert np.allclose(history["head_lr"] / history["lr"], 10)
    assert np.isfinite(history["train_loss"]).all()
    for split in ("val", "test", "extra"):
        predictions = pd.read_csv(target / f"{split}_predictions.csv")
        assert list(predictions.columns) == PREDICTION_COLUMNS + ["prob"]
        assert predictions["prob"].between(0, 1).all()
    assert not (target / "last.pt").exists() and not (target / "best.pt").exists()


@pytest.mark.parametrize("optimized, aux_dr", [(False, False), (True, False), (True, True)])
def test_convnext_finetuning_runs_the_existing_training_pipeline(tiny_dataset, tmp_path, monkeypatch, optimized, aux_dr):
    """Uma atualização da ConvNeXt e previsão nas três saídas, só com dados sintéticos."""
    import macdeg.convnext as convnext
    from macdeg.train import PREDICTION_COLUMNS, run
    requested = []
    clipped = []
    heads = []

    def offline_weights(model):
        assert model.classifier[-1].out_features == 1000
        requested.append(True)

    monkeypatch.setattr(convnext, "load_convnext_tiny_imagenet_weights", offline_weights)
    monkeypatch.setenv("MACDEG_CACHE", str(tmp_path / "cache"))
    config = {"name": "convnext_smoke", "dataset": str(tiny_dataset), "model": "convnext_tiny",
              "pretrained": True, "freeze_first_stages": 2, "head_init": "xavier",
              "image_size": 64, "batch_size": 4, "epochs": 1, "max_steps": 1, "limit": 24,
              "lr": 1e-5, "head_lr": 1e-4, "num_workers": 0, "balance": "pos_weight",
              "device": "cpu", "augment": None}
    if optimized:
        from macdeg.cluster import PROJECT, expand
        experiment_path = PROJECT / "configs/experimentos/012_convnext_aux_dr.yaml"
        configs = expand(yaml.safe_load(experiment_path.read_text(encoding="utf-8")))
        candidate = configs[int(aux_dr)]["config"]
        config.update(micro_batch=2, gradient_checkpointing=True, max_grad_norm=1.0,
                      exclude_norm_bias_from_weight_decay=True, warmup_epochs=0,
                      augment=candidate["augment"], aux_dr=candidate["aux_dr"])
        import torch
        original_clip = torch.nn.utils.clip_grad_norm_

        def record_clip(parameters, max_norm):
            clipped.append(max_norm)
            return original_clip(parameters, max_norm)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    if aux_dr:
        manifest_path = tiny_dataset / "manifest.csv"
        manifest = pd.read_csv(manifest_path)
        manifest.loc[manifest["disease"] == "glaucoma", "disease"] = "retinopatia_diabetica"
        manifest.to_csv(manifest_path, index=False)
    target = tmp_path / "convnext"
    metrics = run(config, outer=0, inner=0, seed=0, out=target,
                  on_model=lambda model: heads.append(model.classifier[-1].out_features))
    assert requested == [True] and metrics["epochs_run"] == 1
    assert heads == [2 if aux_dr else 1]
    assert metrics["gradient_checkpointing"] == optimized
    assert metrics["effective_batch_size"] == 4
    assert clipped == ([1.0] if optimized else [])
    assert metrics["frozen_parameters"] == 1_235_040
    history = pd.read_csv(target / "history.csv")
    assert np.isfinite(history["train_loss"]).all()
    assert history["optimizer_steps"].tolist() == [1]
    assert np.allclose(history["head_lr"] / history["lr"], 10)
    for split in ("val", "test", "extra"):
        predictions = pd.read_csv(target / f"{split}_predictions.csv")
        assert list(predictions.columns) == PREDICTION_COLUMNS + ["prob"]
        assert predictions["prob"].between(0, 1).all()
    assert not (target / "last.pt").exists() and not (target / "best.pt").exists()


def test_convnext_augmentation_is_train_only_and_uses_bilinear_rotation():
    import torch
    from torchvision.transforms import v2
    from macdeg.cluster import PROJECT, expand
    from macdeg.train import transforms

    experiment_path = PROJECT / "configs/experimentos/010_convnext_augmentation.yaml"
    control, candidate = [entry["config"] for entry in expand(yaml.safe_load(experiment_path.read_text(encoding="utf-8")))]
    expected_changes = {key for key in control if control[key] != candidate[key]}
    assert expected_changes == {"name", "augment"}
    control["image_size"] = candidate["image_size"] = 64
    training = transforms(candidate, train=True)
    assert isinstance(training.transforms[0], v2.RandomResizedCrop)
    assert tuple(training.transforms[0].scale) == (0.85, 1.0)
    assert isinstance(training.transforms[1], v2.RandomHorizontalFlip)
    rotation = training.transforms[2]
    assert tuple(rotation.degrees) == (-15.0, 15.0)
    assert rotation.interpolation == v2.InterpolationMode.BILINEAR
    assert isinstance(training.transforms[3], v2.ColorJitter)

    image = (torch.arange(3 * 64 * 64).reshape(3, 64, 64) % 256).to(torch.uint8)
    reference = transforms(control, train=False)(image)
    torch.testing.assert_close(transforms(candidate, train=False)(image), reference, rtol=0, atol=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        augmented = training(image)
    assert augmented.shape == (3, 64, 64) and torch.isfinite(augmented).all()
    assert not torch.equal(augmented, reference)

    candidate["augment"] = {k: v for k, v in candidate["augment"].items() if k != "rotate_interpolation"}
    assert transforms(candidate, train=True).transforms[2].interpolation == v2.InterpolationMode.NEAREST

