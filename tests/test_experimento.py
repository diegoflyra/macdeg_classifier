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
    base = {"name": "x", "dataset": str(tiny_dataset), "model": "resnet18", "pretrained": False, "image_size": 64,
            "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0, "device": "cpu", "augment": None}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({"nome": "r", "base": str(tmp_path / "base.yaml")}))
    monkeypatch.setattr(sys, "argv", ["rodar.py", str(tmp_path / "exp.yaml"), "--teste", "--dados", str(tiny_dataset)])
    rodar.main()
    batch = Path((tmp_path / "runs" / "ultimo_lote.txt").read_text())
    assert (batch / "analise" / "RESUMO.md").exists() and (batch / "logs").is_dir()
    lote = json.loads((batch / "lote.json").read_text())
    assert all((cluster.task_dir(batch, t) / "metrics.json").exists() for t in lote["tarefas"])
    # retomar um lote pronto não treina de novo
    monkeypatch.setattr(sys, "argv", ["rodar.py", "--teste", "--retomar", str(batch), "--dados", str(tiny_dataset)])
    rodar.main()


def test_experiment_script_is_valid_bash():
    import subprocess
    from macdeg.ambiente import PROJECT
    script = PROJECT / "cluster" / "experimento.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    text = script.read_text()
    assert "--partition=short-simple" in text and "--qos=simple" in text and "--gres=gpu:2" in text
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
    assert "PILOTO" in (batch / "analise" / "RESUMO.md").read_text()


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


def test_final_model_is_exported_locally_and_loads(tiny_dataset, tmp_path, monkeypatch):
    """--final guarda o ensemble no lote (float16 + cartão com a origem) sem enviar nada ao HF."""
    import macdeg.cluster as cluster
    from macdeg.analysis import analyze
    from macdeg.hub import load_ensemble
    for key in ("GITHUB_TOKEN", "WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cluster, "RUNS", tmp_path / "runs")
    base = {"name": "x", "dataset": str(tiny_dataset), "model": "resnet18", "pretrained": False, "image_size": 64,
            "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0, "device": "cpu", "augment": None}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({"nome": "f", "base": str(tmp_path / "base.yaml")}))
    origin = cluster.prepare(tmp_path / "exp.yaml", str(tiny_dataset), test=True)
    cluster.run_local(origin, str(tiny_dataset))
    analyze(origin, str(tiny_dataset), push=False)

    batch = cluster.prepare(None, str(tiny_dataset), test=True, final=["f-00"], from_batch=origin)
    cluster.run_local(batch, str(tiny_dataset))
    result = analyze(batch, str(tiny_dataset), push=False)
    folder = batch / "modelo" / "f-00"
    assert sorted(p.name for p in folder.glob("*.safetensors")) == [f"dobra{k}.safetensors" for k in cluster.TEST_OUTERS]
    card = json.loads((folder / "config.json").read_text())
    assert card["lote_de_origem"] == origin.name and card["metricas_validacao_aninhada"]["f2_media_bases"] is not None
    assert not list((batch / "tarefas").rglob("best.pt"))                       # só ficam os float16
    assert (batch / "analise" / "modelo_f-00.json").exists()                     # o cartão vai ao GitHub
    assert not list((batch / "analise").rglob("*.safetensors"))                  # os pesos não
    assert result["final"]["f-00"]["modelo"] == str(folder)
    model = load_ensemble(str(batch), "f-00", device="cpu")
    image = cv2.imread(str(tiny_dataset / pd.read_csv(tiny_dataset / "manifest.csv")["image"][0]))
    assert 0.0 <= model.predict(image) <= 1.0 and model.threshold == card["limiar"]


def test_round2_options_dr_target_oversampling_and_clahe(tiny_dataset):
    """B, C e D: alvo de DR só nos "sem DMRI", sorteio ponderado e CLAHE com cache próprio."""
    from macdeg import cache
    from macdeg.train import dr_target, sample_weights
    frame = pd.DataFrame({"label": [1, 0, 0, 0], "disease": [None, "retinopatia_diabetica|glaucoma", "glaucoma", None]})
    assert dr_target(frame).tolist() == [-1, 1, 0, 0]          # DMRI: não se sabe se tem DR
    assert sample_weights(frame, {"retinopatia_diabetica": 3}).tolist() == [1, 3, 1, 1]
    assert sample_weights(frame, None) is None
    manifest = pd.read_csv(tiny_dataset / "manifest.csv")
    plain = cache.build(tiny_dataset, list(manifest["image"][:5]), 48)
    equal = cache.build(tiny_dataset, list(manifest["image"][:5]), 48, preprocess="clahe")
    assert plain[0] != equal[0]                                # caches separados
    a, b = np.load(plain[0]), np.load(equal[0])
    assert a.shape == b.shape and not np.array_equal(a, b)
    with pytest.raises(ValueError):
        cache.prepare(a[0], 48, "inexistente")


def test_round2_variations_train_and_report_discordances(tiny_dataset, tmp_path, monkeypatch):
    """As variações da rodada 2 treinam; a análise traz o veredito e as discordâncias por base/estágio/doença."""
    import macdeg.cluster as cluster
    from macdeg.analysis import analyze
    from macdeg.hub import load_ensemble
    for key in ("GITHUB_TOKEN", "WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cluster, "RUNS", tmp_path / "runs")
    manifest = pd.read_csv(tiny_dataset / "manifest.csv")
    others = manifest["label_detail"] == "outra_doenca"
    manifest.loc[others, "disease"] = np.where(np.arange(others.sum()) % 2, "retinopatia_diabetica", "glaucoma")
    manifest.to_csv(tiny_dataset / "manifest.csv", index=False)
    base = {"name": "x", "dataset": str(tiny_dataset), "model": "resnet18", "pretrained": False, "image_size": 64,
            "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0, "device": "cpu", "augment": None}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    experiment = {"nome": "r2", "base": str(tmp_path / "base.yaml"), "grade": {}, "referencia": None, "variacoes": [
        {"oversample": {"retinopatia_diabetica": 3}}, {"preprocess": "clahe"}, {"aux_dr": 0.5},
        {"image_size": 96, "cache": False}]}
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(experiment))
    batch = cluster.prepare(tmp_path / "exp.yaml", str(tiny_dataset), test=True)
    for c in json.loads((batch / "lote.json").read_text())["configuracoes"][1:]:
        assert len(c["mudancas"]) >= 1                         # cada variação = a base + a sua mudança
    cluster.run_local(batch, str(tiny_dataset))
    result = analyze(batch, str(tiny_dataset), push=False)
    assert result["tarefas_faltando"] == [] and result["referencia"] == "r2-00"
    for name in ("r2-01", "r2-02", "r2-03", "r2-04"):
        rows = result["configuracoes"][name]["discordancias"]
        groups = {r["grupo"] for r in rows}
        assert {"geral", "base", "doenca"} <= groups
        geral = next(r for r in rows if r["grupo"] == "geral")
        assert geral["dmri_so_referencia"] + geral["dmri_so_nova"] + geral["dmri_nenhuma"] <= geral["n_dmri"]
    assert (batch / "analise" / "discordancias.csv").exists()
    assert "Discordâncias com a referência" in (batch / "analise" / "RESUMO.md").read_text()

    # o modelo com cabeça auxiliar vira um ensemble utilizável (só a saída de DMRI conta)
    final = cluster.prepare(None, str(tiny_dataset), test=True, final=["r2-03"], from_batch=batch)
    cluster.run_local(final, str(tiny_dataset))
    analyze(final, str(tiny_dataset), push=False)
    model = load_ensemble(str(final), "r2-03", device="cpu")
    assert model.card["saidas"] == 2
    image = cv2.imread(str(tiny_dataset / manifest["image"][0]))
    assert 0.0 <= model.predict(image) <= 1.0


def test_round2_experiment_file_changes_one_thing_per_variation():
    from macdeg.ambiente import PROJECT
    from macdeg.cluster import expand
    experiment = yaml.safe_load((PROJECT / "configs/experimentos/002_rodada2.yaml").read_text())
    configs = expand(experiment)
    assert [c["nome"] for c in configs] == [f"rodada2-{i:02d}" for i in range(6)]
    assert configs[0]["mudancas"] == {}                         # a referência retreinada
    technical = {"cache"}                                      # não muda o conteúdo do treino
    assert all(len(set(c["mudancas"]) - technical) == 1 for c in configs[1:])
    assert configs[5]["config"]["image_size"] == 1024 and configs[3]["config"]["preprocess"] == "clahe"


def test_cache_is_skipped_when_the_disk_is_full(tiny_dataset, tmp_path, monkeypatch):
    """Disco sem espaço: o cache não é criado e o treino segue lendo os JPEGs (como no nó com /tmp cheio)."""
    from collections import namedtuple
    from macdeg import cache
    from macdeg.train import run
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(cache.shutil, "disk_usage", lambda path: Usage(10**9, 10**9, 10**6))
    monkeypatch.setenv("MACDEG_CACHE", str(tmp_path / "cache"))
    manifest = pd.read_csv(tiny_dataset / "manifest.csv")
    assert cache.build(tiny_dataset, list(manifest["image"]), 64) is None
    assert not list((tmp_path / "cache").glob("*.npy"))
    config = {"name": "x", "dataset": str(tiny_dataset), "model": "resnet18", "pretrained": False, "image_size": 64,
              "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0, "device": "cpu", "augment": None, "limit": 24}
    run(config, 0, 0, seed=0, out=tmp_path / "t")
    assert (tmp_path / "t" / "metrics.json").exists()


def test_job_script_avoids_the_mig_node_and_checks_disk_space():
    from macdeg.ambiente import PROJECT
    text = (PROJECT / "cluster" / "experimento.sh").read_text()
    assert "--exclude=cluster-node6" in text and "df -Pk" in text and "menos de 5 GB livres" in text


def test_two_jobs_share_a_batch_without_training_the_same_task(tiny_dataset, tmp_path, monkeypatch, capsys):
    """Segundo job no mesmo lote (--retomar --do-fim): começa pela última tarefa, pula a que outro job marcou
    como em andamento, retoma marca abandonada e deixa a análise para quem terminar por último."""
    import os
    import sys
    import time
    import macdeg.cluster as cluster
    import rodar
    for key in ("GITHUB_TOKEN", "WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cluster, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(rodar, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(rodar, "load_dotenv", lambda *a, **k: None)
    base = {"name": "x", "dataset": str(tiny_dataset), "model": "resnet18", "pretrained": False, "image_size": 64,
            "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0, "device": "cpu", "augment": None}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({"nome": "d", "base": str(tmp_path / "base.yaml")}))
    batch = cluster.prepare(tmp_path / "exp.yaml", str(tiny_dataset), test=True)
    lote = json.loads((batch / "lote.json").read_text())
    busy, abandoned = (cluster.task_dir(batch, t) for t in lote["tarefas"][:2])
    for folder in (busy, abandoned):
        folder.mkdir(parents=True)
        (folder / "em_andamento").write_text("outro job")
    old = time.time() - cluster.STALE_SECONDS - 60
    os.utime(abandoned / "em_andamento", (old, old))

    monkeypatch.setattr(sys, "argv", ["rodar.py", "--teste", "--retomar", str(batch), "--do-fim",
                                      "--dados", str(tiny_dataset)])
    rodar.main()
    out = capsys.readouterr().out
    order = [int(line.split()[1].split("/")[0]) for line in out.splitlines() if line.startswith("tarefa ") and "/" in line.split()[1]]
    assert order == sorted(order, reverse=True)                       # do fim para o começo
    assert not (busy / "metrics.json").exists() and (busy / "em_andamento").exists()   # não mexeu na do outro
    assert (abandoned / "metrics.json").exists() and not (abandoned / "em_andamento").exists()
    assert "a análise fica com ele" in out and not (batch / "analise").exists()

    (busy / "em_andamento").unlink()                                  # o outro job termina a dele...
    monkeypatch.setattr(sys, "argv", ["rodar.py", "--teste", "--retomar", str(batch), "--dados", str(tiny_dataset)])
    rodar.main()                                                      # ...e, sendo o último, analisa
    assert (batch / "analise" / "RESUMO.md").exists() and not (batch / "analise_em_andamento").exists()


def test_round3_center_crop_alexnet_normalization_and_micro_batch(tiny_dataset, tmp_path, monkeypatch):
    """Rodada 3: recorte central (cache próprio, mesmos pixels que o JPEG), AlexNet, normalização de cada
    rede e micro-lote igual em qualquer GPU (o BatchNorm também)."""
    import types
    import torch
    from macdeg import cache, train
    image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    assert cache.center_crop(image, 0.6).shape == (60, 60, 3)
    assert np.array_equal(cache.center_crop(image, 0.6), image[20:80, 20:80])
    with pytest.raises(ValueError):
        cache.center_crop(image, 1.5)
    monkeypatch.setenv("MACDEG_CACHE", str(tmp_path / "cache"))
    manifest = pd.read_csv(tiny_dataset / "manifest.csv")
    images = list(manifest["image"][:5])
    plain = cache.build(tiny_dataset, images, 32)
    cropped = cache.build(tiny_dataset, images, 32, crop=0.6)
    assert plain[0] != cropped[0] and not np.array_equal(np.load(plain[0]), np.load(cropped[0]))
    frame = manifest.head(5)
    identity = lambda x: x
    from_cache = train.FundusDataset(frame, tiny_dataset, identity, cropped, 32, None, 0.6)
    from_jpeg = train.FundusDataset(frame, tiny_dataset, identity, None, 32, None, 0.6)
    assert torch.equal(from_cache[2][0], from_jpeg[2][0])

    alexnet = train.build_model("alexnet", pretrained=False, outputs=2)
    assert alexnet(torch.zeros(1, 3, 64, 64)).shape == (1, 2)
    assert train.normalization("resnet50") == (train.IMAGENET_MEAN, train.IMAGENET_STD)
    assert train.normalization("tf_efficientnetv2_s.in21k_ft_in1k") == ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    cuda = types.SimpleNamespace(type="cuda")
    micro = {}
    for gb in (24, 80):
        monkeypatch.setattr(train.torch.cuda, "get_device_properties", lambda d, g=gb: types.SimpleNamespace(total_memory=g * 1e9))
        micro[gb] = [train.micro_batch({"batch_size": 16, "image_size": 512, "model": m}, cuda)
                     for m in ("resnet50", "tf_efficientnet_b4.ns_jft_in1k")]
    assert micro[24] == micro[80] == [16, 8]


def test_stage_auroc_compares_each_stage_with_the_negatives_of_its_bases():
    from macdeg.analysis import stage_auroc
    preds = pd.DataFrame({"sample_id": list("abcdefg"), "source": ["h"] * 5 + ["o"] * 2,
                          "label": [1, 1, 0, 0, 1, 1, 0], "prob": [0.9, 0.2, 0.3, 0.1, 0.8, 0.95, 0.99]})
    manifest = pd.DataFrame({"sample_id": list("abcdefg"),
                             "label_detail": ["precoce", "precoce", "dr", "dr", "tardia", "dmri", "normal"]})
    out = stage_auroc(preds, manifest)
    assert set(out) == {"precoce", "tardia"}                       # "dmri" sem estágio fica de fora
    assert out["precoce"] == {"n": 2, "n_sem_dmri": 2, "bases": ["h"], "auroc": 0.75}
    assert out["tardia"]["auroc"] == 1.0                          # o negativo da outra base não entra


def test_round3_experiment_file_has_a_new_base_and_one_change_per_variation():
    from macdeg.ambiente import PROJECT
    from macdeg.cluster import expand
    experiment = yaml.safe_load((PROJECT / "configs/experimentos/003_rodada3.yaml").read_text())
    configs = expand(experiment)
    assert [c["nome"] for c in configs] == [f"rodada3-{i:02d}" for i in range(5)]
    assert configs[0]["mudancas"] == {}
    assert all(len(c["mudancas"]) == 1 for c in configs[1:])
    for c in configs:                                              # a base nova vale para todas
        assert c["config"]["aux_dr"] == 0.5 and c["config"]["augment"]["rotate_deg"] == 15
        assert c["config"]["image_size"] == 512
    assert configs[4]["config"]["crop"] == 0.6 and configs[4]["config"]["model"] == "resnet50"


def test_round3_base_changes_crop_and_alexnet_train_and_export(tiny_dataset, tmp_path, monkeypatch):
    """base_mudancas vale para todas as configurações; recorte e AlexNet treinam; o ensemble final do
    recorte guarda o recorte e a normalização no cartão e prevê com eles."""
    import macdeg.cluster as cluster
    from macdeg.analysis import analyze
    from macdeg.hub import load_ensemble
    for key in ("GITHUB_TOKEN", "WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cluster, "RUNS", tmp_path / "runs")
    monkeypatch.setenv("MACDEG_CACHE", str(tmp_path / "cache"))
    base = {"name": "x", "dataset": str(tiny_dataset), "model": "resnet18", "pretrained": False, "image_size": 64,
            "batch_size": 4, "epochs": 1, "lr": 1e-3, "num_workers": 0, "device": "cpu", "augment": None}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    experiment = {"nome": "r3", "base": str(tmp_path / "base.yaml"), "grade": {}, "referencia": None,
                  "base_mudancas": {"aux_dr": 0.5, "augment": {"hflip": True}},
                  "variacoes": [{"crop": 0.6}, {"model": "alexnet"}]}
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(experiment))
    batch = cluster.prepare(tmp_path / "exp.yaml", str(tiny_dataset), test=True)
    configs = json.loads((batch / "lote.json").read_text())["configuracoes"]
    assert all(c["config"]["aux_dr"] == 0.5 for c in configs) and configs[0]["mudancas"] == {}
    cluster.run_local(batch, str(tiny_dataset))
    result = analyze(batch, str(tiny_dataset), push=False)
    assert result["tarefas_faltando"] == [] and set(result["configuracoes"]) == {"r3-00", "r3-01", "r3-02"}
    assert "auroc_por_estagio" in result["configuracoes"]["r3-01"]

    final = cluster.prepare(None, str(tiny_dataset), test=True, final=["r3-01"], from_batch=batch)
    cluster.run_local(final, str(tiny_dataset))
    analyze(final, str(tiny_dataset), push=False)
    model = load_ensemble(str(final), "r3-01", device="cpu")
    assert model.card["crop"] == 0.6 and model.card["normalizacao"]["media"] == [0.485, 0.456, 0.406]
    image = cv2.imread(str(tiny_dataset / pd.read_csv(tiny_dataset / "manifest.csv")["image"][0]))
    assert 0.0 <= model.predict(image) <= 1.0
