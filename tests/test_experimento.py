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
    assert "--partition=short-simple" in text and "--qos=simple" in text and "--gres=gpu:1" in text
    assert "--error=logs/%x_%j.err" in text and "python rodar.py" in text and "trap 'rm -rf" in text
