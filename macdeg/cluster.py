"""Lotes de experimento: expande a grade, fixa código e dados, roda as tarefas.

Um experimento (configs/experimentos/*.yaml) vira um LOTE em $MACDEG_RUNS/<nome>-<AAAAMMDD-HHMM>/:
  lote.json       configurações, tarefas, commit, assinatura dos dados
  codigo/         cópia do código no commit do lançamento (registro do que rodou)
  tarefas/<configuração>/outer<k>_inner<j>/   saída de cada treino (macdeg.train)
  logs/           saída e erros do job do SLURM
  analise/        o que vai para o GitHub (macdeg.analysis)
Cada tarefa = (configuração, dobra externa, dobra interna). Tarefa concluída nunca roda de novo.
Quem lança: rodar.py (no cluster, dentro de cluster/experimento.sh).
"""
from __future__ import annotations

import copy
import hashlib
import io
import itertools
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import pandas as pd
import yaml

from macdeg.ambiente import PROJECT, commit_info
from macdeg.data.sources import resolve
from macdeg.data.splits import N_INNER, N_OUTER
from macdeg.train import load_config

RUNS = Path(os.environ.get("MACDEG_RUNS", "~/macdeg_runs")).expanduser()
# Modo de teste: valida o encadeamento inteiro em minutos. Os números não servem como resultado.
TEST_OVERRIDES = {"limit": 48, "epochs": 1, "max_steps": 2, "image_size": 128, "num_workers": 0, "patience": 1}
TEST_OUTERS = [0, 1]
# Apuana (sinfo de 2026-11): short-simple = todos os nós e GPUs, até 2 dias, prioridade mais alta, sem
# preempção. O QoS precisa acompanhar a partição (simple ou complex).
def _set(config: dict, key: str, value) -> None:
    *parents, leaf = key.split(".")
    for parent in parents:
        config = config.setdefault(parent, {})
    config[leaf] = value


def expand(experiment: dict) -> list[dict]:
    """Configurações do experimento: a base, cada combinação da grade e cada variação avulsa.
    Nomes: <nome>-00, <nome>-01, ... na ordem (a 00 é a base com a primeira combinação)."""
    base = load_config(PROJECT / experiment["base"], [])
    grid = experiment.get("grade") or {}
    combos = [dict(zip(grid, values)) for values in itertools.product(*grid.values())] or [{}]
    combos += [dict(v) for v in experiment.get("variacoes") or []]
    configs = []
    for i, changes in enumerate(combos):
        config = copy.deepcopy(base)
        for key, value in changes.items():
            _set(config, key, value)
        name = f"{experiment['nome']}-{i:02d}"
        config["name"] = name
        configs.append({"nome": name, "mudancas": changes, "config": config})
    return configs


def dataset_signature(root: Path, splits: str) -> str:
    """Hash dos rótulos, grupos e divisões. Muda se o conjunto ou as divisões mudarem: resultados com
    assinaturas diferentes não são comparáveis."""
    manifest = pd.read_csv(root / "manifest.csv", dtype=str)
    columns = ["sample_id", "uid", "source", "label", "label_detail", "group_id"]
    digest = hashlib.sha256(manifest[columns].sort_values("uid").to_csv(index=False).encode())
    digest.update((root / splits).read_bytes())
    return digest.hexdigest()[:16]


def snapshot_code(destination: Path, commit: str) -> None:
    """Extrai o código do commit (git archive): o que não foi commitado não entra."""
    archive = subprocess.run(["git", "archive", "--format=tar", commit], cwd=PROJECT, capture_output=True, check=True)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        try:
            tar.extractall(destination, filter="data")
        except TypeError:  # Python sem o parâmetro filter (< 3.11.4): o arquivo vem do próprio git archive
            tar.extractall(destination)


def prepare(experiment_path: Path, dataset: str | None = None, test: bool = False, final: list[str] | None = None,
            from_batch: Path | None = None) -> Path:
    """Cria a pasta do lote e o lote.json. `final`: nomes de configurações para o treino final (5 dobras
    externas, sem dobra interna), reaproveitando as configurações de `from_batch`."""
    info = commit_info()
    if final:
        source = json.loads((from_batch / "lote.json").read_text(encoding="utf-8"))
        configs = [c for c in source["configuracoes"] if c["nome"] in final]
        name, experiment = source["nome"] + "-final", source["experimento"]
    else:
        experiment = yaml.safe_load(Path(experiment_path).read_text(encoding="utf-8"))
        configs, name = expand(experiment), experiment["nome"]
    if test:
        name = "teste-" + name
        for c in configs:
            c["config"].update(TEST_OVERRIDES)
    root = resolve(dataset or configs[0]["config"]["dataset"])
    splits = configs[0]["config"].get("splits", "splits_nested_seed0.csv")
    outers = TEST_OUTERS if test else list(range(N_OUTER))
    inners = [None] if final else list(range(N_INNER))
    tasks = [{"id": i, "configuracao": c["nome"], "outer": k, "inner": j}
             for i, (c, k, j) in enumerate(itertools.product(configs, outers, inners))]

    run_name = f"{name}-{time.strftime('%Y%m%d-%H%M')}"
    batch = RUNS / run_name
    (batch / "logs").mkdir(parents=True)
    snapshot_code(batch / "codigo", "HEAD")
    lote = {"nome": run_name, "experimento": experiment, "teste": test, "final": bool(final),
            "lote_de_origem": str(from_batch) if from_batch else None, **info,
            "criado": time.strftime("%Y-%m-%d %H:%M"), "assinatura_dados": dataset_signature(root, splits),
            "splits": splits, "configuracoes": configs, "tarefas": tasks}
    (batch / "lote.json").write_text(json.dumps(lote, indent=1, ensure_ascii=False), encoding="utf-8")
    return batch


def task_dir(batch: Path, task: dict) -> Path:
    inner = f"inner{task['inner']}" if task["inner"] is not None else "final"
    return batch / "tarefas" / task["configuracao"] / f"outer{task['outer']}_{inner}"


def run_task(batch: Path, task_id: int, dataset: str | None = None, wandb_mode: str | None = None) -> dict | None:
    """Executa uma tarefa do lote. Confere a assinatura dos dados."""
    from macdeg.train import run
    batch = Path(batch)
    lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    task = lote["tarefas"][task_id]
    config = copy.deepcopy(next(c for c in lote["configuracoes"] if c["nome"] == task["configuracao"])["config"])
    if dataset:
        config["dataset"] = str(dataset)
    root = resolve(config["dataset"])
    signature = dataset_signature(root, lote["splits"])
    if signature != lote["assinatura_dados"]:
        raise SystemExit(f"O conjunto mudou desde o lançamento ({signature} != {lote['assinatura_dados']}): "
                         "os resultados não seriam comparáveis. Relance o experimento.")
    out = task_dir(batch, task)
    log = on_model = finish = None
    if wandb_mode != "disabled":
        log, on_model, finish = _wandb(lote, task, config, out, wandb_mode)
    try:
        return run(config, task["outer"], task["inner"], seed=0, out=out, log=log, on_model=on_model)
    finally:
        if finish:
            finish()


def _wandb(lote: dict, task: dict, config: dict, out: Path, mode: str | None):
    """W&B: um run por tarefa, agrupado por configuração. Só números (nada de imagens)."""
    from macdeg.ambiente import secret
    if not secret("WANDB_API_KEY") and mode != "offline":
        print("W&B desligado: sem WANDB_API_KEY.")
        return None, None, None
    import wandb
    inner = f"i{task['inner']}" if task["inner"] is not None else "final"
    settings = dict(project=os.environ.get("WANDB_PROJECT", "macdeg_classifier"), entity=secret("WANDB_ENTITY") or None,
                    group=f"{lote['nome']}/{task['configuracao']}", name=f"{task['configuracao']}-o{task['outer']}-{inner}",
                    job_type="final" if task["inner"] is None else "treino", dir=str(out.parent),
                    config={**config, "lote": lote["nome"], "commit": lote["commit"], "outer": task["outer"],
                            "inner": task["inner"], "assinatura_dados": lote["assinatura_dados"], "teste": lote["teste"]},
                    tags=["teste"] if lote["teste"] else [])
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        run = wandb.init(mode=mode or "online", **settings)
    except Exception:  # sem internet no nó: grava local e o job de análise sincroniza depois
        run = wandb.init(mode="offline", **settings)

    def finish():
        metrics = out / "metrics.json"
        if metrics.exists():
            m = json.loads(metrics.read_text())
            run.summary.update({"auroc_val": m["auroc"], "regua_origem_auroc": m["regua_origem_auroc"],
                                "melhor_epoca": m["best_epoch"], "gpu": m["gpu"]})
        run.finish()

    return run.log, (lambda model: run.watch(model, log="all", log_freq=200)), finish


def run_local(batch: Path, dataset: str | None = None, wandb_mode: str | None = "disabled") -> None:
    """Roda as tarefas do lote uma por vez, no próprio processo (dentro do job do cluster ou no computador).
    Tarefas já concluídas são puladas; uma tarefa interrompida continua do last.pt."""
    lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    for task in lote["tarefas"]:
        print(f"tarefa {task['id'] + 1}/{len(lote['tarefas'])}: {task}", flush=True)
        run_task(batch, task["id"], dataset, wandb_mode)
