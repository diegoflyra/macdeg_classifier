"""Lança um experimento (grade de configurações x dobras) como job array do SLURM, ou localmente.

Uso no nó de login do Apuana:
  python -m macdeg.cluster lancar configs/experimentos/001_referencia.yaml [--teste] [--local]
  python -m macdeg.cluster final  <pasta do lote> <configuração> [--local]

Um experimento vira um LOTE em $MACDEG_RUNS/<nome>-<AAAAMMDD-HHMM>/:
  lote.json       configurações, tarefas, commit, assinatura dos dados
  codigo/         cópia do código no commit do lançamento (todas as tarefas usam a mesma versão)
  tarefas/<configuração>/outer<k>_inner<j>/   saída de cada treino (macdeg.train)
  notebooks/      cópia executada do notebook de cada tarefa
  logs/           saída do SLURM
  analise/        o que vai para o GitHub (macdeg.analysis)
Cada tarefa = (configuração, dobra externa, dobra interna). Uma tarefa concluída nunca roda de novo.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import itertools
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import pandas as pd
import yaml

from macdeg.ambiente import PROJECT, commit_info, repo_url
from macdeg.data.sources import resolve
from macdeg.data.splits import N_INNER, N_OUTER
from macdeg.train import load_config

RUNS = Path(os.environ.get("MACDEG_RUNS", "~/macdeg_runs")).expanduser()
# Modo de teste: valida o encadeamento inteiro em minutos. Os números não servem como resultado.
TEST_OVERRIDES = {"limit": 48, "epochs": 1, "max_steps": 2, "image_size": 128, "num_workers": 0, "patience": 1}
TEST_OUTERS = [0, 1]
# Apuana (sinfo de 2026-11): short-simple = todos os nós e GPUs, até 2 dias, prioridade mais alta, sem
# preempção. O QoS precisa acompanhar a partição (simple ou complex).
# Apuana (guia do Notion + scontrol, 2026-11): short-simple = até 2 dias, sem preempção. Cada QoS limita o
# usuário: simple = 2 GPUs, 16 CPUs, 64 GB; complex = 4 GPUs, 48 CPUs, 500 GB. O QoS acompanha a partição.
SLURM_DEFAULTS = {"partition": "short-simple", "qos": "simple", "cpus": 6, "mem": "32G", "time": "12:00:00",
                  "particao_analise": "short-simple", "qos_analise": "simple"}
QOS_MAX_GPUS = {"simple": 2, "complex": 4}
# Ambiente: módulo Python do cluster + venv (cluster/setup_apuana.sh grava o módulo usado em MODULO_PYTHON)
VENV = Path(os.environ.get("MACDEG_VENV", "~/envs/macdeg")).expanduser()


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
    (batch / "notebooks").mkdir()
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
    """Executa uma tarefa do lote (chamado pelo notebook de tarefa). Confere a assinatura dos dados."""
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


# Início comum dos jobs: diagnóstico (para entender falhas pelo log) e ambiente do guia do Apuana.
SBATCH_PRELUDE = """set -eo pipefail
echo "== $(date '+%Y-%m-%d %H:%M:%S') | nó $(hostname) | job ${{SLURM_JOB_ID}} tarefa ${{SLURM_ARRAY_TASK_ID:-}}"
type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
# Módulo Python só se o venv foi criado com um (cluster/setup_apuana.sh); senão é o Python do sistema
if [ -s {venv}/MODULO_PYTHON ]; then module load "$(cat {venv}/MODULO_PYTHON)"; fi
source {venv}/bin/activate
export PYTHONPATH={lote}/codigo MACDEG_RUNS={runs} MACDEG_ENV={env} MACDEG_REPO_URL={repo}
# Pasta temporária no disco local do nó (NVMe); apagada no fim, mesmo se o job falhar ou for cancelado.
TMPBASE="${{SLURM_TMPDIR:-/tmp}}/$USER-macdeg-$SLURM_JOB_ID-${{SLURM_ARRAY_TASK_ID:-0}}"
trap 'rm -rf "$TMPBASE"' EXIT TERM INT
mkdir -p "$TMPBASE"
DADOS="$TMPBASE/dataset"
"""

SBATCH_ARRAY = """#!/bin/bash
#SBATCH --job-name={nome}
#SBATCH --array=0-{ultimo}%{max_paralelo}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --requeue
#SBATCH --output={lote}/logs/tarefa_%a.out
""" + SBATCH_PRELUDE + """
nvidia-smi -L || echo "nvidia-smi falhou"
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU')"
echo "== copiando o conjunto para $DADOS"
python -m zipfile -e {zip} "$DADOS"
papermill {lote}/codigo/notebooks/03_tarefa.ipynb {lote}/notebooks/tarefa_$SLURM_ARRAY_TASK_ID.ipynb \\
  -p LOTE {lote} -p TAREFA $SLURM_ARRAY_TASK_ID -p DADOS "$DADOS" --cwd {lote}/codigo
echo "== fim $(date '+%Y-%m-%d %H:%M:%S')"
"""

SBATCH_ANALYSIS = """#!/bin/bash
#SBATCH --job-name={nome}-analise
#SBATCH --partition={particao_analise}
#SBATCH --qos={qos_analise}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output={lote}/logs/analise.out
""" + SBATCH_PRELUDE + """
# Runs do W&B gravados offline (nó sem internet) são enviados agora.
find {lote}/tarefas -type d -name "offline-run-*" -exec wandb sync {{}} \\; 2>/dev/null || true
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2], ['manifest.csv', 'splits_nested_seed0.csv'])" {zip} "$DADOS"
set +e
papermill {lote}/codigo/notebooks/04_analise.ipynb {lote}/notebooks/analise.ipynb \\
  -p LOTE {lote} -p DADOS "$DADOS" --cwd {lote}/codigo
STATUS=$?
# Notebooks executados (tarefas e análise) vão para o GitHub mesmo se a análise falhou: ajudam a depurar.
python -m macdeg.analysis notebooks {lote}
exit $STATUS
"""


def write_scripts(batch: Path, zip_path: str, slurm: dict | None = None) -> tuple[Path, Path]:
    lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    values = {**SLURM_DEFAULTS, **(slurm or {}), "nome": lote["nome"][:40], "ultimo": len(lote["tarefas"]) - 1,
              "lote": str(batch), "zip": str(Path(zip_path).expanduser()), "runs": str(RUNS),
              "env": str(PROJECT / ".env"), "repo": repo_url(), "venv": str(VENV)}
    # Mais tarefas simultâneas que o QoS permite só ficariam presas na fila
    values.setdefault("max_paralelo", QOS_MAX_GPUS.get(values["qos"], 2))
    array, analysis = batch / "job_tarefas.sh", batch / "job_analise.sh"
    array.write_text(SBATCH_ARRAY.format(**values))
    analysis.write_text(SBATCH_ANALYSIS.format(**values))
    return array, analysis


def submit(batch: Path, zip_path: str, slurm: dict | None = None) -> str:
    """Submete o array e, depois dele (com ou sem falhas), a análise."""
    array, analysis = write_scripts(batch, zip_path, slurm)
    job = subprocess.run(["sbatch", "--parsable", str(array)], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["sbatch", f"--dependency=afterany:{job}", str(analysis)], check=True)
    return job


def run_local(batch: Path, dataset: str | None = None, wandb_mode: str | None = "disabled") -> None:
    """Sem SLURM: roda as tarefas uma por vez, no próprio processo (modo de teste no computador)."""
    lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    for task in lote["tarefas"]:
        print(f"tarefa {task['id'] + 1}/{len(lote['tarefas'])}: {task}", flush=True)
        run_task(batch, task["id"], dataset, wandb_mode)


# QoS que cada partição do Apuana aceita (scontrol show partition, 2026-11)
PARTITION_QOS = {"debug": "complex", "short-simple": "simple", "short-complex": "complex",
                 "long-simple": "simple", "long-complex": "complex", "install": "simple"}


def slurm_options(args, test: bool) -> dict:
    """Opções do SLURM da linha de comando. O teste vai para a `debug` (fila rápida, até 30 min)."""
    partition = args.particao or ("debug" if test else SLURM_DEFAULTS["partition"])
    qos = args.qos or PARTITION_QOS.get(partition, SLURM_DEFAULTS["qos"])
    options = {"partition": partition, "qos": qos, "particao_analise": partition, "qos_analise": qos,
               "time": args.tempo or ("00:30:00" if test else SLURM_DEFAULTS["time"])}
    if args.max_paralelo:
        options["max_paralelo"] = args.max_paralelo
    return options


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="comando", required=True)
    launch = sub.add_parser("lancar")
    launch.add_argument("experimento", type=Path)
    final = sub.add_parser("final")
    final.add_argument("lote", type=Path)
    final.add_argument("configuracoes", nargs="+")
    for p in (launch, final):
        p.add_argument("--teste", action="store_true", help="passada rápida; os números não servem")
        p.add_argument("--local", action="store_true", help="roda aqui, sem SLURM")
        p.add_argument("--zip", default="~/macdeg_data/dataset_1024.zip")
        p.add_argument("--dados", default=None, help="pasta do conjunto (padrão: a da configuração)")
        p.add_argument("--particao", default=None, help="partição do SLURM (padrão: short-simple; teste: debug)")
        p.add_argument("--qos", default=None, help="QoS da partição (simple ou complex; padrão: o da partição)")
        p.add_argument("--tempo", default=None, help="limite por tarefa, ex.: 12:00:00 (padrão; teste: 00:30:00)")
        p.add_argument("--max-paralelo", type=int, default=None,
                       help="tarefas simultâneas (padrão: o limite de GPUs do QoS: simple 2, complex 4)")
    args = ap.parse_args()
    if args.comando == "lancar":
        batch = prepare(args.experimento, args.dados, args.teste)
    else:
        batch = prepare(None, args.dados, args.teste, final=args.configuracoes, from_batch=args.lote)
    lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    print(f"lote {batch}: {len(lote['configuracoes'])} configuração(ões), {len(lote['tarefas'])} tarefas")
    if lote["codigo_alterado_sem_commit"]:
        print("AVISO: há mudanças sem commit; elas NÃO estão na cópia do código usada pelas tarefas.")
    if args.local:
        run_local(batch, args.dados)
        from macdeg.analysis import analyze
        analyze(batch, args.dados)
    else:
        print("job:", submit(batch, args.zip, slurm_options(args, lote["teste"])))


if __name__ == "__main__":
    main()
