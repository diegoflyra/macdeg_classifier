"""Roda um experimento inteiro: tarefas, análise e envio dos resultados.

As tarefas rodam uma por vez por GPU. No cluster, dois jobs podem dividir o mesmo lote, começando por
pontas opostas. No computador, rode direto.

    python rodar.py                                   # a referência (configs/experimentos/001_referencia.yaml)
    python rodar.py configs/experimentos/002_x.yaml   # outro experimento
    python rodar.py --teste                           # passada rápida: minutos, números sem valor, nada é enviado
    python rodar.py --piloto                          # uma tarefa por configuração (externa 0, interna 0)
    python rodar.py --retomar ~/macdeg_runs/<lote>    # continua um lote interrompido (pula o que já terminou)
    python rodar.py --retomar ~/macdeg_runs/<lote> --do-fim   # começa pela última tarefa pendente
    python rodar.py --final ~/macdeg_runs/<lote> <configuração>   # modelo final (5 dobras) -> Hugging Face

Resultados: ~/macdeg_runs/<lote>/ ; GitHub (branch resultados), W&B (curvas) e, no --final, Hugging Face.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from macdeg.ambiente import commit_info, load_dotenv, push_results, which_secrets
from macdeg.cluster import RUNS, dataset_signature, job_lock, pending_tasks, prepare, run_local
from macdeg.data.sources import resolve

REFERENCIA = "configs/experimentos/001_referencia.yaml"


def check(dataset: str, test: bool) -> None:
    """Confere tudo o que pode falhar antes de gastar GPU."""
    print("segredos:", which_secrets(), flush=True)          # só True/False, nunca o valor
    info = commit_info()
    print(f"commit: {info['commit'][:10]} | mudanças sem commit: {info['codigo_alterado_sem_commit']}")
    root = resolve(dataset)
    if not (root / "manifest.csv").exists():
        sys.exit(f"ERRO: não achei {root}/manifest.csv.")
    print("assinatura dos dados:", dataset_signature(root, "splits_nested_seed0.csv"), flush=True)
    if not test:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "teste_envio.json").write_text(json.dumps({"quando": time.strftime("%Y-%m-%d %H:%M")}))
            if not push_results(tmp, "resultados/_teste_envio", "teste de envio"):
                sys.exit("ERRO: o envio ao GitHub falhou; confira o GITHUB_TOKEN no .env.")
        print("GitHub: envio testado e funcionando", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experimento", nargs="?", default=REFERENCIA)
    ap.add_argument("--teste", action="store_true", help="passada rápida; os números não servem")
    ap.add_argument("--piloto", action="store_true", help="uma tarefa por configuração (dobra externa 0, interna 0)")
    ap.add_argument("--retomar", type=Path, default=None, help="pasta de um lote já criado")
    ap.add_argument("--do-fim", action="store_true", help="com --retomar, começa pela última tarefa pendente")
    ap.add_argument("--final", nargs="+", metavar=("LOTE", "CONFIG"), help="modelo final de configurações de um lote")
    ap.add_argument("--dados", default="~/macdeg_data/dataset_1024", help="pasta do conjunto (com as imagens)")
    ap.add_argument("--sem-wandb", action="store_true")
    args = ap.parse_args()
    if args.do_fim and not args.retomar:
        ap.error("--do-fim requer --retomar com o caminho do lote")

    load_dotenv()
    if args.retomar:
        batch = args.retomar.expanduser()
        lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
        check(args.dados, args.teste or lote["teste"])
    else:
        check(args.dados, args.teste)
        if args.final:
            batch = prepare(None, args.dados, args.teste, final=args.final[1:], from_batch=Path(args.final[0]).expanduser())
        else:
            batch = prepare(args.experimento, args.dados, args.teste, pilot=args.piloto)
        lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    RUNS.mkdir(parents=True, exist_ok=True)
    (batch / "logs").mkdir(exist_ok=True)
    (RUNS / "ultimo_lote.txt").write_text(str(batch), encoding="utf-8")
    if job_lote := os.environ.get("MACDEG_JOB_LOTE"):
        pointer = Path(job_lote)
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text(str(batch.resolve()), encoding="utf-8")
        temporary.replace(pointer)
    print(f"lote {batch}: {len(lote['configuracoes'])} configuração(ões), {len(lote['tarefas'])} tarefas", flush=True)

    wandb_mode = "disabled" if (args.teste or lote["teste"] or args.sem_wandb) else None
    run_local(batch, args.dados, wandb_mode, from_end=args.do_fim)

    missing = pending_tasks(batch, lote)
    if missing:
        print(f"faltam {len(missing)} tarefa(s) no lote {batch}; a análise começa após a última", flush=True)
        return
    with job_lock(batch / "analise.lock") as acquired:
        done = batch / "analise_concluida"
        if not acquired or done.exists():
            return
        from macdeg.analysis import analyze
        analyze(batch, args.dados, push=not lote["teste"])
        done.touch()
    print(f"fim: {batch}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
