"""Roda um experimento inteiro: tarefas, análise e envio dos resultados.

As tarefas rodam uma por vez neste processo ou, se o job tem várias GPUs (CUDA_VISIBLE_DEVICES), uma por
GPU ao mesmo tempo. No cluster, quem chama é o cluster/experimento.sh (um job, 2 GPUs). No computador,
rode direto.

    python rodar.py                                   # a referência (configs/experimentos/001_referencia.yaml)
    python rodar.py configs/experimentos/002_x.yaml   # outro experimento
    python rodar.py --teste                           # passada rápida: minutos, números sem valor, nada é enviado
    python rodar.py --piloto                          # UM treino de verdade (dobra 0): números reais em ~1-2 h
    python rodar.py --retomar ~/macdeg_runs/<lote>    # continua um lote interrompido (pula o que já terminou)
    python rodar.py --retomar ~/macdeg_runs/<lote> --do-fim   # outro job no mesmo lote, a partir da última
                                                      # tarefa; a análise roda no job que terminar por último
    python rodar.py --final ~/macdeg_runs/<lote> <configuração>   # modelo final (5 dobras), guardado no lote

Resultados: ~/macdeg_runs/<lote>/ ; GitHub (branch resultados) e W&B (curvas). No --final, os pesos ficam
em <lote>/modelo/; o Hugging Face é à mão: python -m macdeg.hub publicar <lote> <configuração>.
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
from macdeg.cluster import RUNS, dataset_signature, failed_tasks, pending_tasks, prepare, run_local
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
    ap.add_argument("--piloto", action="store_true", help="um treino de verdade só (dobra externa 0, interna 0)")
    ap.add_argument("--retomar", type=Path, default=None, help="pasta de um lote já criado")
    ap.add_argument("--final", nargs="+", metavar=("LOTE", "CONFIG"), help="modelo final de configurações de um lote")
    ap.add_argument("--do-fim", action="store_true",
                    help="com --retomar: começa pela última tarefa (para um segundo job dividir o mesmo lote)")
    ap.add_argument("--dados", default="~/macdeg_data/dataset_1024", help="pasta do conjunto (com as imagens)")
    ap.add_argument("--sem-wandb", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    check(args.dados, args.teste)
    if args.retomar:
        batch = args.retomar.expanduser()
    elif args.final:
        batch = prepare(None, args.dados, args.teste, final=args.final[1:], from_batch=Path(args.final[0]).expanduser())
    else:
        batch = prepare(args.experimento, args.dados, args.teste, pilot=args.piloto)
    lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "ultimo_lote.txt").write_text(str(batch))     # o experimento.sh copia os logs do job para cá
    (batch / "logs").mkdir(exist_ok=True)
    print(f"lote {batch}: {len(lote['configuracoes'])} configuração(ões), {len(lote['tarefas'])} tarefas", flush=True)

    wandb_mode = "disabled" if (args.teste or args.sem_wandb) else None
    run_local(batch, args.dados, wandb_mode, from_end=args.do_fim)

    missing = pending_tasks(batch, lote)
    failed = failed_tasks(batch, lote)
    if failed:    # falharam em outro job: ninguém vai analisar até alguém refazê-las
        sys.exit(f"tarefa(s) {[i + 1 for i in failed]} falharam (falhou.txt); refaça com --retomar {batch}")
    if missing:   # outro job ainda está nelas: ele faz a análise quando terminar
        print(f"faltam {len(missing)} tarefa(s) em andamento em outro job; a análise fica com ele", flush=True)
        return
    lock = batch / "analise_em_andamento"
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:   # os dois jobs terminaram juntos: só um analisa
        print(f"outro job já está analisando este lote (se ele morreu, apague {lock})", flush=True)
        return
    try:
        from macdeg.analysis import analyze
        analyze(batch, args.dados, push=not lote["teste"])
    finally:
        lock.unlink(missing_ok=True)
    print(f"fim: {batch}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
