"""Curvas de treino do W&B: baixa só os números por época e gera gráficos locais.

Uso: python -m macdeg.curvas [--grupo rodada6-] [--out ~/macdeg_runs/curvas]

Os runs são agrupados por `<lote>/<configuração>` (macdeg.cluster._wandb). Para cada grupo cujo nome começa
com `--grupo`, grava em `--out`:
  - runs.csv: um run por linha (grupo, nome, estado, dobras, épocas gravadas, épocas baixadas);
  - curvas.csv: uma linha por run e época, com as métricas de CURVAS;
  - <configuração>.png: AUROC de validação e perda de treino por época (cada dobra em cinza, a média em
    destaque). Junta os lotes da mesma configuração (ex.: a rodada 6, dividida entre duas contas).
Os runs `<lote>-analise` (resumo da análise: só a tabela de ranking) não têm curvas e ficam fora dos
gráficos; no site, são eles que deixam painéis com "There's no data" quando estão selecionados.
Também imprime um diagnóstico: runs sem nenhuma época (offline nunca sincronizado, ou caiu antes da
primeira época), que são os que aparecem como "There's no data" no site.

Lê a chave pelo .env (macdeg.ambiente.secret); nunca a imprime. O W&B só tem números (CLAUDE.md, seção 6).
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

CURVAS = ["epoca", "perda_treino", "auroc_val", "lr", "segundos_epoca", "segundos_esperando_dados"]
PAINEIS = [("auroc_val", "AUROC de validação"), ("perda_treino", "Perda de treino")]


def fetch(api, path: str, group_prefix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs do projeto `path` (entidade/projeto) cujo grupo começa com `group_prefix`, e as curvas deles.
    Só baixa as chaves de CURVAS (os histogramas de pesos do wandb.watch ficam de fora: pesados)."""
    filters = {"group": {"$regex": f"^{re.escape(group_prefix)}"}} if group_prefix else {}
    runs, curves = [], []
    for run in api.runs(path, filters=filters):
        group = run.group or ""
        if not group.startswith(group_prefix):
            continue
        rows = [{k: row.get(k) for k in CURVAS} for row in run.scan_history(keys=CURVAS)]
        config = run.config or {}
        runs.append({"grupo": group, "run": run.name, "estado": run.state, "outer": config.get("outer"),
                     "inner": config.get("inner"), "epocas_gravadas": getattr(run, "lastHistoryStep", -1) + 1,
                     "epocas_baixadas": len(rows)})
        curves += [{"grupo": group, "run": run.name, "outer": config.get("outer"), "inner": config.get("inner"), **r}
                   for r in rows]
    runs = pd.DataFrame(runs, columns=["grupo", "run", "estado", "outer", "inner", "epocas_gravadas", "epocas_baixadas"])
    curves = pd.DataFrame(curves, columns=["grupo", "run", "outer", "inner", *CURVAS])
    return runs.sort_values(["grupo", "outer", "inner"], ignore_index=True), curves


def plot(curves: pd.DataFrame, out: Path) -> list[Path]:
    """Um PNG por configuração (parte do grupo depois da "/"): um painel por métrica (eixo x = época), dobras
    em cinza fino, média em destaque (só nas épocas que pelo menos metade das dobras alcançou: no fim, com
    poucas dobras, a média só reflete as que treinaram mais)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    written = []
    curves = curves[curves["grupo"].str.contains("/")]
    for group, frame in curves.groupby(curves["grupo"].str.split("/").str[-1]):
        fig, axes = plt.subplots(1, len(PAINEIS), figsize=(5 * len(PAINEIS), 3.6), constrained_layout=True)
        for ax, (key, title) in zip(axes, PAINEIS):
            data = frame.dropna(subset=[key])
            for _, run in data.groupby("run"):
                ax.plot(run["epoca"], run[key], color="#9aa0a6", linewidth=1, alpha=0.7)
            by_epoch = data.groupby("epoca")[key]
            mean = by_epoch.mean()[by_epoch.count() >= data["run"].nunique() / 2]
            ax.plot(mean.index, mean.values, color="#1f4e8c", linewidth=2, label=f"média ({data['run'].nunique()} dobras)")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("época")
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
        lotes = ", ".join(sorted(frame["grupo"].str.split("/").str[0].unique()))
        fig.suptitle(f"{group}  ({lotes})", fontsize=11)
        path = out / (re.sub(r"[^A-Za-z0-9_.-]+", "_", group) + ".png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def main() -> None:
    from macdeg.ambiente import load_dotenv, secret
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grupo", default="", help="prefixo do grupo (ex.: rodada6-); vazio = todos")
    ap.add_argument("--out", default="~/macdeg_runs/curvas", help="pasta de saída (fora do repositório)")
    args = ap.parse_args()
    load_dotenv()
    if not secret("WANDB_API_KEY"):
        raise SystemExit("Sem WANDB_API_KEY no .env.")
    import wandb
    api = wandb.Api(api_key=secret("WANDB_API_KEY"))
    path = f"{secret('WANDB_ENTITY') or api.default_entity}/{os.environ.get('WANDB_PROJECT', 'macdeg_classifier')}"
    runs, curves = fetch(api, path, args.grupo)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    curves.to_csv(out / "curvas.csv", index=False)
    written = plot(curves, out) if len(curves) else []
    print(f"{len(runs)} runs em {runs['grupo'].nunique()} grupo(s); {len(written)} gráfico(s) em {out}")
    summary = runs.groupby("grupo").agg(runs=("run", "size"), sem_epocas=("epocas_baixadas", lambda s: int((s == 0).sum())),
                                        estados=("estado", lambda s: ",".join(f"{k}:{v}" for k, v in s.value_counts().items())))
    print(summary.to_string())


if __name__ == "__main__":
    main()
