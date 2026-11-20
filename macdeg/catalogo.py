"""Catálogo de tudo o que foi treinado, gerado a partir da branch `resultados`.

Uso: python -m macdeg.catalogo [--ref origin/resultados] [--out docs/experimentos.md]
     (antes: git fetch origin resultados)

Para cada lote: quando, com que commit (o código exato: `git show <commit>:macdeg/train.py`), o experimento
(base, mudanças, referência), a receita comum e, por configuração, a métrica principal, o veredito, as
métricas secundárias e o custo. Entre lotes com a mesma assinatura dos dados, compara a configuração 00
de cada lote com a 00 do lote anterior (veredito pareado, nas mesmas imagens).
Só lê agregados e previsões com sample_id (o que já está no GitHub).
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from macdeg.metrics import paired_verdict

RECIPE_KEYS = ["model", "pretrained", "image_size", "batch_size", "epochs", "patience", "lr", "weight_decay",
               "warmup_epochs", "balance", "augment", "aux_dr", "aux_drusas", "preprocess", "crop", "oversample"]


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}".replace(".", ",")
    return str(value)


def _ic(pair) -> str:
    return f"[{_fmt(pair[0])}; {_fmt(pair[1])}]" if pair else "—"


def _changes(changes: dict) -> str:
    if not changes:
        return "(base)"
    return "; ".join(f"`{k}: {json.dumps(v, ensure_ascii=False)}`" for k, v in changes.items())


def load_batches(folder: Path) -> list[dict]:
    """Lotes de verdade (sem teste), em ordem de criação. folder = .../resultados."""
    batches = []
    for path in sorted(folder.glob("*/lote.json")):
        lote = json.loads(path.read_text(encoding="utf-8"))
        report_path = path.parent / "relatorio.json"
        if lote.get("teste") or not report_path.exists():
            continue
        history = path.parent / "historico.csv"
        batches.append({"dir": path.parent, "lote": lote, "relatorio": json.loads(report_path.read_text(encoding="utf-8")),
                        "historico": pd.read_csv(history) if history.exists() else None})
    return sorted(batches, key=lambda b: b["lote"].get("criado", ""))


def _predictions(batch: dict, config: str) -> pd.DataFrame | None:
    path = batch["dir"] / f"previsoes_externas_{config}.csv"
    return pd.read_csv(path) if path.exists() else None


def _config_rows(batch: dict) -> list[str]:
    report, history = batch["relatorio"], batch["historico"]
    rows = []
    for name, r in report["configuracoes"].items():
        verdict = r.get("veredito")
        verdict_text = (f"{_fmt(verdict['delta_f2_media_bases'])} {_ic(verdict['ic95'])} **{verdict['veredito']}**"
                        if verdict else "referência")
        stage = r.get("auroc_por_estagio") or {}
        early = (stage.get("dmri_precoce") or {}).get("auroc")
        dr = (r.get("especificidade_por_doenca") or {}).get("retinopatia_diabetica", {}).get("especificidade")
        epochs = r.get("epocas_melhores") or []
        cost = "—"
        if history is not None and "configuracao" in history:
            part = history[history["configuracao"] == name]
            if len(part):
                cost = f"{_fmt(part['seconds'].sum() / 3600, 1)} h ({_fmt(part['seconds'].mean(), 0)} s/época)"
        rows.append(f"| `{name}` | {_changes(r.get('mudancas') or {})} | {_fmt(r['f2_media_bases'])} "
                    f"{_ic(r.get('f2_media_bases_ic95'))} | {verdict_text} | {_fmt(r.get('auroc'))} | "
                    f"{_fmt(r.get('auprc'))} | {_fmt(r.get('sensibilidade'))} | {_fmt(dr)} | {_fmt(early)} | "
                    f"{_fmt(float(np.median(epochs)), 1) if epochs else '—'} | {cost} |")
    return rows


def render(batches: list[dict]) -> str:
    lines = ["# Catálogo dos experimentos", "",
             "Gerado por `python -m macdeg.catalogo` a partir da branch `resultados` (não edite à mão).",
             "Código exato de cada lote: `git show <commit>:macdeg/train.py` (ou `git checkout <commit>`).",
             "Métrica principal: média do F2 dentro de cada base, limiar escolhido em outras dobras.",
             "Veredito: diferença pareada contra a referência do lote, IC95 por paciente.", "",
             "| lote | criado | commit | experimento | configurações | assinatura dos dados |",
             "|---|---|---|---|---|---|"]
    for b in batches:
        lote = b["lote"]
        lines.append(f"| [{lote['nome']}](#{lote['nome']}) | {lote.get('criado', '—')} | `{lote['commit'][:8]}` | "
                     f"{lote['experimento'].get('nome', '—')} | {len(b['relatorio']['configuracoes'])} | "
                     f"`{lote['assinatura_dados']}` |")
    previous = None
    for b in batches:
        lote, report = b["lote"], b["relatorio"]
        experiment = lote["experimento"]
        first = next(iter(report["configuracoes"]))
        recipe = next(c["config"] for c in lote["configuracoes"] if c["nome"] == first)
        lines += ["", f"## {lote['nome']}", "",
                  f"- Criado em {lote.get('criado', '—')}; commit `{lote['commit']}`"
                  + (" (**com código alterado sem commit**)" if lote.get("codigo_alterado_sem_commit") else "") + ".",
                  f"- Tarefas: {lote.get('n_tarefas', '—')}; faltando na análise: {len(report.get('tarefas_faltando') or [])}.",
                  f"- Mudanças da base (`base_mudancas`): {_changes(experiment.get('base_mudancas') or {})}.",
                  f"- Referência do veredito: `{report.get('referencia', first)}`.",
                  "- Receita comum: " + ", ".join(f"`{k}: {json.dumps(recipe[k], ensure_ascii=False)}`"
                                                  for k in RECIPE_KEYS if recipe.get(k) is not None) + ".", "",
                  "| configuração | mudança | F2 médio por base [IC95] | Δ contra a referência [IC95] | AUROC | AUPRC | "
                  "sensibilidade | esp. em DR | DMRI precoce × DR (AUROC) | melhor época (mediana) | GPU |",
                  "|---|---|---|---|---|---|---|---|---|---|---|", *_config_rows(b)]
        nested = report.get("aninhado")
        if nested:
            chosen = pd.Series(nested.get("escolhida_por_dobra", {})).value_counts().to_dict()
            lines += ["", f"Seleção aninhada (a melhor configuração escolhida só com a validação interna): F2 "
                          f"{_fmt(nested['f2_media_bases'])} {_ic(nested.get('f2_media_bases_ic95'))}; escolhas: "
                          + ", ".join(f"`{k}` × {v}" for k, v in chosen.items()) + "."]
        trained_base = experiment.get("treinar_base", True) is not False
        if not trained_base:
            lines += ["", "Este lote não treinou a base (`treinar_base: false`): as variações foram comparadas "
                          f"com `{experiment.get('referencia')}`."]
        if (trained_base and previous is not None
                and previous["lote"]["assinatura_dados"] == lote["assinatura_dados"]):
            prev_first = next(iter(previous["relatorio"]["configuracoes"]))
            a, c = _predictions(previous, prev_first), _predictions(b, first)
            if a is not None and c is not None:
                v = paired_verdict(a, c)
                lines += ["", f"Base deste lote (`{first}`) contra a base do lote anterior (`{prev_first}`): "
                              f"{_fmt(v['delta_f2_media_bases'])} {_ic(v['ic95'])}, **{v['veredito']}**."]
        if trained_base:
            previous = b
    return "\n".join(lines) + "\n"


def extract(ref: str, destination: Path) -> Path:
    """Extrai a pasta `resultados/` da branch (sem checkout), sem os logs."""
    archive = subprocess.run(["git", "archive", "--format=tar", ref, "resultados"], capture_output=True, check=True)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        members = [m for m in tar.getmembers() if "/logs/" not in m.name]
        tar.extractall(destination, members=members, filter="data")
    return destination / "resultados"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="origin/resultados")
    ap.add_argument("--out", type=Path, default=Path("docs/experimentos.md"))
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        batches = load_batches(extract(args.ref, Path(tmp)))
        args.out.write_text(render(batches), encoding="utf-8")
    print(f"{len(batches)} lotes -> {args.out}")


if __name__ == "__main__":
    main()
