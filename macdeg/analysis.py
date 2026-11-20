"""Análise de um lote: junta as dobras e calcula tudo o que vai para o GitHub.

Para cada configuração c e dobra externa k:
  - previsões das 4 dobras internas (validação) -> limiar de k (maximiza a média do F2 por base) e nota
    de seleção (F2 com o limiar de cada interna vindo das outras 3);
  - teste externo k = média das notas dos 4 modelos internos, cortada no limiar de k.
O teste externo de k nunca influenciou nada da dobra k: juntando as 5 dobras, cada imagem do conjunto
tem uma nota "de teste". Isso dá:
  - resultado por configuração (honesto para configurações definidas antes de olhar os resultados);
  - resultado ANINHADO: em cada k, vence a configuração com a melhor nota de seleção (só validação
    interna); é a estimativa honesta do procedimento "rodar a grade e escolher a melhor";
  - veredito pareado de cada configuração contra a referência, nas mesmas imagens.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from macdeg.ambiente import push_results, fetch_results, secret
from macdeg.cluster import task_dir
from macdeg.data.sources import resolve
from macdeg.metrics import (auroc, best_threshold, crossfit, f2_mean, family, paired_verdict, report)

KINDS = ["normal", "outra", "drusas", "dmri"]


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def collect(batch: Path, lote: dict) -> tuple[dict, list]:
    """{configuração: {(outer, inner): {val, test, extra, history}}} e a lista de tarefas que faltam."""
    runs, missing = {}, []
    for task in lote["tarefas"]:
        folder = task_dir(batch, task)
        if not (folder / "metrics.json").exists():
            missing.append(task["id"])
            continue
        runs.setdefault(task["configuracao"], {})[(task["outer"], task["inner"])] = {
            name: _read(folder / f"{name}_predictions.csv") for name in ("val", "test", "extra")} | {
            "history": _read(folder / "history.csv"), "metrics": json.loads((folder / "metrics.json").read_text())}
    return runs, missing


def outer_predictions(runs: dict) -> tuple[pd.DataFrame, dict]:
    """Notas de teste externo de uma configuração e a nota de seleção de cada dobra externa."""
    frames, selection = [], {}
    for k in sorted({o for o, _ in runs}):
        inner = {j: r for (o, j), r in runs.items() if o == k}
        oof = pd.concat([r["val"].assign(inner=j) for j, r in inner.items()], ignore_index=True)
        # Com várias internas, a nota de seleção usa o limiar das outras internas; com uma só (piloto), o
        # limiar da própria validação (otimista, mas só para seleção: o teste externo continua honesto).
        selection[k] = f2_mean(crossfit(oof, "inner")) if oof["inner"].nunique() > 1 else \
            f2_mean(oof, (oof["prob"] >= best_threshold(oof)).astype(int).to_numpy())
        threshold = best_threshold(oof)
        tests = pd.concat([r["test"] for r in inner.values()], ignore_index=True)
        test = tests.groupby("sample_id", as_index=False).agg(
            prob=("prob", "mean"), **{c: (c, "first") for c in ("source", "group_id", "label", "kind", "disease_group")})
        frames.append(test.assign(outer=k, threshold=threshold, pred=(test["prob"] >= threshold).astype(int)))
    return pd.concat(frames, ignore_index=True), selection


def rulers(predictions: pd.DataFrame, manifest: pd.DataFrame | None) -> dict:
    """Réguas que o modelo precisa superar: só a base de origem e só a idade."""
    if manifest is None:
        return {}
    table = predictions.merge(manifest[["sample_id", "age", "outer"]], on=["sample_id", "outer"], how="left")
    scores = pd.Series(np.nan, index=table.index)
    for k in table["outer"].unique():  # fração de DMRI de cada base nas outras 4 dobras externas
        rate = manifest[manifest["outer"] != k].groupby("source")["label"].mean()
        here = table["outer"] == k
        scores[here] = table.loc[here, "source"].map(rate)
    aged = table[table["age"].notna()]
    return {"regua_origem_auroc": auroc(table["label"], scores),
            "regua_idade_auroc": auroc(aged["label"], aged["age"]), "regua_idade_cobertura": round(len(aged) / len(table), 3),
            "regua_idade_por_base": {s: auroc(g["label"], g["age"]) for s, g in aged.groupby(family(aged["source"]))}}


def score_distribution(predictions: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    """Quantis da nota por tipo de olho (Q38: as drusas sem diagnóstico ficam entre normais e DMRI?)."""
    frames = [predictions[["kind", "prob"]]]
    if extra is not None and len(extra):
        frames.append(extra.groupby("sample_id", as_index=False)["prob"].mean().assign(kind="drusas"))
    table = pd.concat(frames, ignore_index=True)
    threshold = float(predictions["threshold"].median())
    rows = []
    for kind_name in KINDS:
        probs = table.loc[table["kind"] == kind_name, "prob"]
        if len(probs):
            rows.append({"tipo": kind_name, "n": len(probs), **{f"p{q}": round(float(probs.quantile(q / 100)), 4)
                         for q in (10, 25, 50, 75, 90)}, "acima_do_limiar": round(float((probs >= threshold).mean()), 4)})
    return pd.DataFrame(rows)


AGE_BANDS = [0, 55, 65, 75, 200]
AGE_LABELS = ["ate_55", "56_65", "66_75", "acima_75"]


def drusas_by_age(extra: pd.DataFrame | None, ages: pd.DataFrame | None, threshold: float) -> list[dict]:
    """Notas das drusas sem diagnóstico por faixa etária (só agregados). A idade entra só aqui, na leitura,
    nunca no treino. Com aux_drusas, também a nota da saída de drusas (prob_drusas)."""
    if extra is None or not len(extra) or ages is None:
        return []
    columns = [c for c in ("prob", "prob_drusas") if c in extra]
    table = extra.groupby("sample_id", as_index=False)[columns].mean().merge(ages, on="sample_id", how="left")
    band = pd.cut(table["age"], AGE_BANDS, labels=AGE_LABELS).astype(object).where(table["age"].notna(), "sem_idade")
    rows = []
    for name in [*AGE_LABELS, "sem_idade"]:
        part = table[band == name]
        if not len(part):
            continue
        row = {"faixa": name, "n": len(part), "mediana_nota_dmri": round(float(part["prob"].median()), 4),
               "acima_do_limiar_dmri": round(float((part["prob"] >= threshold).mean()), 4)}
        if "prob_drusas" in part:
            row["mediana_nota_drusas"] = round(float(part["prob_drusas"].median()), 4)
        rows.append(row)
    return rows


def _load_drusas_ages(dataset: str | None, lote: dict) -> pd.DataFrame | None:
    from macdeg.data.drusas import FOLDS_FILE
    try:
        root = resolve(dataset or lote["configuracoes"][0]["config"]["dataset"])
        return pd.read_csv(root / FOLDS_FILE, usecols=["sample_id", "age"])
    except (FileNotFoundError, ValueError):
        return None


def stage_auroc(predictions: pd.DataFrame, manifest: pd.DataFrame | None) -> dict:
    """AUROC de cada estágio de DMRI contra os "sem DMRI" das mesmas bases (só os estágios informados;
    hoje, só o HYAMD estadia, e os "sem DMRI" dele são todos olhos com DR: é a separação precoce x DR)."""
    if manifest is None or "label_detail" not in manifest:
        return {}
    detail = manifest.drop_duplicates("sample_id").set_index("sample_id")["label_detail"]
    table = predictions.assign(estagio=predictions["sample_id"].map(detail))
    out = {}
    for stage, positives in table[(table["label"] == 1) & (table["estagio"] != "dmri")].groupby("estagio"):
        sources = sorted(positives["source"].unique())
        negatives = table[(table["label"] == 0) & table["source"].isin(sources)]
        both = pd.concat([positives, negatives])
        out[stage] = {"n": len(positives), "n_sem_dmri": len(negatives), "bases": sources,
                      "auroc": auroc(both["label"], both["prob"])}
    return out


def discordances(reference: pd.DataFrame, candidate: pd.DataFrame, manifest: pd.DataFrame | None) -> list[dict]:
    """Onde a configuração e a referência discordam, imagem a imagem (mesmas imagens, cada uma com o seu
    limiar escolhido fora da dobra). Responde "que diagnósticos se ganham ou se perdem com a mudança?":
    - DMRI: detectada só pela referência (perdida com a mudança), só pela nova (ganha) ou por nenhuma;
    - sem DMRI: falso positivo só da referência (corrigido) ou só da nova (criado).
    Por base, por estágio da DMRI (quando a base informa, ex.: HYAMD precoce x intermediária/tardia) e,
    nos "sem DMRI", por doença. Só contagens."""
    joined = reference[["sample_id", "source", "label", "kind", "disease_group", "pred"]].merge(
        candidate[["sample_id", "pred"]], on="sample_id", suffixes=("_ref", "_nova"), validate="one_to_one")
    if manifest is not None and "label_detail" in manifest:
        joined["estagio"] = joined["sample_id"].map(manifest.drop_duplicates("sample_id").set_index("sample_id")["label_detail"])
    rows = []

    def add(group: str, value: str, part: pd.DataFrame) -> None:
        dmri, negative = part[part["label"] == 1], part[part["label"] == 0]
        ref, new = dmri["pred_ref"] == 1, dmri["pred_nova"] == 1
        fp_ref, fp_new = negative["pred_ref"] == 1, negative["pred_nova"] == 1
        rows.append({"grupo": group, "valor": value, "n_dmri": len(dmri), "dmri_so_referencia": int((ref & ~new).sum()),
                     "dmri_so_nova": int((~ref & new).sum()), "dmri_nenhuma": int((~ref & ~new).sum()),
                     "n_sem_dmri": len(negative), "falso_positivo_so_referencia": int((fp_ref & ~fp_new).sum()),
                     "falso_positivo_so_nova": int((~fp_ref & fp_new).sum())})

    add("geral", "todas", joined)
    for name, part in joined.groupby(family(joined["source"])):
        add("base", name, part)
    if "estagio" in joined:
        for name, part in joined[joined["label"] == 1].groupby(joined["estagio"].fillna("?")):
            add("estagio_dmri", name, part)
    negatives = joined[joined["label"] == 0]
    for name, part in negatives.groupby(negatives["disease_group"].fillna("normal")):
        add("doenca", name, part)
    return rows


def _load_manifest(dataset: str | None, lote: dict) -> pd.DataFrame | None:
    try:
        root = resolve(dataset or lote["configuracoes"][0]["config"]["dataset"])
        manifest = pd.read_csv(root / "manifest.csv", dtype={"patient_id": str})
        return manifest.merge(pd.read_csv(root / lote["splits"])[["uid", "outer"]], on="uid")
    except FileNotFoundError:
        return None


def _reference(lote: dict, outer: dict) -> tuple[str, pd.DataFrame | None]:
    target = lote["experimento"].get("referencia")
    if not target:
        name = lote["configuracoes"][0]["nome"]
        return name, outer.get(name)
    folder = Path(lote.get("_tmp", "/tmp")) / "referencia"
    run, config = target.rsplit("/", 1)
    fetch_results(run, folder)
    reference = json.loads((folder / "relatorio.json").read_text())
    if reference["assinatura_dados"] != lote["assinatura_dados"]:
        raise SystemExit("A referência foi medida em outro conjunto ou outras divisões: o veredito não vale.")
    return target, pd.read_csv(folder / f"previsoes_externas_{config}.csv")


def analyze(batch: Path | str, dataset: str | None = None, push: bool = True) -> dict:
    batch = Path(batch)
    lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    out = batch / "analise"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    runs, missing = collect(batch, lote)
    manifest = _load_manifest(dataset, lote)
    drusas_ages = _load_drusas_ages(dataset, lote)
    result = {"lote": lote["nome"], "commit": lote["commit"], "assinatura_dados": lote["assinatura_dados"],
              "teste": lote["teste"], "piloto": lote.get("piloto", False), "tarefas_faltando": missing, "analisado": time.strftime("%Y-%m-%d %H:%M"),
              "configuracoes": {}}
    if lote["final"]:
        result["final"] = {name: final_threshold(r) for name, r in runs.items()}
        from macdeg.hub import export_final    # pesos ficam no cluster; o HF é manual (python -m macdeg.hub)
        for name in runs:
            folder = export_final(batch, name, result["final"][name])
            result["final"][name]["modelo"] = str(folder)
            shutil.copy(folder / "config.json", out / f"modelo_{name}.json")   # cartão, sem pesos, vai ao GitHub
        _write(out, lote, result, None, None)
        return _publish(batch, out, lote, result, push)

    outer, selection, histories = {}, {}, []
    for name, config_runs in runs.items():
        complete = {k for k in {o for o, _ in config_runs}
                    if sum(1 for o, _ in config_runs if o == k) == len({t["inner"] for t in lote["tarefas"]})}
        usable = {key: r for key, r in config_runs.items() if key[0] in complete}
        if not usable:
            continue
        outer[name], selection[name] = outer_predictions(usable)
        extra = pd.concat([r["extra"] for r in usable.values() if r["extra"] is not None] or [pd.DataFrame()])
        changes = next(c["mudancas"] for c in lote["configuracoes"] if c["nome"] == name)
        result["configuracoes"][name] = {"mudancas": changes, "dobras_externas": sorted(complete),
                                         **report(outer[name]), **rulers(outer[name], manifest),
                                         "auroc_por_estagio": stage_auroc(outer[name], manifest),
                                         "nota_selecao_por_dobra": selection[name],
                                         "epocas_melhores": [r["metrics"]["best_epoch"] for r in usable.values()],
                                         "gpus": sorted({r["metrics"]["gpu"] for r in usable.values()})}
        dist = score_distribution(outer[name], extra if len(extra) else None)
        dist.insert(0, "configuracao", name)
        result["configuracoes"][name]["distribuicao_notas"] = dist.drop(columns="configuracao").to_dict("records")
        by_age = drusas_by_age(extra if len(extra) else None, drusas_ages, float(outer[name]["threshold"].median()))
        if by_age:
            result["configuracoes"][name]["drusas_por_idade"] = by_age
        for (k, j), r in usable.items():
            if r["history"] is not None:
                histories.append(r["history"].assign(configuracao=name, outer=k, inner=j))

    if not outer:
        raise SystemExit("Nenhuma dobra externa completa ainda.")
    reference_name, reference = _reference({**lote, "_tmp": str(out.parent)}, outer)
    result["referencia"] = reference_name
    for name, preds in outer.items():
        if reference is not None and name != reference_name:
            result["configuracoes"][name]["veredito"] = paired_verdict(reference, preds)
            result["configuracoes"][name]["discordancias"] = discordances(reference, preds, manifest)

    if len(outer) > 1:  # resultado aninhado: em cada dobra externa, a melhor configuração pela validação interna
        folds = sorted(set.intersection(*[set(s) for s in selection.values()]))
        chosen = {k: max(selection, key=lambda c: selection[c][k]) for k in folds}
        nested = pd.concat([outer[c][outer[c]["outer"] == k] for k, c in chosen.items()], ignore_index=True)
        result["aninhado"] = {"escolhida_por_dobra": chosen, **report(nested)}

    _write(out, lote, result, outer, histories)
    return _publish(batch, out, lote, result, push)


def final_threshold(runs: dict) -> dict:
    """Treino final (5 dobras externas): limiar do ensemble, escolhido nas notas fora da dobra."""
    oof = pd.concat([r["val"].assign(outer=o) for (o, _), r in runs.items()], ignore_index=True)
    return {"limiar": best_threshold(oof), "f2_media_bases_fora_da_dobra": f2_mean(crossfit(oof, "outer")),
            "dobras": sorted(o for o, _ in runs), "epocas_melhores": [r["metrics"]["best_epoch"] for r in runs.values()]}


def ranking(result: dict) -> pd.DataFrame:
    rows = []
    for name, r in result["configuracoes"].items():
        rows.append({"configuracao": name, "mudancas": json.dumps(r["mudancas"], ensure_ascii=False),
                     "f2_media_bases": r["f2_media_bases"], "f2_ic95": r["f2_media_bases_ic95"],
                     "auroc": r["auroc"], "sensibilidade": r["sensibilidade"],
                     "esp_normais": r["especificidade_normais"], "esp_outras": r["especificidade_outras_doencas"],
                     "regua_origem": r.get("regua_origem_auroc"),
                     "veredito": (r.get("veredito") or {}).get("veredito", "referência")})
    return pd.DataFrame(rows).sort_values("f2_media_bases", ascending=False)


def _write(out: Path, lote: dict, result: dict, outer: dict | None, histories: list | None) -> None:
    public = {k: v for k, v in lote.items() if k not in ("tarefas",)}
    public["n_tarefas"] = len(lote["tarefas"])
    (out / "lote.json").write_text(json.dumps(public, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "relatorio.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    if outer:
        for name, preds in outer.items():
            preds.to_csv(out / f"previsoes_externas_{name}.csv", index=False)
        ranking(result).to_csv(out / "ranking.csv", index=False)
        rows = [{"configuracao": n, **d} for n, r in result["configuracoes"].items() for d in r.get("discordancias", [])]
        if rows:
            pd.DataFrame(rows).to_csv(out / "discordancias.csv", index=False)
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(out / "historico.csv", index=False)
    (out / "RESUMO.md").write_text(summary_markdown(result), encoding="utf-8")


def _fmt(value, digits=3):
    return "—" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def summary_markdown(result: dict) -> str:
    lines = [f"# {result['lote']}", "", f"- commit: `{result['commit']}`",
             f"- assinatura dos dados: `{result['assinatura_dados']}`", f"- analisado: {result['analisado']}"]
    if result["teste"]:
        lines.append("- **MODO DE TESTE: os números não servem como resultado.**")
    if result.get("piloto"):
        lines.append("- **PILOTO: um treino só (dobra externa 0). Números reais, mas de 1/5 do conjunto; IC largo.**")
    if result["tarefas_faltando"]:
        lines.append(f"- tarefas faltando: {len(result['tarefas_faltando'])}")
    if result.get("final"):
        lines += ["", "## Treino final", ""]
        for name, f in result["final"].items():
            lines.append(f"- {name}: limiar {f['limiar']:.4f}, F2 médio por base fora da dobra "
                         f"{_fmt(f['f2_media_bases_fora_da_dobra'])}; pesos em `{f.get('modelo', '—')}` "
                         f"(cartão: `modelo_{name}.json`; publicar: `python -m macdeg.hub publicar <lote> {name}`)")
        return "\n".join(lines) + "\n"
    lines += ["", f"Referência: `{result.get('referencia')}`", "",
              "| configuração | mudanças | F2 médio por base [IC95] | AUROC | sens. | esp. normais | esp. outras | régua origem | veredito |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in ranking(result).iterrows():
        ci = r["f2_ic95"] or [None, None]
        lines.append(f"| {r['configuracao']} | `{r['mudancas']}` | {_fmt(r['f2_media_bases'])} [{_fmt(ci[0])}, {_fmt(ci[1])}] "
                     f"| {_fmt(r['auroc'])} | {_fmt(r['sensibilidade'])} | {_fmt(r['esp_normais'])} | {_fmt(r['esp_outras'])} "
                     f"| {_fmt(r['regua_origem'])} | {r['veredito']} |")
    if "aninhado" in result:
        n = result["aninhado"]
        lines += ["", f"**Aninhado** (escolha da melhor configuração feita só com validação interna): F2 médio por base "
                  f"{_fmt(n['f2_media_bases'])} {n['f2_media_bases_ic95']}; escolhida por dobra: {n['escolhida_por_dobra']}"]
    for name, r in result["configuracoes"].items():
        lines += ["", f"## {name}", "", "| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |", "|---|---|---|---|---|---|---|"]
        for base, b in r["por_base"].items():
            lines.append(f"| {base} | {b['n_dmri']} | {b['n_sem_dmri']} | {_fmt(b['f2'])} | {_fmt(b['auroc'])} "
                         f"| {_fmt(b['sensibilidade'])} | {_fmt(b['especificidade'])} |")
        lines += ["", "Distribuição das notas (quantis) e fração acima do limiar:", "",
                  "| tipo | n | p10 | p50 | p90 | acima do limiar |", "|---|---|---|---|---|---|"]
        for d in r["distribuicao_notas"]:
            lines.append(f"| {d['tipo']} | {d['n']} | {d['p10']} | {d['p50']} | {d['p90']} | {d['acima_do_limiar']} |")
        if r.get("discordancias"):
            lines += ["", "Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / "
                      "só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):",
                      "", "| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for d in r["discordancias"]:
                lines.append(f"| {d['grupo']} | {d['valor']} | {d['n_dmri']} | {d['dmri_so_referencia']} | {d['dmri_so_nova']} "
                             f"| {d['dmri_nenhuma']} | {d['n_sem_dmri']} | {d['falso_positivo_so_referencia']} "
                             f"| {d['falso_positivo_so_nova']} |")
        if r.get("auroc_por_estagio"):
            lines += ["", "AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: " + ", ".join(
                f"{s} {v['auroc']:.3f} (n={v['n']} x {v['n_sem_dmri']}, {'/'.join(v['bases'])})"
                for s, v in r["auroc_por_estagio"].items() if v["auroc"] is not None)]
        if r.get("especificidade_por_doenca"):
            lines += ["", "Especificidade por doença: " + ", ".join(
                f"{g} {v['especificidade']:.2f} (n={v['n']})" for g, v in r["especificidade_por_doenca"].items())]
    return "\n".join(lines) + "\n"


def _publish(batch: Path, out: Path, lote: dict, result: dict, push: bool) -> dict:
    if push and secret("GITHUB_TOKEN") and not lote["teste"]:
        ok = push_results(out, f"resultados/{lote['nome']}", f"resultados: {lote['nome']}")
        print("GitHub:", "enviado" if ok else "FALHOU (os arquivos estão em " + str(out) + ")")
    if secret("WANDB_API_KEY") and not lote["teste"]:   # lote de teste não vai a lugar nenhum
        _wandb_summary(lote, result, out)
    print((out / "RESUMO.md").read_text(encoding="utf-8"))
    return result


def _wandb_summary(lote: dict, result: dict, out: Path) -> None:
    import os
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "macdeg_classifier"), entity=secret("WANDB_ENTITY") or None,
                     group=lote["nome"], name=f"{lote['nome']}-analise", job_type="analise",
                     config={"commit": lote["commit"], "assinatura_dados": lote["assinatura_dados"],
                             "experimento": lote["experimento"]}, tags=["teste"] if lote["teste"] else [])
    if (out / "ranking.csv").exists():
        run.log({"ranking": wandb.Table(dataframe=pd.read_csv(out / "ranking.csv"))})
    for name, r in result.get("configuracoes", {}).items():
        run.summary[name] = {k: r[k] for k in ("f2_media_bases", "auroc", "sensibilidade",
                                               "especificidade_normais", "especificidade_outras_doencas")}
    run.finish()


def push_logs(batch: Path | str) -> bool:
    """Envia os logs (texto) das tarefas e da análise para resultados/<lote>/logs. Roda no fim do job de
    análise, depois da análise. Os logs só têm agregados, caminhos do cluster e métricas (nenhum ID de imagem)."""
    import tempfile
    batch = Path(batch)
    lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    logs = sorted([*(batch / "logs").glob("*.out"), *(batch / "logs").glob("*.err")])
    if lote["teste"] or not logs or not secret("GITHUB_TOKEN"):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        for path in logs:
            shutil.copy(path, Path(tmp) / path.name)
        ok = push_results(tmp, f"resultados/{lote['nome']}/logs", f"logs: {lote['nome']}")
    print("GitHub (logs):", "enviados" if ok else "FALHOU")
    return ok


if __name__ == "__main__":
    import sys
    from macdeg.ambiente import load_dotenv
    load_dotenv()
    if len(sys.argv) == 3 and sys.argv[1] == "logs":
        push_logs(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "analisar":
        analyze(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print("uso: python -m macdeg.analysis analisar <lote> [dados] | logs <lote>")
