"""Métricas do classificador DMRI x sem DMRI.

Métrica principal: média do F2 dentro de cada base (`f2_media_bases`). O F2 pesa a sensibilidade
4 vezes mais que a precisão: perder uma DMRI custa mais que um alarme falso. A média por base
impede que reconhecer a base (câmera, população) infle a nota.

Limiar: nunca escolhido nas mesmas imagens em que o F2 é medido. Em validação cruzada, o limiar de
cada dobra vem das outras dobras (`crossfit`).

Intervalos de confiança: reamostragem de grupos (pacientes), pois imagens do mesmo paciente não
são independentes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# A RFMiD 2.0 tem só 10 DMRI: conta junto com a RFMiD (mesmo grupo de origem) nas métricas por base.
SOURCE_FAMILY = {"rfmid2": "rfmid"}
N_THRESHOLDS = 200


def family(sources: pd.Series) -> pd.Series:
    return sources.map(lambda s: SOURCE_FAMILY.get(s, s))


def _both_classes(labels) -> bool:
    return len(np.unique(labels)) == 2


def auroc(labels, scores) -> float | None:
    return float(roc_auc_score(labels, scores)) if _both_classes(labels) else None


def auprc(labels, scores) -> float | None:
    return float(average_precision_score(labels, scores)) if _both_classes(labels) else None


def fbeta(labels, predicted, beta: float = 2.0) -> float | None:
    labels, predicted = np.asarray(labels).astype(bool), np.asarray(predicted).astype(bool)
    if not labels.any():
        return None
    tp = float((labels & predicted).sum())
    fp = float((~labels & predicted).sum())
    fn = float((labels & ~predicted).sum())
    b2 = beta * beta
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)


def _codes(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    fam = family(frame["source"])
    names = sorted(fam.unique())
    return fam.map({n: i for i, n in enumerate(names)}).to_numpy(), names


def _f2_by_family(labels: np.ndarray, predicted: np.ndarray, codes: np.ndarray, n_families: int) -> np.ndarray:
    """F2 de cada base; NaN se a base não tem as duas classes nessas imagens."""
    labels, predicted = labels.astype(bool), predicted.astype(bool)
    out = np.full(n_families, np.nan)
    for c in range(n_families):
        here = codes == c
        y = labels[here]
        if y.any() and (~y).any():
            p = predicted[here]
            tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
            out[c] = 5 * tp / (5 * tp + 4 * fn + fp)
    return out


def f2_mean(frame: pd.DataFrame, predicted=None) -> float | None:
    """Média do F2 dentro das bases que têm as duas classes nas imagens avaliadas."""
    codes, names = _codes(frame)
    predicted = frame["pred"].to_numpy() if predicted is None else predicted
    values = _f2_by_family(frame["label"].to_numpy(), np.asarray(predicted), codes, len(names))
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def best_threshold(frame: pd.DataFrame) -> float:
    """Limiar que maximiza a média do F2 por base nas imagens dadas (`label`, `prob`, `source`)."""
    probs = frame["prob"].to_numpy()
    candidates = np.unique(np.quantile(probs, np.linspace(0, 1, N_THRESHOLDS + 1)))
    codes, names = _codes(frame)
    labels = frame["label"].to_numpy()
    scores = [np.nanmean(_f2_by_family(labels, probs >= t, codes, len(names))) for t in candidates]
    return float(candidates[int(np.nanargmax(scores))])


def crossfit(frame: pd.DataFrame, fold_column: str) -> pd.DataFrame:
    """Para cada dobra, o limiar vem das outras dobras e é aplicado nela. Acrescenta `threshold` e `pred`."""
    frame = frame.copy()
    for fold in frame[fold_column].unique():
        here = frame[fold_column] == fold
        threshold = best_threshold(frame[~here])
        frame.loc[here, "threshold"] = threshold
        frame.loc[here, "pred"] = (frame.loc[here, "prob"] >= threshold).astype(int)
    frame["pred"] = frame["pred"].astype(int)
    return frame


def group_bootstrap(frame: pd.DataFrame, statistic, n: int = 1000, seed: int = 0) -> list[float] | None:
    """IC de 95% de `statistic(frame_reamostrado)`, reamostrando grupos inteiros."""
    groups = list(frame.groupby("group_id").indices.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        index = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        value = statistic(frame.iloc[index])
        if value is not None and np.isfinite(value):
            values.append(value)
    if len(values) < n * 0.9:
        return None
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def source_only_scores(train: pd.DataFrame, evaluated: pd.DataFrame) -> np.ndarray:
    """Régua da origem: a nota de cada imagem é a fração de DMRI da sua base no treino."""
    rate = train.groupby("source")["label"].mean()
    return evaluated["source"].map(rate).fillna(train["label"].mean()).to_numpy()


def _rates(frame: pd.DataFrame) -> dict:
    y, p = frame["label"].to_numpy().astype(bool), frame["pred"].to_numpy().astype(bool)
    return {"sensibilidade": float(p[y].mean()) if y.any() else None,
            "especificidade": float((~p[~y]).mean()) if (~y).any() else None}


def report(frame: pd.DataFrame, train: pd.DataFrame | None = None, n_bootstrap: int = 1000) -> dict:
    """Relatório completo. `frame`: label, prob, pred, source, group_id, kind (dmri/normal/outra) e,
    se houver, disease_group."""
    labels, probs = frame["label"].to_numpy(), frame["prob"].to_numpy()
    result = {"n": int(len(frame)), "n_dmri": int(labels.sum()),
              "n_normal": int((frame["kind"] == "normal").sum()), "n_outra_doenca": int((frame["kind"] == "outra").sum()),
              "f2_media_bases": f2_mean(frame),
              "f2_media_bases_ic95": group_bootstrap(frame, f2_mean, n_bootstrap),
              "f2_geral": fbeta(labels, frame["pred"]), "auroc": auroc(labels, probs), "auprc": auprc(labels, probs),
              "auroc_ic95": group_bootstrap(frame, lambda f: auroc(f["label"], f["prob"]), n_bootstrap),
              **_rates(frame)}
    negatives = frame[frame["label"] == 0]
    result["especificidade_normais"] = float((negatives.loc[negatives["kind"] == "normal", "pred"] == 0).mean()) \
        if (negatives["kind"] == "normal").any() else None
    result["especificidade_outras_doencas"] = float((negatives.loc[negatives["kind"] == "outra", "pred"] == 0).mean()) \
        if (negatives["kind"] == "outra").any() else None
    if "disease_group" in frame:
        result["especificidade_por_doenca"] = {
            str(g): {"n": int(len(rows)), "especificidade": float((rows["pred"] == 0).mean())}
            for g, rows in negatives[negatives["kind"] == "outra"].groupby("disease_group")}
    if train is not None:
        result["regua_origem_auroc"] = auroc(labels, source_only_scores(train, frame))
    result["por_base"] = {}
    for name, rows in frame.groupby(family(frame["source"])):
        result["por_base"][name] = {"n_dmri": int(rows["label"].sum()), "n_sem_dmri": int((rows["label"] == 0).sum()),
                                    "f2": fbeta(rows["label"], rows["pred"]) if _both_classes(rows["label"]) else None,
                                    "auroc": auroc(rows["label"], rows["prob"]), **_rates(rows)}
    return result


def paired_verdict(reference: pd.DataFrame, candidate: pd.DataFrame, n: int = 1000, seed: int = 0) -> dict:
    """Compara duas configurações nas MESMAS imagens (alinhadas por sample_id), pela diferença da média
    do F2 por base. ganhou = IC95 da diferença todo acima de zero; perdeu = todo abaixo; senão empatou."""
    joined = reference.merge(candidate[["sample_id", "pred"]], on="sample_id", suffixes=("_ref", "_cand"),
                             validate="one_to_one")
    codes, names = _codes(joined)
    labels = joined["label"].to_numpy()
    ref, cand = joined["pred_ref"].to_numpy(), joined["pred_cand"].to_numpy()

    def delta(index):
        a = np.nanmean(_f2_by_family(labels[index], ref[index], codes[index], len(names)))
        b = np.nanmean(_f2_by_family(labels[index], cand[index], codes[index], len(names)))
        return b - a

    groups = list(joined.groupby("group_id").indices.values())
    rng = np.random.default_rng(seed)
    values = [delta(np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])) for _ in range(n)]
    low, high = np.nanpercentile(values, [2.5, 97.5])
    verdict = "ganhou" if low > 0 else "perdeu" if high < 0 else "empatou"
    return {"delta_f2_media_bases": round(float(delta(np.arange(len(joined)))), 4),
            "ic95": [round(float(low), 4), round(float(high), 4)], "veredito": verdict}


def evaluate(predictions: pd.DataFrame, train: pd.DataFrame, n_bootstrap: int = 200) -> dict:
    """Resumo de uma execução, sem limiar (o limiar exige as outras dobras): AUROC geral e por base."""
    labels, scores = predictions["label"].to_numpy(), predictions["prob"].to_numpy()
    result = {"n": int(len(predictions)), "n_dmri": int(labels.sum()), "auroc": auroc(labels, scores),
              "auprc": auprc(labels, scores),
              "auroc_ic95": group_bootstrap(predictions, lambda f: auroc(f["label"], f["prob"]), n_bootstrap),
              "regua_origem_auroc": auroc(labels, source_only_scores(train, predictions)), "por_base": {}}
    for name, rows in predictions.groupby(family(predictions["source"])):
        result["por_base"][name] = {"n_dmri": int(rows["label"].sum()), "auroc": auroc(rows["label"], rows["prob"])}
    return result
