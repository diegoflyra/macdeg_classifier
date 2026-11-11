"""Compara os cliques com o detector automático. Só produz agregados, sem identificar imagens.

Uso: python -m macdeg.annotation_report [--clicks annotations/disc_clicks.csv] [--cache cache]

Distâncias em DD (diâmetros de disco). Um erro acima de 0,5 DD põe o ponto fora do disco.
Essa medida supõe que o clique é a verdade; a repetibilidade indica o quanto o próprio clique varia.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from macdeg.annotate import load_answers
from macdeg.localization import DD_PRIOR_R, DISC_TO_FOVEA

WRONG_DD = 0.5


def compare(clicks: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por imagem marcada, com o erro do disco e o deslocamento que o clique daria ao recorte."""
    images = metadata.drop_duplicates("file")
    table = clicks[clicks["repeat"] == 0].merge(images, on="file", how="inner")
    dd = DD_PRIOR_R * table["fov_radius"]
    x = pd.to_numeric(table["x_native"], errors="coerce")  # vazio em "não consigo ver"
    y = pd.to_numeric(table["y_native"], errors="coerce")

    auto_x = table["fov_cx"] + table["disc_dx"] * table["fov_radius"]
    auto_y = table["fov_cy"] + table["disc_dy"] * table["fov_radius"]
    table["disc_error_dd"] = np.hypot(x - auto_x, y - auto_y) / dd

    # O lado vem do próprio clique: o disco fica no lado nasal da imagem.
    sign = np.sign(x - table["fov_cx"])
    fovea_x = x + sign * DISC_TO_FOVEA[0] * table["fov_radius"]
    fovea_y = y + DISC_TO_FOVEA[1] * table["fov_radius"]
    used_x = table["fov_cx"] + table["fovea_dx"] * table["fov_radius"]
    used_y = table["fov_cy"] + table["fovea_dy"] * table["fov_radius"]
    table["crop_shift_dd"] = np.hypot(fovea_x - used_x, fovea_y - used_y) / dd
    return table


def _summary(group: pd.DataFrame) -> dict:
    clicked = group[group["status"] == "clicked"]
    result = {"n": int(len(group)), "not_visible": round(float((group["status"] == "not_visible").mean()), 4)}
    if len(clicked):
        result.update(
            disc_error_median_dd=round(float(clicked["disc_error_dd"].median()), 3),
            disc_error_p95_dd=round(float(clicked["disc_error_dd"].quantile(0.95)), 3),
            disc_wrong_frac=round(float((clicked["disc_error_dd"] > WRONG_DD).mean()), 4),
            crop_shift_median_dd=round(float(clicked["crop_shift_dd"].median()), 3),
            crop_shift_over_half_dd_frac=round(float((clicked["crop_shift_dd"] > WRONG_DD).mean()), 4),
        )
    return result


def repeatability(clicks: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    first = clicks[clicks["repeat"] == 0].set_index("file")
    second = clicks[clicks["repeat"] == 1].set_index("file")
    common = first.index.intersection(second.index)
    if len(common) == 0:
        return {"n": 0}
    a, b = first.loc[common], second.loc[common]
    both = (a["status"] == "clicked") & (b["status"] == "clicked")
    radius = metadata.drop_duplicates("file").set_index("file").loc[common, "fov_radius"]
    distance = np.hypot(pd.to_numeric(a["x_native"][both]) - pd.to_numeric(b["x_native"][both]),
                        pd.to_numeric(a["y_native"][both]) - pd.to_numeric(b["y_native"][both]))
    distance = distance / (DD_PRIOR_R * radius[both])
    result = {"n": int(len(common)), "same_status": round(float((a["status"] == b["status"]).mean()), 4)}
    if len(distance):
        result.update(median_dd=round(float(distance.median()), 3), p95_dd=round(float(distance.quantile(0.95)), 3))
    return result


def build_report(clicks: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    table = compare(clicks, metadata)
    report: dict = {"marked_images": int(len(table)), "total_images": int(metadata["file"].nunique()),
                    "overall": _summary(table), "by_AMD": {}, "by_AMD_and_flag": {}}
    for amd, group in table.groupby("AMD"):
        report["by_AMD"][str(amd)] = _summary(group)
        for uncertain, part in group.groupby("disc_uncertain"):
            report["by_AMD_and_flag"][f"{amd}_{'incerto' if uncertain else 'confiante'}"] = _summary(part)
    report["repeatability"] = repeatability(clicks, metadata)
    seconds = clicks["seconds"].astype(float)
    report["median_seconds"] = round(float(seconds.median()), 2) if len(seconds) else None
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clicks", type=Path, default=Path("annotations/disc_clicks.csv"))
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/annotation/report.json"))
    args = ap.parse_args()

    answers = load_answers(args.clicks)
    if not answers:
        raise SystemExit("Nenhuma marcação encontrada.")
    clicks = pd.DataFrame(answers.values())
    clicks["repeat"] = clicks["repeat"].astype(int)
    metadata = pd.read_csv(args.cache / "metadata.csv")
    report = build_report(clicks, metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
