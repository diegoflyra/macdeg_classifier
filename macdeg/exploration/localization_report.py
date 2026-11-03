"""Roda a localização em toda a base (só leitura) e imprime agregados, sem IDs."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from macdeg.exploration.inspection import describe
from macdeg.localization import locate
from macdeg.metadata import load_metadata

OUT = Path("outputs/step1")


def _job(args: tuple[str, str, str]) -> dict:
    path, file, side = args
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    _, row = locate(bgr, side)
    row.update(file=file, width=bgr.shape[1])
    return row


def summarize(table: pd.DataFrame) -> dict:
    sign = np.where(table["side_from_image"] == "R", 1.0, -1.0)
    table = table.assign(cdx=table["disc_dx"] * sign, fovea_cdx=table["fovea_dx"] * sign,
                         dd_over_r=table["dd_measured_px"] / table["fov_radius"])
    any_flag = table["disc_uncertain"] | table["side_conflict"] | table["fovea_disagree"] | table["fov_suspect"]

    def block(g: pd.DataFrame) -> dict:
        sure = g[~g["disc_uncertain"]]
        return {
            "n": len(g),
            "disc_uncertain": round(float(g["disc_uncertain"].mean()), 3),
            "side_conflict": round(float(g["side_conflict"].mean()), 3),
            "fovea_disagree": round(float(g["fovea_disagree"].mean()), 3),
            "any_flag": round(float(any_flag[g.index].mean()), 3),
            "disc_cdx_sure": describe(sure["cdx"]),
            "disc_dy_sure": describe(sure["disc_dy"]),
            "fovea_cdx": describe(g["fovea_cdx"]),
            "fovea_dy": describe(g["fovea_dy"]),
        }

    return {
        "fov_radius": describe(table["fov_radius"]),
        "fov_rms": describe(table["fov_rms"]),
        "fov_suspect": int(table["fov_suspect"].sum()),
        "dd_valid_frac": round(float(table["dd_valid"].mean()), 3),
        "dd_over_R_valid": describe(table.loc[table["dd_valid"], "dd_over_r"]),
        "dd_spread": describe(table["dd_spread"]),
        "by_width": {str(int(k)): block(g) for k, g in table.groupby("width")},
        "by_AMD": {str(k): block(g) for k, g in table.groupby("AMD")},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    jobs = [(f"hyamd/Images/{r.file}", r.file, r.side) for r in meta.itertuples()]
    with ProcessPoolExecutor(6) as pool:
        rows = list(pool.map(_job, jobs, chunksize=8))
    table = meta.merge(pd.DataFrame(rows), on="file", validate="one_to_one")
    table.to_csv(OUT / "localization.csv", index=False)
    report = summarize(table)
    (OUT / "localization_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, separators=(",", ":"), default=str))


if __name__ == "__main__":
    main()
