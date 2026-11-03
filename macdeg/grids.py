"""Grades de verificação visual. Contêm imagens de pacientes: ficam só em disco local.

Uso: python -m macdeg.grids [--cache cache] [--out outputs/grids]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from macdeg.augment import Augmenter, get_variant, load_patch, load_registry


def _tile(image: np.ndarray, size: int) -> np.ndarray:
    scale = size / max(image.shape[:2])
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:small.shape[0], :small.shape[1]] = small
    return canvas


def _sheet(tiles: list[np.ndarray], cols: int = 4) -> np.ndarray:
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def _annotated(row: pd.Series, size: int = 360) -> np.ndarray:
    """Original com caixa do ROI (verde), fóvea (magenta), disco (ciano) e centro do campo (branco)."""
    image = cv2.imread(row["original_path"], cv2.IMREAD_COLOR)
    scale = size / max(image.shape[:2])
    tile = _tile(image, size)
    to_px = lambda x, y: (int(x * scale), int(y * scale))
    half = row["roi_half_native"]
    center_x = row["crop_x0"] + row["crop_side_native"] / 2
    center_y = row["crop_y0"] + row["crop_side_native"] / 2
    cv2.rectangle(tile, to_px(center_x - half, center_y - half), to_px(center_x + half, center_y + half), (0, 255, 0), 1)
    cv2.circle(tile, to_px(row["fovea_x"], row["fovea_y"]), 6, (255, 0, 255), 1)
    cv2.drawMarker(tile, to_px(row["fov_cx"], row["fov_cy"]), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
    if pd.notna(row["disc_dx"]):
        disc = to_px(row["fov_cx"] + row["disc_dx"] * row["fov_radius"], row["fov_cy"] + row["disc_dy"] * row["fov_radius"])
        cv2.circle(tile, disc, int(0.12 * row["fov_radius"] * scale), (255, 255, 0), 1)
    return tile


def crop_box_grid(metadata: pd.DataFrame, out: Path, roi: str = "R2.0") -> None:
    rows = metadata[metadata["roi"] == roi]
    pick = rows.sample(min(16, len(rows)), random_state=0)
    cv2.imwrite(str(out / f"caixa_recorte_{roi}.png"), _sheet([_annotated(r) for _, r in pick.iterrows()]))


def localization_grids(metadata: pd.DataFrame, out: Path, roi: str = "R2.0") -> None:
    """Uma grade de cada caso: disco incerto e disco confiante nas classes com DMRI."""
    rows = metadata[metadata["roi"] == roi]
    cases = {"disco_incerto": rows[rows["disc_uncertain"]],
             "disco_confiante_com_DMRI": rows[~rows["disc_uncertain"] & (rows["AMD"] > 0)],
             "disco_confiante_controle": rows[~rows["disc_uncertain"] & (rows["AMD"] == 0)]}
    for name, group in cases.items():
        if len(group):
            pick = group.sample(min(16, len(group)), random_state=1)
            cv2.imwrite(str(out / f"localizacao_{name}.png"), _sheet([_annotated(r) for _, r in pick.iterrows()]))


def level_grids(metadata: pd.DataFrame, cache: Path, out: Path, roi: str = "R2.0", resolution: str = "S512") -> None:
    registry = load_registry(flag_summary=cache / "flag_summary.json")
    recipes = {"A0": "A0", "A1-G1": "A1-G1", "A2-G1": "A2-G1", "A3-G1": "A3-G1"}
    if registry["rois"][roi].get("rotation"):
        recipes.update({"A1-G2": "A1-G2", "A2-G2": "A2-G2", "A3-G2": "A3-G2"})
    rows = metadata[metadata["roi"] == roi]
    pick = rows.sample(min(16, len(rows)), random_state=2)
    for name, recipe in recipes.items():
        variant = get_variant(f"{roi}_{resolution}_{recipe}", registry)
        augmenter = Augmenter(variant, train=True, registry=registry)
        rng = np.random.default_rng(3)
        tiles = []
        for _, row in pick.iterrows():
            output = augmenter(load_patch(cache, variant, row["file"]), rng, bool(row["rotation_ok"]))
            tiles.append(_tile((output * 255 + 0.5).astype(np.uint8)[..., ::-1], 300))
        cv2.imwrite(str(out / f"nivel_{name}_{roi}_{resolution}.png"), _sheet(tiles))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/grids"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.cache / "metadata.csv", dtype={"patient_id": str})
    crop_box_grid(metadata, args.out)
    localization_grids(metadata, args.out)
    level_grids(metadata, args.cache, args.out)
    print(f"{len(list(args.out.glob('*.png')))} grades em {args.out}")


if __name__ == "__main__":
    main()
