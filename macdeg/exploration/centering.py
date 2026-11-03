"""Mede a centralização na mácula: acha o disco e localiza a mácula em médias alinhadas pelo disco.

Coordenadas: origem no centro do campo de visão, unidade = raio do campo (R). Em `cdx`, as
imagens estão espelhadas para que o disco fique em dx > 0 (lado nasal).
"""
from __future__ import annotations

import json
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from macdeg.exploration.inspection import describe
from macdeg.metadata import load_metadata

THUMB_R = 128  # raio do campo na miniatura, em pixels
THUMB = 2 * THUMB_R + 32  # margem de 16 px em volta do campo
THUMB_C = THUMB // 2


def make_thumb(args: tuple[str, float, float, float]) -> tuple[np.ndarray, list[str]]:
    """Miniatura z-normalizada do canal verde, centrada no círculo do campo."""
    path, cx, cy, radius = args
    raw = Path(path).read_bytes()
    chunks: list[str] = []
    pos = 8
    while pos < len(raw):
        length, kind = struct.unpack(">I4s", raw[pos:pos + 8])
        if kind not in (b"IHDR", b"IDAT", b"IEND"):
            chunks.append(kind.decode("latin1"))
        pos += length + 12
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    scale = THUMB_R / radius
    matrix = np.array([[scale, 0, THUMB_C - cx * scale], [0, scale, THUMB_C - cy * scale]])
    green = cv2.warpAffine(img[..., 1], matrix, (THUMB, THUMB), flags=cv2.INTER_AREA,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
    yy, xx = np.mgrid[:THUMB, :THUMB]
    mask = ((np.hypot(xx - THUMB_C, yy - THUMB_C) < 0.95 * THUMB_R) & (green > 3)).astype(np.float32)

    # Remove a iluminação de baixa frequência e reduz os vasos finos.
    def masked_blur(sigma: float) -> np.ndarray:
        num = cv2.GaussianBlur(green * mask, (0, 0), sigma)
        return num / np.maximum(cv2.GaussianBlur(mask, (0, 0), sigma), 1e-3)

    high = masked_blur(2.0) - masked_blur(40.0)
    high *= mask
    inside = mask > 0
    high[inside] = (high[inside] - high[inside].mean()) / (high[inside].std() + 1e-6)
    return high.astype(np.float32), chunks


def find_disc(thumb: np.ndarray) -> tuple[float, float, float]:
    """Disco como a mancha clara do tamanho esperado, longe da borda. Devolve dx, dy (em R) e score."""
    response = cv2.GaussianBlur(thumb, (0, 0), 8) - cv2.GaussianBlur(thumb, (0, 0), 24)
    yy, xx = np.mgrid[:THUMB, :THUMB]
    inside = np.hypot(xx - THUMB_C, yy - THUMB_C) < 0.85 * THUMB_R
    y, x = np.unravel_index(np.argmax(np.where(inside, response, -np.inf)), response.shape)
    score = (response[y, x] - response[inside].mean()) / (response[inside].std() + 1e-6)
    return (x - THUMB_C) / THUMB_R, (y - THUMB_C) / THUMB_R, float(score)


def temporal_sector(disc_xy: tuple[float, float], side: float) -> np.ndarray:
    """Região onde a fóvea pode estar: temporal ao disco, a 0.45-0.95 R dele, a até 30 graus."""
    yy, xx = np.mgrid[:THUMB, :THUMB]
    vx = (xx - THUMB_C) / THUMB_R - disc_xy[0]
    vy = (yy - THUMB_C) / THUMB_R - disc_xy[1]
    dist = np.hypot(vx, vy)
    angle = np.degrees(np.abs(np.arctan2(vy, -side * vx)))
    in_field = np.hypot(xx - THUMB_C, yy - THUMB_C) < 0.85 * THUMB_R
    return (dist > 0.45) & (dist < 0.95) & (angle < 30) & in_field


def find_fovea(thumb: np.ndarray, disc_xy: tuple[float, float], side: float) -> tuple[float, float, float]:
    """Mancha escura no setor temporal, depois de remover os vasos por fechamento morfológico."""
    closed = cv2.morphologyEx(thumb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    response = cv2.GaussianBlur(closed, (0, 0), 24) - cv2.GaussianBlur(closed, (0, 0), 6)
    sector = temporal_sector(disc_xy, side)
    if not sector.any():
        return np.nan, np.nan, np.nan
    y, x = np.unravel_index(np.argmax(np.where(sector, response, -np.inf)), response.shape)
    score = (response[y, x] - response[sector].mean()) / (response[sector].std() + 1e-6)
    return (x - THUMB_C) / THUMB_R, (y - THUMB_C) / THUMB_R, float(score)


def disc_aligned_fovea(thumbs: np.ndarray, disc: np.ndarray, side: np.ndarray) -> dict:
    """Média com o disco na origem e nasal em +x. O ponto mais escuro do setor temporal é a mácula."""
    aligned = []
    for thumb, (dx, dy), s in zip(thumbs, disc, side):
        canonical = thumb if s > 0 else thumb[:, ::-1]
        cdx = dx if s > 0 else -dx
        matrix = np.array([[1.0, 0, -cdx * THUMB_R], [0, 1.0, -dy * THUMB_R]])
        aligned.append(cv2.warpAffine(canonical, matrix, (THUMB, THUMB)))
    mean = cv2.GaussianBlur(np.mean(aligned, axis=0), (0, 0), 4)
    sector = temporal_sector((0.0, 0.0), 1.0)
    y, x = np.unravel_index(np.argmin(np.where(sector, mean, np.inf)), mean.shape)
    return {"n": len(aligned), "fovea_from_disc_R": [round((x - THUMB_C) / THUMB_R, 3), round((y - THUMB_C) / THUMB_R, 3)],
            "dark_value": round(float(mean[y, x]), 3), "sector_mean": round(float(mean[sector].mean()), 3),
            "sector_std": round(float(mean[sector].std()), 3)}


def run_centering(root: Path | str = "hyamd", out: Path | str = "outputs/step0", workers: int = 6) -> dict:
    root, out = Path(root), Path(out)
    meta = load_metadata(root)
    fov = pd.read_csv(out / "per_image.csv")
    table = meta.merge(fov, on="file", validate="one_to_one")

    thumbs_path = out / "thumbs.npz"
    if thumbs_path.exists():
        saved = np.load(thumbs_path, allow_pickle=True)
        thumbs, chunk_counts = saved["thumbs"], json.loads(str(saved["chunks"]))
    else:
        jobs = [(str(root / "Images" / r.file), r.cx, r.cy, r.radius) for r in table.itertuples()]
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(make_thumb, jobs, chunksize=8))
        thumbs = np.stack([r[0] for r in results])
        chunk_counts = pd.Series([c for r in results for c in r[1]], dtype=str).value_counts().to_dict()
        np.savez(thumbs_path, thumbs=thumbs, chunks=json.dumps(chunk_counts))

    # A lateralidade vem da posição do disco na imagem, não da coluna `side`.
    disc = np.array([find_disc(t) for t in thumbs])
    table["disc2_dx"], table["disc2_dy"], table["disc2_score"] = disc[:, 0], disc[:, 1], disc[:, 2]
    table["disc2_r"] = np.hypot(disc[:, 0], disc[:, 1])
    table["image_side"] = np.where(disc[:, 0] > 0, "R", "L")
    side_sign = np.where(disc[:, 0] > 0, 1.0, -1.0)
    fovea = np.array([find_fovea(t, (d[0], d[1]), s) for t, d, s in zip(thumbs, disc, side_sign)])
    table["fovea2_dx"], table["fovea2_dy"], table["fovea2_score"] = fovea[:, 0], fovea[:, 1], fovea[:, 2]
    table["fovea2_r"] = np.hypot(fovea[:, 0], fovea[:, 1])
    table["fovea2_cdx"] = fovea[:, 0] * side_sign
    table["disc2_cdx"] = disc[:, 0] * side_sign

    confident = table["disc2_score"] >= table["disc2_score"].quantile(0.25)
    usable = ~table["side_anomaly"]
    rep: dict = {
        "png_ancillary_chunks": chunk_counts,
        "disc_score": describe(table["disc2_score"]),
        "disc_r": describe(table["disc2_r"]),
        "laterality_agrees_with_column": {
            "all": round(float((table.loc[usable, "image_side"] == table.loc[usable, "side"]).mean()), 3),
            "top_75pct_disc_score": round(float((table.loc[usable & confident, "image_side"]
                                                 == table.loc[usable & confident, "side"]).mean()), 3),
            "disagree_n_top_75pct": int((table.loc[usable & confident, "image_side"]
                                         != table.loc[usable & confident, "side"]).sum()),
        },
        "anomalous_files": [{"name_side": r.name_side, "column_side": r.side, "image_side": r.image_side,
                             "disc_score": round(r.disc2_score, 2)}
                            for r in table[table["side_anomaly"]].itertuples()],
    }

    def group_summary(g: pd.DataFrame) -> dict:
        keep = g.index[g["disc2_score"] >= table["disc2_score"].quantile(0.25)]
        vector = disc_aligned_fovea(thumbs[keep], disc[keep, :2], side_sign[keep])
        median_disc = g.loc[keep, ["disc2_cdx", "disc2_dy"]].median().to_numpy()
        implied = median_disc + np.array(vector["fovea_from_disc_R"])
        return {
            "n": len(g), "patients": int(g["patient_id"].nunique()),
            "disc_r": describe(g["disc2_r"]),
            "disc_cdx": describe(g.loc[keep, "disc2_cdx"]),
            "disc_dy": describe(g.loc[keep, "disc2_dy"]),
            "disc_aligned_mean": vector,
            "fovea_implied_xy_R": [round(float(v), 3) for v in implied],
            "fovea_per_image_r": describe(g.loc[keep, "fovea2_r"]),
            "fovea_per_image_cdx": describe(g.loc[keep, "fovea2_cdx"]),
            "fovea_per_image_dy": describe(g.loc[keep, "fovea2_dy"]),
            "fovea_score": describe(g.loc[keep, "fovea2_score"]),
            "fovea_within_R": {str(t): round(float((g.loc[keep, "fovea2_r"] < t).mean()), 3) for t in (0.1, 0.2, 0.3, 0.4)},
        }

    rep["by_width"] = {str(int(k)): group_summary(g) for k, g in table.groupby("width")}
    rep["by_AMD"] = {str(k): group_summary(g) for k, g in table.groupby("AMD")}
    rep["by_width_AMD_images"] = {f"{int(w)}|{a}": int(c) for (w, a), c in table.groupby(["width", "AMD"]).size().items()}
    rep["eyes_with_both_widths"] = int((table.groupby("eye_id")["width"].nunique() > 1).sum())
    rep["patients_with_both_widths"] = int((table.groupby("patient_id")["width"].nunique() > 1).sum())
    rep["circle_fit_suspect"] = {"rms_gt_20": int((table["circle_rms"] > 20).sum()),
                                 "radius_outside_920_1010": int((~table["radius"].between(920, 1010)).sum())}
    rep["patients_per_any_AMD"] = {str(k): int(table.loc[table["AMD"] == k, "patient_id"].nunique()) for k in (0, 1, 2)}
    rep["eyes_per_AMD"] = {str(k): int(table.loc[table["AMD"] == k, "eye_id"].nunique()) for k in (0, 1, 2)}

    table.to_csv(out / "per_image_centering.csv", index=False)
    (out / "centering_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def contact_sheet_by_width(root: Path | str = "hyamd", out: Path | str = "outputs/step0",
                           per_width: int = 15, vector: tuple[float, float] = (-0.61, 0.09)) -> Path:
    """Grade local para inspeção humana, separada por formato.

    Cruz branca: centro do campo. Ciano: disco. Amarelo: fóvea pela mancha escura.
    Magenta: fóvea prevista por disco + vetor médio.
    """
    root, out = Path(root), Path(out)
    table = pd.read_csv(out / "per_image_centering.csv", dtype={"patient_id": str})
    tile, cols = 320, 5
    rows = []
    for _, group in table.groupby("width"):
        picks = group.sample(per_width, random_state=0)
        tiles = []
        for r in picks.itertuples():
            img = cv2.imread(str(root / "Images" / r.file), cv2.IMREAD_COLOR)
            f = tile / max(img.shape[:2])
            thumb = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            canvas = np.zeros((tile, tile, 3), np.uint8)
            canvas[:thumb.shape[0], :thumb.shape[1]] = thumb
            center = np.array([r.cx, r.cy]) * f
            radius = r.radius * f
            point = lambda dx, dy: tuple(int(v) for v in center + np.array([dx, dy]) * radius)
            sign = 1.0 if r.disc2_dx > 0 else -1.0
            cv2.drawMarker(canvas, point(0, 0), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
            cv2.circle(canvas, point(r.disc2_dx, r.disc2_dy), int(0.13 * radius), (255, 255, 0), 1)
            cv2.circle(canvas, point(r.fovea2_dx, r.fovea2_dy), int(0.08 * radius), (0, 255, 255), 1)
            cv2.circle(canvas, point(r.disc2_dx + sign * vector[0], r.disc2_dy + vector[1]),
                       int(0.05 * radius), (255, 0, 255), 1)
            tiles.append(canvas)
        rows.extend(np.hstack(tiles[i:i + cols]) for i in range(0, per_width, cols))
    path = out / "contact_sheet_by_width.png"
    cv2.imwrite(str(path), np.vstack(rows))
    return path


if __name__ == "__main__":
    print(json.dumps(run_centering(), separators=(",", ":"), default=str))
    contact_sheet_by_width()
