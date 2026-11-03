"""Passo 0: inspeção da base HYAMD.

Privacidade: a saída no terminal traz apenas agregados, nunca image_id/patient_id
(o nome do arquivo contém o patient_id). Detalhes por imagem e a grade visual ficam
em `outputs/step0/`, que não é versionado. Os originais são abertos só para leitura.

A centralização na mácula é medida em `macdeg/exploration/centering.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WORK_RADIUS = 240  # raio-alvo aproximado do campo de visão na análise, em pixels


def analyse(path_str: str) -> dict:
    path = Path(path_str)
    raw = path.read_bytes()
    out: dict = {
        "file": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        out["readable"] = False
        return out
    out.update(readable=True, height=img.shape[0], width=img.shape[1],
               channels=1 if img.ndim == 2 else img.shape[2], dtype=str(img.dtype))
    if img.ndim != 3 or img.shape[2] < 3:
        return out
    bgr = img[..., :3]
    if bgr.dtype != np.uint8:
        bgr = (bgr / 257).astype(np.uint8)
    h, w = bgr.shape[:2]

    # Campo de visão: pixels não pretos, maior componente, buracos preenchidos.
    full_max = bgr.max(axis=2)
    binary = (cv2.GaussianBlur(full_max, (0, 0), 3) > 12).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n < 2:
        out["fov_ok"] = False
        return out
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fov = (cc == largest).astype(np.uint8)
    contours, _ = cv2.findContours(fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)[:, 0, :]
    fov = cv2.drawContours(np.zeros_like(fov), [contour], -1, 1, cv2.FILLED)

    # Círculo ajustado só com a borda real: pontos na moldura da imagem indicam
    # truncamento, não o limite circular da câmera.
    margin = 3
    arc = contour[(contour[:, 0] > margin) & (contour[:, 0] < w - 1 - margin)
                  & (contour[:, 1] > margin) & (contour[:, 1] < h - 1 - margin)]
    pts = arc if len(arc) > 100 else contour
    design = np.c_[2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))]
    sol = np.linalg.lstsq(design, (pts.astype(np.float64) ** 2).sum(axis=1), rcond=None)[0]
    cx, cy = float(sol[0]), float(sol[1])
    radius = float(np.sqrt(sol[2] + cx * cx + cy * cy))
    residual = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - radius)
    ys, xs = np.nonzero(fov)
    out.update(
        fov_ok=True, cx=cx, cy=cy, radius=radius,
        circle_rms=float(np.sqrt((residual ** 2).mean())),
        fov_area_frac=float(fov.mean()),
        bbox_left=int(xs.min()), bbox_right=int(xs.max()),
        bbox_top=int(ys.min()), bbox_bottom=int(ys.max()),
        cut_top=bool(ys.min() <= margin), cut_bottom=bool(ys.max() >= h - 1 - margin),
        cut_left=bool(xs.min() <= margin), cut_right=bool(xs.max() >= w - 1 - margin),
    )

    # Texto ou marcas queimadas fora do campo.
    outside = fov == 0
    out["outside_bright_frac"] = (
        float((full_max[outside] > 40).mean()) if outside.any() else 0.0
    )

    # Trabalho em escala reduzida, normalizada pelo raio do campo.
    scale = WORK_RADIUS / radius
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(fov, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    scx, scy, sr = cx * scale, cy * scale, radius * scale
    green = small[..., 1].astype(np.float32)
    red = small[..., 2].astype(np.float32)
    weight = mask.astype(np.float32)

    def masked_blur(channel: np.ndarray, sigma: float) -> np.ndarray:
        num = cv2.GaussianBlur(channel * weight, (0, 0), sigma)
        den = cv2.GaussianBlur(weight, (0, 0), sigma)
        return num / np.maximum(den, 1e-3)

    out["green_mean"] = float(green[mask > 0].mean())
    out["red_sat_frac"] = float((small[..., 2][mask > 0] >= 250).mean())

    # Disco: mancha clara em verde + vermelho, com tamanho aproximado do disco.
    combined = green + 0.5 * red
    resp = masked_blur(combined, 0.05 * sr) - masked_blur(combined, 0.25 * sr)
    resp[mask == 0] = -np.inf
    dy, dx = np.unravel_index(int(np.argmax(resp)), resp.shape)
    finite = resp[mask > 0]
    out["disc_score"] = float((resp[dy, dx] - np.median(finite)) / (finite.std() + 1e-6))
    out["disc_dx"] = float((dx - scx) / sr)
    out["disc_dy"] = float((dy - scy) / sr)

    # Fóvea: a mancha mais escura, sem prior de centro e fora da região do disco.
    # O fechamento morfológico remove os vasos, que são mais finos que a mácula.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.04 * sr) + 1,) * 2)
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    dark = masked_blur(closed, 0.25 * sr) - masked_blur(closed, 0.06 * sr)
    yy, xx = np.mgrid[: small.shape[0], : small.shape[1]]
    valid = mask > 0
    valid &= np.hypot(xx - scx, yy - scy) < 0.80 * sr
    valid &= np.hypot(xx - dx, yy - dy) > 0.30 * sr
    if valid.any():
        scored = np.where(valid, dark, -np.inf)
        fy, fx = np.unravel_index(int(np.argmax(scored)), scored.shape)
        out["fovea_dx"] = float((fx - scx) / sr)
        out["fovea_dy"] = float((fy - scy) / sr)
        out["fovea_score"] = float((dark[fy, fx] - dark[valid].mean()) / (dark[valid].std() + 1e-6))
        out["disc_fovea_dist"] = float(np.hypot(fx - dx, fy - dy) / sr)
    return out


def describe(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    if s.empty:
        return {"n": 0}
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(3).tolist()
    return dict(zip(["n", "min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"],
                    [len(s), round(s.min(), 3), *q, round(s.max(), 3)]))


def run_inspection(root: Path = Path("hyamd"), out: Path = Path("outputs/step0"),
                   workers: int = 6) -> dict:
    """Mede a base (reaproveita `per_image.csv` se existir) e devolve só agregados."""
    args = argparse.Namespace(root=Path(root), out=Path(out), workers=workers)
    args.out.mkdir(parents=True, exist_ok=True)
    images_dir = args.root / "Images"
    per_image_csv = args.out / "per_image.csv"

    if per_image_csv.exists():
        df = pd.read_csv(per_image_csv)
    else:
        paths = sorted(str(p) for p in images_dir.iterdir() if p.suffix.lower() == ".png")
        with ProcessPoolExecutor(args.workers) as pool:
            df = pd.DataFrame(list(pool.map(analyse, paths, chunksize=8)))
        df.to_csv(per_image_csv, index=False)

    rep: dict = {"n_files": len(df), "unreadable": int((df["readable"] != True).sum())}  # noqa: E712

    # Integridade em relação ao SHA256SUMS publicado.
    expected: dict[str, str] = {}
    for line in (args.root / "SHA256SUMS.txt").read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            expected[Path(parts[1].strip().lstrip("*")).name] = parts[0].lower()
    in_sums = df["file"].isin(expected)
    rep["sha256"] = {
        "listed_images": int(in_sums.sum()),
        "mismatch": int((df.loc[in_sums, "sha256"] != df.loc[in_sums, "file"].map(expected)).sum()),
        "not_listed": int((~in_sums).sum()),
        "duplicate_groups_of_identical_bytes": int((df["sha256"].value_counts() > 1).sum()),
    }

    # Formato.
    rep["size_counts"] = {f"{int(h)}x{int(w)}(HxW)": int(c) for (h, w), c in
                          df.groupby(["height", "width"]).size().items()}
    rep["channels"] = df["channels"].value_counts().to_dict()
    rep["dtype"] = df["dtype"].value_counts().to_dict()
    rep["file_mb"] = describe(df["bytes"] / 1e6)

    # Metadados e correspondência com os arquivos.
    labels = pd.read_csv(args.root / "labels" / "labels.csv", dtype={"image_id": str, "patient_id": str})
    stems = df["file"].str.replace(r"\.png$", "", regex=True, case=False)
    m = {
        "rows": len(labels), "columns": list(labels.columns),
        "nulls": {k: int(v) for k, v in labels.isna().sum().items()},
        "duplicate_image_id": int(labels["image_id"].duplicated().sum()),
        "ids_without_file": int((~labels["image_id"].isin(stems)).sum()),
        "files_without_row": int((~stems.isin(labels["image_id"])).sum()),
        "patients": int(labels["patient_id"].nunique()),
        "side_values": labels["side"].value_counts(dropna=False).to_dict(),
        "AMD_values": {str(k): int(v) for k, v in labels["AMD"].value_counts(dropna=False).sort_index().items()},
        "sex_values": labels["sex"].value_counts(dropna=False).to_dict(),
        "age": describe(labels["age"]),
    }
    per_patient = labels.groupby("patient_id")
    per_eye = labels.groupby(["patient_id", "side"])
    m["images_per_patient"] = describe(per_patient.size())
    m["eyes"] = int(per_eye.ngroups)
    m["images_per_eye"] = describe(per_eye.size())
    m["eyes_per_patient"] = per_patient["side"].nunique().value_counts().sort_index().to_dict()
    m["eye_image_count_hist"] = per_eye.size().clip(upper=8).value_counts().sort_index().to_dict()
    m["patients_with_multi_label"] = int((per_patient["AMD"].nunique() > 1).sum())
    m["eyes_with_multi_label"] = int((per_eye["AMD"].nunique() > 1).sum())
    m["patients_with_multi_sex"] = int((per_patient["sex"].nunique() > 1).sum())
    m["patients_with_multi_age"] = int((per_patient["age"].nunique() > 1).sum())
    m["eyes_with_multi_age"] = int((per_eye["age"].nunique() > 1).sum())
    m["patient_age_span"] = describe(per_patient["age"].max() - per_patient["age"].min())
    m["patients_per_max_AMD"] = {str(k): int(v) for k, v in per_patient["AMD"].max().value_counts().sort_index().items()}
    m["eyes_per_max_AMD"] = {str(k): int(v) for k, v in per_eye["AMD"].max().value_counts().sort_index().items()}
    m["age_by_AMD"] = {str(k): describe(g["age"]) for k, g in labels.groupby("AMD")}
    m["sex_by_AMD"] = {str(k): g["sex"].value_counts().to_dict() for k, g in labels.groupby("AMD")}

    # Convenção do nome do arquivo.
    parsed = labels["image_id"].str.extract(r"^(?P<pid>\d+)_(?P<side>[A-Za-z]+)(?:_(?P<suffix>.+?)_?)?$")
    m["name_regex_fail"] = int(parsed["pid"].isna().sum())
    m["name_pid_matches_column"] = int((parsed["pid"] == labels["patient_id"]).sum())
    m["name_side_matches_column"] = int((parsed["side"] == labels["side"]).sum())
    m["name_side_vs_column"] = {f"{a}->{b}": int(c) for (a, b), c in
                                pd.crosstab(parsed["side"], labels["side"]).stack().items() if c}
    m["suffix_present"] = int(parsed["suffix"].notna().sum())
    numeric_suffix = pd.to_numeric(parsed["suffix"], errors="coerce")
    m["suffix_numeric"] = int(numeric_suffix.notna().sum())
    m["suffix_digits"] = parsed["suffix"].dropna().str.len().value_counts().sort_index().to_dict()
    m["suffix_summary"] = describe(numeric_suffix)
    m["suffix_within_eye_unique"] = bool(
        parsed.assign(pid2=labels["patient_id"], side2=labels["side"])
        .dropna(subset=["suffix"]).duplicated(["pid2", "side2", "suffix"]).sum() == 0)
    # O sufixo acompanha a idade dentro de um olho? Se sim, pode ser ordem temporal.
    temp = labels.assign(suffix=numeric_suffix).dropna(subset=["suffix"])
    corr = [g["suffix"].corr(g["age"], method="spearman") for _, g in temp.groupby(["patient_id", "side"])
            if len(g) >= 2 and g["age"].nunique() >= 2]
    m["suffix_age_spearman_within_eye"] = describe(pd.Series(corr, dtype=float))
    rep["metadata"] = m

    # Tabela geral de contagem por cruzamento com images_per_patient_id.csv.
    ipp = pd.read_csv(args.root / "labels" / "images_per_patient_id.csv", dtype={"patient_id": str})
    joined = ipp.set_index("patient_id")["num_images"].reindex(per_patient.size().index)
    rep["images_per_patient_csv_mismatch"] = int((joined != per_patient.size()).sum())

    # Campo de visão.
    ok = df[df["fov_ok"] == True]  # noqa: E712
    rep["fov"] = {
        "n_ok": len(ok),
        "radius_px": describe(ok["radius"]),
        "center_offset_x_px": describe(ok["cx"] - (ok["width"] - 1) / 2),
        "center_offset_y_px": describe(ok["cy"] - (ok["height"] - 1) / 2),
        "circle_rms_px": describe(ok["circle_rms"]),
        "area_frac": describe(ok["fov_area_frac"]),
        "cut": {k: int(ok[f"cut_{k}"].sum()) for k in ["top", "bottom", "left", "right"]},
        "outside_bright_frac": describe(ok["outside_bright_frac"]),
        "images_with_outside_bright_gt_0.1pct": int((ok["outside_bright_frac"] > 0.001).sum()),
        "green_mean": describe(ok["green_mean"]),
        "red_saturated_frac": describe(ok["red_sat_frac"]),
    }

    (args.out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("hyamd"))
    ap.add_argument("--out", type=Path, default=Path("outputs/step0"))
    ap.add_argument("--workers", type=int, default=6)
    cli = ap.parse_args()
    print(json.dumps(run_inspection(cli.root, cli.out, cli.workers), indent=1, default=str))
