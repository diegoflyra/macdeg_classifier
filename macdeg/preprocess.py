"""Passo 1: pré-processamento determinístico. Gera os caches e o CSV de metadados.

Uso: python -m macdeg.preprocess [--out cache] [--workers 6] [--limit N]

- Reexecutável: retoma de onde parou, com base em `progress.jsonl`.
- Só lê os originais. Checa o espaço em disco antes de começar.
- Erros numa imagem não param a execução: ficam em `errors.csv`.

DD: usa-se 0.247 R em todas as imagens. A medida por imagem foi válida em só 19% da base e
misturá-la tornaria a escala do recorte dependente da qualidade do olho. Ela é gravada como
diagnóstico em `dd_measured_px`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from macdeg.augment import FULL_FIELD_PATCH_MARGIN, ROI_PATCH_MARGIN, load_registry, patch_side
from macdeg.localization import DD_PRIOR_R, Fov, locate
from macdeg.metadata import load_metadata

MASK_STEP = 4  # as buscas de encaixe usam a máscara reduzida
RIM_SAFETY_PX = 8  # a borda do campo escurece: exige esta folga
MAX_SHIFT_HALF = 0.5  # deslocamento máximo da janela, como fração do meio lado do ROI
FLAG_LIMIT = 0.10
POPULATION_RADIUS = 945.0
FREE_SPACE_RESERVE = 5e9
FLAG_COLUMNS = ["disc_uncertain", "side_conflict", "fovea_disagree", "fov_suspect", "margin_black"]


def _nearest(feasible: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    """Ponto viável mais próximo de (x, y), em coordenadas da máscara reduzida."""
    ys, xs = np.nonzero(feasible)
    if len(xs) == 0:
        return None
    i = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return float(xs[i]), float(ys[i])


def fit_window(mask: np.ndarray, fovea_x: float, fovea_y: float, half_px: float) -> dict:
    """Escolhe o centro do ROI: fóvea, ou a menor mudança que faça o necessário caber no campo."""
    small = cv2.resize(mask, None, fx=1 / MASK_STEP, fy=1 / MASK_STEP, interpolation=cv2.INTER_NEAREST)
    fx, fy = fovea_x / MASK_STEP, fovea_y / MASK_STEP
    limit = MAX_SHIFT_HALF * half_px / MASK_STEP
    padded = cv2.copyMakeBorder(small, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    safety = RIM_SAFETY_PX / MASK_STEP

    rotation_center = _nearest(dist >= ROI_PATCH_MARGIN * half_px / MASK_STEP + safety, fx, fy)
    kernel_side = max(1, int(round(2 * (half_px + RIM_SAFETY_PX) / MASK_STEP)))
    eroded = cv2.erode(small, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_side, kernel_side)),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    square_center = _nearest(eroded > 0, fx, fy)

    def within(center: tuple[float, float] | None) -> bool:
        return center is not None and np.hypot(center[0] - fx, center[1] - fy) <= limit

    rotation_ok = within(rotation_center)
    square_ok = within(square_center)
    if rotation_ok:
        center = rotation_center
    elif square_ok:
        center = square_center
    else:
        # Não cabe: aproxima do ponto de maior folga, respeitando o limite.
        target = np.unravel_index(int(np.argmax(dist)), dist.shape)
        vector = np.array([target[1] - fx, target[0] - fy], dtype=np.float64)
        norm = np.hypot(*vector)
        step = vector * min(1.0, limit / norm) if norm > 0 else vector
        center = (fx + step[0], fy + step[1])
    cx, cy = center[0] * MASK_STEP, center[1] * MASK_STEP
    if rotation_ok or square_ok:
        # Sem deslocamento desnecessário: a grade reduzida não deve mover a janela à toa.
        if np.hypot(cx - fovea_x, cy - fovea_y) <= MASK_STEP * 1.5:
            cx, cy = fovea_x, fovea_y
    return {"center_x": cx, "center_y": cy, "shift_px": float(np.hypot(cx - fovea_x, cy - fovea_y)),
            "rotation_ok": bool(rotation_ok), "square_ok": bool(square_ok), "margin_black": not square_ok}


def extract_patch(bgr: np.ndarray, mask: np.ndarray, center_x: float, center_y: float, half_native: float) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Recorte nativo com lado par, zerado fora do campo e preenchido com preto fora da imagem."""
    half_int = int(np.ceil(half_native))
    x0, y0 = int(round(center_x)) - half_int, int(round(center_y)) - half_int
    side = 2 * half_int
    patch = np.zeros((side, side, 3), np.uint8)
    patch_mask = np.zeros((side, side), np.uint8)
    h, w = bgr.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        src_mask = mask[sy0:sy1, sx0:sx1]
        patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = bgr[sy0:sy1, sx0:sx1] * src_mask[..., None]
        patch_mask[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = src_mask
    return patch, patch_mask, x0, y0, side


def resize_antialiased(image: np.ndarray, side: int) -> np.ndarray:
    """Área para reduzir, Lanczos para ampliar. Nunca vizinho mais próximo nem bilinear."""
    interpolation = cv2.INTER_AREA if side <= image.shape[0] else cv2.INTER_LANCZOS4
    return cv2.resize(image, (side, side), interpolation=interpolation)


def outside_fraction(patch_mask: np.ndarray, native_side: int, half_px: float) -> float:
    """Fração exata do ROI final fora do campo."""
    half = int(round(half_px))
    c = native_side // 2
    roi = patch_mask[c - half:c + half, c - half:c + half]
    return float(1.0 - roi.mean()) if roi.size else 1.0


def process_image(job: dict) -> list[dict]:
    """Gera todos os patches de uma imagem e devolve uma linha por ROI."""
    bgr = cv2.imread(job["path"], cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("imagem ilegível")
    fov, loc = locate(bgr, job["side"])
    radius_for_scale = POPULATION_RADIUS if fov.suspect else fov.radius
    dd_px = DD_PRIOR_R * radius_for_scale
    fovea_x = fov.cx + loc["fovea_dx"] * fov.radius
    fovea_y = fov.cy + loc["fovea_dy"] * fov.radius

    rows = []
    for roi, spec in job["rois"].items():
        if spec.get("full_field"):
            half_px = fov.radius
            window = {"center_x": fov.cx, "center_y": fov.cy, "shift_px": 0.0, "rotation_ok": True,
                      "square_ok": True, "margin_black": False}
            half_native = half_px * FULL_FIELD_PATCH_MARGIN
        else:
            half_px = spec["half_dd"] * dd_px
            window = fit_window(fov.mask, fovea_x, fovea_y, half_px)
            if not spec.get("rotation", False):
                window["rotation_ok"] = False
            half_native = half_px * ROI_PATCH_MARGIN
        patch, patch_mask, x0, y0, native_side = extract_patch(bgr, fov.mask, window["center_x"], window["center_y"], half_native)
        for resolution, res in job["resolutions"].items():
            target = patch_side(res["side"], bool(spec.get("full_field")))
            out_path = Path(job["out"]) / roi / resolution / job["file"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out_path), resize_antialiased(patch, target), [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError("falha ao gravar o cache")
        rows.append({
            "file": job["file"], "original_path": job["path"], "roi": roi, "half_dd": spec.get("half_dd"),
            "crop_x0": x0, "crop_y0": y0, "crop_side_native": native_side, "roi_half_native": half_px,
            "dd_px": dd_px, "fovea_x": fovea_x, "fovea_y": fovea_y,
            "outside_fov_frac": outside_fraction(patch_mask, native_side, half_px),
            **window, **loc,
        })
    return rows


def estimate_bytes(jobs: list[dict], n_total: int, sample: int = 8) -> float:
    """Gera algumas imagens numa pasta temporária do mesmo disco e extrapola o tamanho."""
    target = Path(jobs[0]["out"])
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as tmp:
        step = max(1, len(jobs) // sample)
        chosen = jobs[::step][:sample]
        for job in chosen:
            process_image({**job, "out": tmp})
        used = sum(p.stat().st_size for p in Path(tmp).rglob("*.png"))
    return used / len(chosen) * n_total


def flag_summary(table: pd.DataFrame) -> dict:
    """Taxas de sinalização por ROI e por classe, com alerta acima de 10%."""
    summary: dict = {"limit": FLAG_LIMIT, "by_roi": {}, "alerts": []}
    for roi, g in table.groupby("roi"):
        rates = {c: round(float(g[c].mean()), 4) for c in FLAG_COLUMNS}
        rates["rotation_ok"] = round(float(g["rotation_ok"].mean()), 4)
        rates["median_shift_px"] = round(float(g["shift_px"].median()), 1)
        rates["outside_fov_frac_mean"] = round(float(g["outside_fov_frac"].mean()), 4)
        rates["by_AMD"] = {str(k): {c: round(float(h[c].mean()), 4) for c in FLAG_COLUMNS}
                           for k, h in g.groupby("AMD")}
        summary["by_roi"][roi] = rates
        for column in ("margin_black", "disc_uncertain", "side_conflict", "fov_suspect"):
            if rates[column] > FLAG_LIMIT:
                summary["alerts"].append(f"{roi}: {column} = {rates[column]:.1%}")
    # Rotação só vale para a variante de ROI se a margem couber em quase todas as imagens.
    summary["rotation_allowed"] = {roi: bool(g["rotation_ok"].mean() >= 1 - FLAG_LIMIT) for roi, g in table.groupby("roi")}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("hyamd"))
    ap.add_argument("--out", type=Path, default=Path("cache"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="processa só as N primeiras imagens (teste)")
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--resolutions", nargs="*", default=None)
    ap.add_argument("--yes-disk", action="store_true", help="ignora o bloqueio de espaço em disco")
    args = ap.parse_args()

    registry = load_registry()
    rois = {k: v for k, v in registry["rois"].items() if args.rois is None or k in args.rois}
    resolutions = {k: v for k, v in registry["resolutions"].items() if args.resolutions is None or k in args.resolutions}
    meta = load_metadata(args.root)
    excluded = meta[meta["side_anomaly"]]
    kept = meta[~meta["side_anomaly"]]
    if args.limit:
        kept = kept.sample(min(args.limit, len(kept)), random_state=0)

    args.out.mkdir(parents=True, exist_ok=True)
    excluded.assign(reason="lado do nome (D/E) em conflito com a coluna side").to_csv(args.out / "excluded.csv", index=False)
    progress = args.out / "progress.jsonl"
    done: set[str] = set()
    if progress.exists():
        with progress.open(encoding="utf-8") as handle:
            done = {json.loads(line)["file"] for line in handle if line.strip()}

    jobs = [{"path": str(args.root / "Images" / r.file), "file": r.file, "side": r.side, "out": str(args.out),
             "rois": rois, "resolutions": resolutions} for r in kept.itertuples() if r.file not in done]
    print(f"imagens: {len(kept)}; já feitas: {len(kept) - len(jobs)}; excluídas: {len(excluded)}", flush=True)

    if jobs:
        estimate = estimate_bytes(jobs, len(jobs))
        free = shutil.disk_usage(args.out).free
        print(f"cache estimado: {estimate / 1e9:.1f} GB; livre: {free / 1e9:.1f} GB", flush=True)
        if estimate * 1.2 + FREE_SPACE_RESERVE > free and not args.yes_disk:
            raise SystemExit("espaço insuficiente; nada foi gerado. Use --rois/--resolutions ou --out em outro disco.")

        start = time.time()
        errors = []
        with ProcessPoolExecutor(args.workers) as pool, progress.open("a", encoding="utf-8") as log:
            futures = {pool.submit(process_image, job): job["file"] for job in jobs}
            for count, future in enumerate(as_completed(futures), 1):
                file = futures[future]
                try:
                    for row in future.result():
                        log.write(json.dumps(row, default=float) + "\n")
                    log.flush()
                except Exception as error:  # noqa: BLE001 - uma imagem ruim não deve parar a noite inteira
                    errors.append({"file": file, "error": repr(error)})
                if count % 25 == 0 or count == len(jobs):
                    elapsed = time.time() - start
                    print(f"{count}/{len(jobs)} em {elapsed / 60:.1f} min; restam ~{elapsed / count * (len(jobs) - count) / 60:.0f} min; erros: {len(errors)}", flush=True)
        if errors:
            pd.DataFrame(errors).to_csv(args.out / "errors.csv", index=False)

    rows = pd.read_json(progress, lines=True, dtype={"file": str})
    table = meta[["file", "patient_id", "eye_id", "side", "name_side", "sex", "age", "AMD"]].merge(rows, on="file", validate="one_to_many")
    table.to_csv(args.out / "metadata.csv", index=False)
    summary = flag_summary(table)
    summary["n_images"] = int(table["file"].nunique())
    summary["bytes_on_disk"] = int(sum(p.stat().st_size for p in args.out.rglob("*.png")))
    (args.out / "flag_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_images", "bytes_on_disk", "rotation_allowed", "alerts")}, indent=1))


if __name__ == "__main__":
    main()
