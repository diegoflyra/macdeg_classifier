"""Passo 3: auditoria de perda de informação, sem rótulos. Proxy comparativo, não detector de drusa.

Uso: python -m macdeg.audit [--cache cache] [--out outputs/audit] [--n 200] [--workers 6]

Referência: ROI na resolução nativa do original. Detector: DoG no canal verde, em diâmetros de
0.03 a 0.08 DD. Os limiares são fixados uma vez na referência e nunca ajustados por variante.
As imagens avaliadas passam pelo Augmenter real, sobre o cache real.
"""
from __future__ import annotations

import argparse
import json
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial import cKDTree

from macdeg.augment import Augmenter, geometry_matrix, get_variant, list_variant_ids, load_patch, load_registry

DIAMETERS_DD = (0.03, 0.045, 0.06, 0.08)
THRESHOLD_PERCENTILE = 98.0
MIN_TOLERANCE_PX = 1.5  # ~1 px na escala final, com folga para o arredondamento do pico
METRICS = ["retention", "retention_low_contrast", "contrast_ratio", "energy_ratio", "spurious_rate",
           "roi_area_lost", "frac_in_frame", "px_per_dd"]

# Direção esperada de cada controle negativo, definida antes de ver os resultados.
EXPECTED_LOSS = {
    "Q1": [("retention", "down"), ("energy_ratio", "down")],
    "Q2": [("spurious_rate", "up")],
    "Q3": [("retention", "down"), ("energy_ratio", "down")],
    "Q4": [("retention", "down"), ("contrast_ratio", "down")],
}


def _sigma(diameter_dd: float, px_per_dd: float) -> float:
    return max(0.3, diameter_dd * px_per_dd / (2 * np.sqrt(2)))


def dog_responses(green: np.ndarray, px_per_dd: float) -> list[np.ndarray]:
    """Resposta DoG, positiva em manchas claras, para cada diâmetro."""
    green = green.astype(np.float32)
    out = []
    for diameter in DIAMETERS_DD:
        sigma = _sigma(diameter, px_per_dd)
        out.append(cv2.GaussianBlur(green, (0, 0), sigma) - cv2.GaussianBlur(green, (0, 0), 1.6 * sigma))
    return out


def valid_interior(valid: np.ndarray, px_per_dd: float) -> np.ndarray:
    """Tira uma faixa junto à borda do campo, onde o DoG responde ao degrau para o preto."""
    margin = int(np.ceil(3 * _sigma(DIAMETERS_DD[-1], px_per_dd))) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * margin + 1, 2 * margin + 1))
    return cv2.erode(valid.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0


def fit_thresholds(responses_per_image: list[list[np.ndarray]]) -> list[float]:
    """Um limiar por diâmetro: percentil das respostas positivas nas referências."""
    thresholds = []
    for i in range(len(DIAMETERS_DD)):
        pool = np.concatenate([r[i] for r in responses_per_image])
        thresholds.append(float(np.percentile(pool[pool > 0], THRESHOLD_PERCENTILE)))
    return thresholds


def detect(responses: list[np.ndarray], interior: np.ndarray, thresholds: list[float], px_per_dd: float) -> np.ndarray:
    """Máximos locais acima do limiar. Linhas: x, y, resposta, índice do diâmetro."""
    found = []
    for i, response in enumerate(responses):
        size = max(3, 2 * int(round(_sigma(DIAMETERS_DD[i], px_per_dd))) + 1)
        peaks = (response == ndimage.maximum_filter(response, size=size)) & (response > thresholds[i]) & interior
        ys, xs = np.nonzero(peaks)
        found.append(np.c_[xs, ys, response[ys, xs], np.full(len(xs), i)])
    return np.vstack(found) if found else np.empty((0, 4))


def transform_points(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return (matrix[:2, :2] @ xy.T).T + matrix[:2, 2]


def area_lost(native_to_output: np.ndarray, corners: np.ndarray, side: int) -> float:
    """Fração exata do ROI que sai do quadro de saída."""
    polygon = transform_points(native_to_output, corners).astype(np.float32)
    frame = np.array([[-0.5, -0.5], [side - 0.5, -0.5], [side - 0.5, side - 0.5], [-0.5, side - 0.5]], np.float32)
    if cv2.contourArea(polygon, oriented=True) < 0:
        polygon = polygon[::-1]
    area, _ = cv2.intersectConvexConvex(polygon, frame)
    return float(max(0.0, 1.0 - area / cv2.contourArea(polygon)))


@dataclass
class Reference:
    """Referência já preparada para um ROI."""

    green: np.ndarray
    valid: np.ndarray
    px_per_dd: float
    responses: list[np.ndarray]
    points: np.ndarray  # coordenadas nativas: x, y, resposta, índice do diâmetro
    corners: np.ndarray
    origin_xy: np.ndarray
    interior: np.ndarray  # onde a referência pode detectar: base de comparação para estruturas novas


def make_reference(green: np.ndarray, valid: np.ndarray, px_per_dd: float, thresholds: list[float],
                   origin_xy: tuple[float, float] = (0.0, 0.0)) -> Reference:
    """`green` e `valid` são o ROI nativo. `origin_xy` é a posição do canto do ROI na imagem nativa."""
    responses = dog_responses(green, px_per_dd)
    interior = valid_interior(valid, px_per_dd)
    points = detect(responses, interior, thresholds, px_per_dd)
    origin = np.asarray(origin_xy, dtype=np.float64)
    points = points.copy()
    points[:, :2] += origin
    h, w = green.shape
    corners = origin + np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
    return Reference(green, valid, px_per_dd, responses, points, corners, origin, interior)


def evaluate(reference: Reference, green: np.ndarray, valid: np.ndarray, px_per_dd: float,
             native_to_output: np.ndarray, thresholds: list[float]) -> dict:
    """Compara uma imagem transformada com a referência. Coordenadas ligadas por `native_to_output`."""
    side = green.shape[0]
    responses = dog_responses(green, px_per_dd)
    interior = valid_interior(valid, px_per_dd)
    found = detect(responses, interior, thresholds, px_per_dd)
    tolerance = max(MIN_TOLERANCE_PX, 0.3 * DIAMETERS_DD[0] * px_per_dd)

    mapped = transform_points(native_to_output, reference.points[:, :2])
    col, row = np.round(mapped[:, 0]).astype(int), np.round(mapped[:, 1]).astype(int)
    in_frame = (col >= 0) & (col < side) & (row >= 0) & (row < side)
    in_frame[in_frame] = interior[row[in_frame], col[in_frame]]
    n_reference = len(mapped)
    usable = reference.points[in_frame]
    usable_mapped = mapped[in_frame]

    result = {"n_reference": n_reference, "n_usable": len(usable), "n_found": len(found),
              "frac_in_frame": len(usable) / n_reference if n_reference else np.nan, "px_per_dd": px_per_dd,
              "roi_area_lost": area_lost(native_to_output, reference.corners, side)}
    if len(usable) == 0:
        result.update(retention=np.nan, retention_low_contrast=np.nan, contrast_ratio=np.nan, spurious_rate=np.nan)
    else:
        retained = np.zeros(len(usable), bool)
        if len(found):
            distance, _ = cKDTree(found[:, :2]).query(usable_mapped, distance_upper_bound=tolerance)
            retained = np.isfinite(distance)
        result["retention"] = float(retained.mean())
        low = usable[:, 2] <= np.median(usable[:, 2])
        result["retention_low_contrast"] = float(retained[low].mean()) if low.any() else np.nan

        # Contraste na mesma escala, no melhor ponto dentro da tolerância.
        size = 2 * int(np.ceil(tolerance)) + 1
        ratios = []
        for i in range(len(DIAMETERS_DD)):
            selected = retained & (usable[:, 3] == i)
            if selected.any():
                peak = ndimage.maximum_filter(responses[i], size=size)
                c = np.clip(np.round(usable_mapped[selected, 0]).astype(int), 0, side - 1)
                r = np.clip(np.round(usable_mapped[selected, 1]).astype(int), 0, side - 1)
                ratios.append(peak[r, c] / usable[selected, 2])
        result["contrast_ratio"] = float(np.concatenate(ratios).mean()) if ratios else np.nan

        # Detecções novas: só onde a referência também poderia ter detectado. Com rotação, o quadro
        # mostra área fora do ROI, sem referência, que não pode contar como estrutura criada.
        comparable = np.zeros(len(found), bool)
        if len(found):
            back = transform_points(np.linalg.inv(native_to_output), found[:, :2]) - reference.origin_xy
            bx, by = np.round(back[:, 0]).astype(int), np.round(back[:, 1]).astype(int)
            ref_h, ref_w = reference.interior.shape
            inside = (bx >= 0) & (bx < ref_w) & (by >= 0) & (by < ref_h)
            comparable[inside] = reference.interior[by[inside], bx[inside]]
        if comparable.any():
            distance, _ = cKDTree(mapped).query(found[comparable, :2], distance_upper_bound=tolerance)
            result["spurious_rate"] = float(np.mean(~np.isfinite(distance)))
        else:
            result["spurious_rate"] = 0.0

    # Energia na faixa dessas escalas, nos mesmos pixels físicos.
    inverse = np.linalg.inv(native_to_output)
    ys, xs = np.nonzero(interior)
    pick = np.arange(0, len(xs), 7)
    native = transform_points(inverse, np.c_[xs[pick], ys[pick]].astype(np.float64)) - reference.origin_xy
    nx, ny = np.round(native[:, 0]).astype(int), np.round(native[:, 1]).astype(int)
    h, w = reference.green.shape
    inside = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
    keep = np.zeros(len(nx), bool)
    keep[inside] = reference.valid[ny[inside], nx[inside]]
    ratios = []
    if keep.any():
        for variant_response, reference_response in zip(responses, reference.responses):
            denominator = np.mean(reference_response[ny[keep], nx[keep]] ** 2)
            if denominator > 0:
                ratios.append(np.mean(variant_response[ys[pick][keep], xs[pick][keep]] ** 2) / denominator)
    result["energy_ratio"] = float(np.mean(ratios)) if ratios else np.nan
    return result


def native_reference(bgr: np.ndarray, mask: np.ndarray, row: pd.Series, thresholds: list[float]) -> Reference:
    """ROI nativo centrado na janela gravada em metadata.csv."""
    half = int(round(row["roi_half_native"]))
    center_x = row["crop_x0"] + row["crop_side_native"] // 2
    center_y = row["crop_y0"] + row["crop_side_native"] // 2
    x0, y0 = center_x - half, center_y - half
    h, w = bgr.shape[:2]
    green = np.zeros((2 * half, 2 * half), np.uint8)
    valid = np.zeros((2 * half, 2 * half), np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * half, w), min(y0 + 2 * half, h)
    if sx1 > sx0 and sy1 > sy0:
        green[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = bgr[sy0:sy1, sx0:sx1, 1] * mask[sy0:sy1, sx0:sx1]
        valid[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = mask[sy0:sy1, sx0:sx1]
    return make_reference(green, valid, float(row["dd_px"]), thresholds, (x0, y0))


def native_to_patch(row: pd.Series, patch: int) -> np.ndarray:
    scale = patch / row["crop_side_native"]
    matrix = np.eye(3)
    matrix[0, 0] = matrix[1, 1] = scale
    matrix[0, 2] = (0.5 - row["crop_x0"]) * scale - 0.5
    matrix[1, 2] = (0.5 - row["crop_y0"]) * scale - 0.5
    return matrix


def _audit_image(job: dict) -> list[dict]:
    from macdeg.localization import detect_fov

    registry = load_registry(flag_summary=Path(job["cache"]) / "flag_summary.json")
    bgr = cv2.imread(job["path"], cv2.IMREAD_COLOR)
    mask = detect_fov(bgr).mask
    rows = []
    for roi, row in job["rows"].items():
        row = pd.Series(row)
        reference = native_reference(bgr, mask, row, job["thresholds"])
        for variant_id in job["variants"]:
            if not variant_id.startswith(f"{roi}_"):
                continue
            variant = get_variant(variant_id, registry)
            rng = np.random.default_rng([job["index"], zlib.crc32(variant_id.encode())])
            augmenter = Augmenter(variant, train=True, registry=registry)
            patch = load_patch(job["cache"], variant, job["file"])
            params = augmenter.sample_params(rng, bool(row["rotation_ok"]))
            output = augmenter.apply_one(patch, params, rng)
            green = output[..., 1] * 255.0
            # A máscara do campo vem do patch, com a mesma geometria e sem interpolação cúbica.
            patch_valid = (patch.max(axis=2) > 0).astype(np.uint8)
            if params["geometry"]["kind"] == "d4":
                moved = np.fliplr(patch_valid) if params["geometry"]["flip"] else patch_valid
                moved = np.rot90(moved, params["geometry"]["k"])
                start = (moved.shape[0] - variant.side) // 2
                valid = moved[start:start + variant.side, start:start + variant.side]
            else:
                matrix = geometry_matrix(params["geometry"], variant.patch, variant.side)
                valid = cv2.warpAffine(patch_valid, matrix[:2], (variant.side, variant.side), flags=cv2.INTER_NEAREST)
            to_output = geometry_matrix(params["geometry"], variant.patch, variant.side) @ native_to_patch(row, variant.patch)
            scale = abs(np.linalg.det(to_output[:2, :2])) ** 0.5
            metrics = evaluate(reference, green, valid, scale * float(row["dd_px"]), to_output, job["thresholds"])
            rows.append({"sample": job["index"], "variant": variant_id, "roi": roi, "resolution": variant.resolution,
                         "recipe": variant.recipe, "rotation_fallback": variant.geometry["type"] == "rotate" and params["geometry"]["kind"] != "rotate",
                         **metrics})
    return rows


def choose_sample(metadata: pd.DataFrame, n: int) -> list[str]:
    """Uma imagem por paciente quando possível."""
    files = metadata.drop_duplicates("file")
    first = files.sample(frac=1.0, random_state=0).drop_duplicates("patient_id")
    chosen = list(first["file"].head(n))
    if len(chosen) < n:
        rest = files[~files["file"].isin(chosen)].sample(frac=1.0, random_state=1)
        chosen += list(rest["file"].head(n - len(chosen)))
    return chosen


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["variant", "roi", "resolution", "recipe"])
    parts = []
    for metric in METRICS:
        block = grouped[metric].agg(mean="mean", p5=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95))
        block.columns = [f"{metric}_{c}" for c in block.columns]
        parts.append(block)
    table = pd.concat(parts, axis=1).reset_index()
    table.insert(4, "n_images", grouped.size().to_numpy())
    return table


def check_controls(rows: pd.DataFrame) -> pd.DataFrame:
    """Compara cada controle com A0, por imagem. Passa se o IC de 95% confirma a direção esperada."""
    base = rows[rows["recipe"] == "A0"].set_index(["sample", "roi", "resolution"])
    records = []
    for recipe, expected in EXPECTED_LOSS.items():
        control = rows[rows["recipe"] == recipe].set_index(["sample", "roi", "resolution"])
        for (roi, resolution), group in control.groupby(level=["roi", "resolution"]):
            for metric, direction in expected:
                delta = (group[metric] - base.loc[group.index, metric]).dropna()
                mean, ci = float("nan"), float("nan")
                if len(delta) >= 2:
                    mean = float(delta.mean())
                    ci = float(stats.t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta)))
                detected = bool(len(delta) >= 2 and (mean + ci < 0 if direction == "down" else mean - ci > 0))
                records.append({"recipe": recipe, "roi": roi, "resolution": resolution, "metric": metric,
                                "expected": direction, "mean_delta": mean, "ci95": ci, "n": len(delta), "detected": detected})
    return pd.DataFrame(records)


def _dot_plot(ax: plt.Axes, table: pd.DataFrame, metric: str, labels: str, reference: float | None = None) -> None:
    """Pontos com intervalo p5-p95. Uma só série, identificada pelo título."""
    ordered = table.iloc[::-1]
    y = np.arange(len(ordered))
    mean = ordered[f"{metric}_mean"].to_numpy()
    ax.hlines(y, ordered[f"{metric}_p5"], ordered[f"{metric}_p95"], color="#9bb8e0", linewidth=2)
    ax.scatter(mean, y, s=36, color="#1f5fb4", zorder=3)
    ax.set_yticks(y, ordered[labels])
    if reference is not None:
        ax.axvline(reference, color="#8a8a8a", linewidth=1, linestyle=(0, (3, 3)))
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)


def make_plots(table: pd.DataFrame, out: Path, roi: str = "R2.0", resolution: str = "S512") -> None:
    """Gráficos simples: por operação, num ROI e resolução fixos, e por ROI e resolução, sem aumento."""
    by_recipe = table[(table["roi"] == roi) & (table["resolution"] == resolution)]
    panels = [("retention", "Retenção de estruturas", 1.0), ("contrast_ratio", "Razão de contraste", 1.0),
              ("energy_ratio", "Razão de energia", 1.0), ("spurious_rate", "Taxa de estruturas novas", None)]
    if len(by_recipe):
        fig, axes = plt.subplots(1, 4, figsize=(15, 0.33 * len(by_recipe) + 1.6), sharey=True)
        for ax, (metric, title, reference) in zip(axes, panels):
            _dot_plot(ax, by_recipe, metric, "recipe", reference)
            ax.set_title(title, loc="left", fontsize=10)
        fig.suptitle(f"Por operação em {roi}, {resolution}: média e intervalo p5-p95 entre imagens", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"por_operacao_{roi}_{resolution}.png", dpi=130)
        plt.close(fig)

    base = table[table["recipe"] == "A0"].assign(label=lambda d: d["roi"] + " " + d["resolution"])
    if len(base):
        fig, axes = plt.subplots(1, 3, figsize=(12, 0.33 * len(base) + 1.6), sharey=True)
        for ax, (metric, title, reference) in zip(axes, panels[:3]):
            _dot_plot(ax, base, metric, "label", reference)
            ax.set_title(title, loc="left", fontsize=10)
        fig.suptitle("Efeito de ROI e resolução, sem augmentation (A0)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "por_roi_resolucao.png", dpi=130)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/audit"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--resolutions", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.cache / "metadata.csv", dtype={"patient_id": str})
    registry = load_registry(flag_summary=args.cache / "flag_summary.json")
    variants = [v for v in list_variant_ids(registry)
                if (args.rois is None or v.split("_")[0] in args.rois)
                and (args.resolutions is None or v.split("_")[1] in args.resolutions)]
    rois = sorted({v.split("_")[0] for v in variants})
    sample = choose_sample(metadata, args.n)

    from macdeg.localization import detect_fov

    # Limiares: uma única vez, na referência, e reaproveitados nas próximas execuções.
    thresholds_path = args.out / "thresholds.json"
    if thresholds_path.exists():
        thresholds = json.loads(thresholds_path.read_text())["thresholds"]
    else:
        calibration_roi = "R2.0" if "R2.0" in rois else rois[0]
        calibration = metadata[(metadata["roi"] == calibration_roi) & metadata["file"].isin(sample)]
        pools = []
        for _, row in calibration.iterrows():
            bgr = cv2.imread(row["original_path"], cv2.IMREAD_COLOR)
            reference = native_reference(bgr, detect_fov(bgr).mask, row, [np.inf] * len(DIAMETERS_DD))
            interior = valid_interior(reference.valid, reference.px_per_dd)
            pools.append([r[interior][::11] for r in reference.responses])
        thresholds = fit_thresholds(pools)
        thresholds_path.write_text(json.dumps({"thresholds": thresholds, "diameters_dd": DIAMETERS_DD,
                                               "percentile": THRESHOLD_PERCENTILE, "calibration_roi": calibration_roi}, indent=1))

    jobs = []
    for index, file in enumerate(sample):
        subset = metadata[(metadata["file"] == file) & metadata["roi"].isin(rois)]
        jobs.append({"index": index, "file": file, "path": subset.iloc[0]["original_path"], "cache": str(args.cache),
                     "rows": {r["roi"]: r.to_dict() for _, r in subset.iterrows()}, "variants": variants, "thresholds": thresholds})

    rows: list[dict] = []
    errors = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_audit_image, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            try:
                rows.extend(future.result())
            except Exception as error:  # noqa: BLE001
                errors += 1
                print(f"erro em uma imagem: {type(error).__name__}: {error}", flush=True)
            if count % 10 == 0 or count == len(jobs):
                print(f"auditoria: {count}/{len(jobs)} imagens; erros: {errors}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "audit_per_image.csv", index=False)
    table = summarize(frame)
    table.to_csv(args.out / "audit_table.csv", index=False)
    controls = check_controls(frame)
    controls.to_csv(args.out / "negative_controls.csv", index=False)
    make_plots(table, args.out)

    by_recipe = controls.groupby("recipe")["detected"].mean().round(3).to_dict()
    missed = controls[~controls["detected"]]
    verdict = {
        "n_images": int(frame["sample"].nunique()), "errors": errors, "n_variants": int(frame["variant"].nunique()),
        "control_detection_rate_by_recipe": by_recipe,
        "audit_sensitive": bool(controls["detected"].all()),
        "missed": missed[["recipe", "roi", "resolution", "metric"]].to_dict("records"),
    }
    (args.out / "audit_verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    print(json.dumps({k: verdict[k] for k in ("n_images", "errors", "n_variants", "control_detection_rate_by_recipe", "audit_sensitive")}, indent=1))
    if not verdict["audit_sensitive"]:
        print(f"ALERTA: {len(missed)} combinações de controle negativo sem perda detectada. Veja audit_verdict.json.")


if __name__ == "__main__":
    main()
