"""Campo de visão, disco óptico, fóvea e diâmetro do disco (DD), por imagem.

Coordenadas internas em R (raio do campo), com origem no centro do campo.

O detector de disco evita confundir drusas com o disco:
- procura só no lado nasal, numa faixa horizontal;
- a abertura morfológica remove manchas claras menores que o disco, como as drusas;
- exige densidade de vasos, que é alta no disco e baixa num campo de drusas.

A fóvea vem de disco + vetor médio. A mancha escura é só diagnóstico, pois falha em olhos doentes.
Quando o disco é incerto, a fóvea cai na posição típica do formato e a imagem é sinalizada.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_R = 128
WORK_SIZE = 2 * WORK_R + 32
WORK_C = (WORK_SIZE - 1) / 2

# Medidos nas médias alinhadas pelo disco (Passo 0), com o disco espelhado para dx > 0.
DISC_TO_FOVEA = (-0.61, 0.09)
FOVEA_PRIOR_BY_WIDTH = {1960: (-0.21, 0.06), 2576: (0.01, 0.05)}
DISC_CDX_PRIOR_BY_WIDTH = {1960: 0.42, 2576: 0.62}
DD_PRIOR_R = 0.247  # supõe 2.5 DD entre disco e fóvea: 0.617 R / 2.5

R_EXPECTED_RANGE = (920.0, 1010.0)
DISC_SCORE_MIN = 2.0
DISC_MARGIN_MIN = 0.4
FOVEA_DISAGREE_R = 0.25


@dataclass
class Fov:
    mask: np.ndarray  # uint8, 1 dentro do campo, na resolução original
    cx: float
    cy: float
    radius: float
    rms: float
    suspect: bool


def detect_fov(bgr: np.ndarray) -> Fov:
    """Máscara do campo e círculo ajustado de forma robusta só com a borda real."""
    h, w = bgr.shape[:2]
    binary = (cv2.GaussianBlur(bgr.max(axis=2), (0, 0), 3) > 12).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n < 2:
        raise ValueError("campo de visão não encontrado")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((cc == largest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)[:, 0, :]
    mask = cv2.drawContours(np.zeros((h, w), np.uint8), [contour], -1, 1, cv2.FILLED)

    margin = 3
    arc = contour[(contour[:, 0] > margin) & (contour[:, 0] < w - 1 - margin)
                  & (contour[:, 1] > margin) & (contour[:, 1] < h - 1 - margin)]
    pts = (arc if len(arc) > 100 else contour).astype(np.float64)
    cx = cy = radius = 0.0
    keep = pts
    for _ in range(5):  # mínimos quadrados aparados: descarta recortes e clarões da borda
        design = np.c_[2 * keep[:, 0], 2 * keep[:, 1], np.ones(len(keep))]
        sol = np.linalg.lstsq(design, (keep ** 2).sum(axis=1), rcond=None)[0]
        cx, cy = float(sol[0]), float(sol[1])
        radius = float(np.sqrt(sol[2] + cx * cx + cy * cy))
        residual = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - radius)
        limit = max(4.0, 2.5 * float(np.median(residual)))
        selected = pts[residual <= limit]
        if len(selected) < 50:
            break
        keep = selected
    rms = float(np.sqrt(((np.hypot(keep[:, 0] - cx, keep[:, 1] - cy) - radius) ** 2).mean()))
    suspect = not (R_EXPECTED_RANGE[0] <= radius <= R_EXPECTED_RANGE[1]) or rms > 20
    return Fov(mask, cx, cy, radius, rms, suspect)


def work_image(bgr: np.ndarray, fov: Fov) -> tuple[np.ndarray, np.ndarray]:
    """Imagem reduzida com antialiasing, centrada no campo. Fora do campo fica preto."""
    half = fov.radius * WORK_SIZE / (2 * WORK_R)
    x0, y0 = int(round(fov.cx - half)), int(round(fov.cy - half))
    side = int(round(2 * half))
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas_mask = np.zeros((side, side), np.uint8)
    h, w = bgr.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    src = bgr[sy0:sy1, sx0:sx1]
    src_mask = fov.mask[sy0:sy1, sx0:sx1]
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = src * src_mask[..., None]
    canvas_mask[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = src_mask
    size = (WORK_SIZE, WORK_SIZE)
    small = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
    small_mask = (cv2.resize(canvas_mask, size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    return small, small_mask


def _grid() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:WORK_SIZE, :WORK_SIZE].astype(np.float32)
    return (xx - WORK_C) / WORK_R, (yy - WORK_C) / WORK_R


def _masked_blur(channel: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    num = cv2.GaussianBlur(channel * weight, (0, 0), sigma)
    return num / np.maximum(cv2.GaussianBlur(weight, (0, 0), sigma), 1e-3)


def _ellipse(diameter: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def _disc_maps(small: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mapa de mancha clara sólida do tamanho do disco e mapa de densidade de vasos."""
    weight = cv2.erode(mask, _ellipse(9)).astype(np.float32)
    green = small[..., 1].astype(np.float32)
    red = small[..., 2].astype(np.float32)

    # Fechar remove os vasos; abrir remove o que é menor que ~0.45 DD, como as drusas.
    combined = green + 0.5 * red
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _ellipse(7))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, _ellipse(15))
    blob = _masked_blur(opened, weight, 6) - _masked_blur(opened, weight, 26)

    vessels = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, _ellipse(9))
    density = _masked_blur(vessels, weight, 12)

    inside = weight > 0
    for arr in (blob, density):
        arr -= arr[inside].mean()
        arr /= arr[inside].std() + 1e-6
    blob[~inside] = density[~inside] = -10.0
    return blob, density


def _candidates(score: np.ndarray, allowed: np.ndarray, k: int = 6, nms: int = 20) -> list[tuple[int, int]]:
    working = np.where(allowed, score, -np.inf)
    found = []
    for _ in range(k):
        y, x = np.unravel_index(int(np.argmax(working)), working.shape)
        if not np.isfinite(working[y, x]):
            break
        found.append((int(x), int(y)))
        cv2.circle(working, (int(x), int(y)), nms, -np.inf, -1)
    return found


def _best_disc(blob: np.ndarray, density: np.ndarray, mask: np.ndarray, sign: float,
               width: int) -> dict:
    """Melhor candidato no lado nasal indicado por `sign` (+1: disco em dx > 0)."""
    x, y = _grid()
    prior_x = DISC_CDX_PRIOR_BY_WIDTH.get(width, 0.55)
    cdx = sign * x
    allowed = (cdx > 0.15) & (cdx < 0.92) & (np.abs(y) < 0.40) & (mask > 0)
    prior = -0.5 * ((cdx - prior_x) / 0.18) ** 2 - 0.5 * ((y + 0.04) / 0.14) ** 2
    combined = blob + 0.6 * density + 0.5 * prior
    candidates = _candidates(combined, allowed)
    if not candidates:
        return {"score": -np.inf, "margin": 0.0, "dx": np.nan, "dy": np.nan, "blob": np.nan, "density": np.nan}
    values = [combined[v, u] for u, v in candidates]
    u, v = candidates[0]
    return {"score": float(values[0]), "margin": float(values[0] - values[1]) if len(values) > 1 else 9.0,
            "dx": float(x[v, u]), "dy": float(y[v, u]),
            "blob": float(blob[v, u]), "density": float(density[v, u])}


def _dark_spot(small: np.ndarray, mask: np.ndarray, disc: tuple[float, float], sign: float) -> tuple[float, float]:
    """Mancha escura no setor temporal ao disco. Só diagnóstico."""
    weight = cv2.erode(mask, _ellipse(9)).astype(np.float32)
    closed = cv2.morphologyEx(small[..., 1].astype(np.float32), cv2.MORPH_CLOSE, _ellipse(9))
    response = _masked_blur(closed, weight, 24) - _masked_blur(closed, weight, 6)
    x, y = _grid()
    vx, vy = x - disc[0], y - disc[1]
    dist = np.hypot(vx, vy)
    sector = (dist > 0.45) & (dist < 0.95) & (np.degrees(np.abs(np.arctan2(vy, -sign * vx))) < 30) & (weight > 0)
    if not sector.any():
        return np.nan, np.nan
    v, u = np.unravel_index(int(np.argmax(np.where(sector, response, -np.inf))), response.shape)
    return float(x[v, u]), float(y[v, u])


def measure_dd(bgr: np.ndarray, fov: Fov, disc_x: float, disc_y: float) -> tuple[float, float]:
    """DD em px a partir das bordas radiais do disco no canal vermelho. Devolve (dd_px, dispersão relativa)."""
    scale = 256 / fov.radius
    half = int(round(0.32 * fov.radius))
    x0, y0 = int(round(disc_x)) - half, int(round(disc_y)) - half
    h, w = bgr.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
        return np.nan, np.nan
    crop = bgr[y0:y0 + 2 * half, x0:x0 + 2 * half, 2]
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    closed = cv2.GaussianBlur(cv2.morphologyEx(small, cv2.MORPH_CLOSE, _ellipse(11)), (0, 0), 1.5)
    c = (closed.shape[0] - 1) / 2
    px_per_r = fov.radius * scale
    radii = np.arange(0.06 * px_per_r, 0.21 * px_per_r)
    edges = []
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        map_x = (c + radii * np.cos(angle)).astype(np.float32)[None]
        map_y = (c + radii * np.sin(angle)).astype(np.float32)[None]
        profile = cv2.remap(closed, map_x, map_y, cv2.INTER_LINEAR)[0]
        edges.append(radii[int(np.argmin(np.gradient(profile)))])
    edges = np.array(edges)
    median = float(np.median(edges))
    spread = float((np.percentile(edges, 75) - np.percentile(edges, 25)) / max(median, 1e-6))
    return 2 * median / scale, spread


def locate(bgr: np.ndarray, side: str, width: int | None = None) -> tuple[Fov, dict]:
    """Localiza tudo numa imagem BGR completa. `side` é o lado do labels.csv (L ou R)."""
    fov = detect_fov(bgr)
    width = width if width is not None else bgr.shape[1]
    small, mask = work_image(bgr, fov)
    return fov, locate_from_work(small, mask, side, width, fov, bgr)


def locate_from_work(small: np.ndarray, mask: np.ndarray, side: str, width: int, fov: Fov,
                     bgr: np.ndarray | None = None) -> dict:
    blob, density = _disc_maps(small, mask)
    label_sign = 1.0 if side == "R" else -1.0
    own = _best_disc(blob, density, mask, label_sign, width)
    other = _best_disc(blob, density, mask, -label_sign, width)

    side_conflict = bool(np.isfinite(other["score"]) and other["score"] > own["score"] + 1.0)
    chosen, sign = (other, -label_sign) if side_conflict else (own, label_sign)
    disc_uncertain = bool(not np.isfinite(chosen["score"]) or chosen["blob"] < DISC_SCORE_MIN
                          or chosen["margin"] < DISC_MARGIN_MIN)

    prior = FOVEA_PRIOR_BY_WIDTH.get(width, (0.0, 0.05))
    if disc_uncertain:
        method = "width_prior"
        fovea_dx, fovea_dy = sign * prior[0], prior[1]
    else:
        method = "disc_vector"
        fovea_dx = chosen["dx"] + sign * DISC_TO_FOVEA[0]
        fovea_dy = chosen["dy"] + DISC_TO_FOVEA[1]

    dark_dx, dark_dy = (np.nan, np.nan) if disc_uncertain else _dark_spot(
        small, mask, (chosen["dx"], chosen["dy"]), sign)
    disagree = bool(np.isfinite(dark_dx) and np.hypot(dark_dx - fovea_dx, dark_dy - fovea_dy) > FOVEA_DISAGREE_R)

    dd_measured, dd_spread = np.nan, np.nan
    if bgr is not None and not disc_uncertain:
        dd_measured, dd_spread = measure_dd(bgr, fov, fov.cx + chosen["dx"] * fov.radius,
                                            fov.cy + chosen["dy"] * fov.radius)
    dd_valid = bool(np.isfinite(dd_measured) and dd_spread < 0.25
                    and 0.17 * fov.radius <= dd_measured <= 0.34 * fov.radius)
    return {
        "fov_cx": fov.cx, "fov_cy": fov.cy, "fov_radius": fov.radius, "fov_rms": fov.rms, "fov_suspect": fov.suspect,
        "disc_dx": chosen["dx"], "disc_dy": chosen["dy"], "disc_score": chosen["score"], "disc_margin": chosen["margin"],
        "disc_blob": chosen["blob"], "disc_density": chosen["density"], "disc_uncertain": disc_uncertain,
        "side_from_image": "R" if sign > 0 else "L", "side_conflict": side_conflict,
        "fovea_dx": float(fovea_dx), "fovea_dy": float(fovea_dy), "fovea_method": method,
        "dark_dx": dark_dx, "dark_dy": dark_dy, "fovea_disagree": disagree,
        "dd_measured_px": dd_measured, "dd_spread": dd_spread, "dd_valid": dd_valid,
    }
