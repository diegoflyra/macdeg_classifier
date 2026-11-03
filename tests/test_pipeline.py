"""Testes com imagens sintéticas (não precisam da base), mais checagens do cache, se ele existir."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from macdeg.audit import DIAMETERS_DD, evaluate, fit_thresholds, make_reference
from macdeg.augment import (Augmenter, apply_geometry, center_crop, geometry_matrix, get_variant, list_variant_ids,
                            load_patch, load_registry, patch_side, transform_identity_params)
from macdeg.preprocess import fit_window, resize_antialiased

CACHE = Path("cache")


def textured_patch(variant, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(20, 235, (variant.patch // 8 + 1, variant.patch // 8 + 1, 3), dtype=np.uint8)
    patch = cv2.resize(noise, (variant.patch, variant.patch), interpolation=cv2.INTER_CUBIC)
    return np.clip(patch, 1, 255)


def test_a_identity_returns_center_crop():
    variant = get_variant("R2.0_S512_A0")
    patch = textured_patch(variant)
    out = Augmenter(variant, train=True)(patch, np.random.default_rng(0))
    assert np.array_equal(out, center_crop(patch, variant.side).astype(np.float32) / 255.0)
    # Rotação de parâmetros nulos: também exato.
    assert np.array_equal(apply_geometry(patch, transform_identity_params(), variant.side), center_crop(patch, variant.side))


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_b_d4_preserves_content_and_matches_matrix(flip, k):
    variant = get_variant("R2.0_S512_G1")
    patch = textured_patch(variant, 1)
    params = {"kind": "d4", "flip": flip, "k": k}
    out = apply_geometry(patch, params, variant.side)
    # Sem perda: o conteúdo do ROI é o mesmo, numa nova posição.
    crop = center_crop(patch, variant.side)
    assert np.array_equal(np.sort(out.reshape(-1, 3), axis=0), np.sort(crop.reshape(-1, 3), axis=0))
    # A matriz usada na auditoria aponta para o mesmo pixel.
    matrix = geometry_matrix(params, variant.patch, variant.side)
    start = (variant.patch - variant.side) // 2
    for x, y in [(start + 3, start + 7), (start + 100, start + 20), (start + 400, start + 300)]:
        u, v = (matrix @ np.array([x, y, 1.0]))[:2]
        assert np.array_equal(out[int(round(v)), int(round(u))], patch[y, x])


def test_c_eval_has_no_random_transform():
    variant = get_variant("R2.0_S512_A3-G2")
    patch = textured_patch(variant, 2)
    augmenter = Augmenter(variant, train=False)
    a = augmenter(patch, np.random.default_rng(1))
    b = augmenter(patch, np.random.default_rng(999))
    assert np.array_equal(a, b)
    assert np.array_equal(a, center_crop(patch, variant.side).astype(np.float32) / 255.0)


def disk_mask(size: int = 2000, radius: int = 945) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, 1, -1)
    return mask


def test_d_rotation_creates_no_black_when_margin_fits():
    mask = disk_mask()
    window = fit_window(mask, 1000.0, 1000.0, 468.0)
    assert window["rotation_ok"] and not window["margin_black"]

    variant = get_variant("R2.0_S512_G3")
    # Patch inteiramente válido no disco inscrito, como garante `rotation_ok`.
    patch = textured_patch(variant, 3)
    yy, xx = np.mgrid[:variant.patch, :variant.patch]
    center = (variant.patch - 1) / 2
    patch[np.hypot(xx - center, yy - center) > variant.patch / 2] = 0
    augmenter = Augmenter(variant, train=True)
    for seed in range(20):
        out = augmenter(patch, np.random.default_rng(seed))
        assert (out.max(axis=2) > 0).all(), f"preto criado com a semente {seed}"


def test_d_oversized_roi_is_flagged():
    window = fit_window(disk_mask(), 1000.0, 1000.0, 702.0)
    assert window["margin_black"] and not window["rotation_ok"]


def test_d_rotation_falls_back_to_d4_when_margin_missing():
    augmenter = Augmenter("R2.0_S512_G2", train=True)
    params = augmenter.sample_params(np.random.default_rng(0), rotation_ok=False)
    assert params["geometry"]["kind"] == "d4"


@pytest.mark.parametrize("recipe,check", [
    ("F1", lambda p: 0.9 <= p["brightness"] <= 1.1 and 0.9 <= p["contrast"] <= 1.1),
    ("F2", lambda p: 0.9 <= p["gamma"] <= 1.1),
    ("F3", lambda p: 0.9 <= p["saturation"] <= 1.1),
])
def test_e_parameters_within_limits(recipe, check):
    augmenter = Augmenter(f"R2.0_S512_{recipe}", train=True)
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert check(augmenter.sample_params(rng)["photometry"])


@pytest.mark.parametrize("recipe", ["F1", "F2", "F3", "Fmix"])
def test_e_intensity_change_is_bounded(recipe):
    variant = get_variant(f"R2.0_S512_{recipe}")
    patch = np.full((variant.patch, variant.patch, 3), (150, 110, 70), np.uint8)
    patch[::2] = (170, 120, 80)
    crop = center_crop(patch, variant.side).astype(np.float32) / 255.0
    augmenter = Augmenter(variant, train=True)
    for seed in range(50):
        out = augmenter(patch, np.random.default_rng(seed))
        assert 0.0 <= out.min() and out.max() <= 1.0
        ratio = out.mean() / crop.mean()
        assert 0.80 <= ratio <= 1.20  # brilho 0.9-1.1, gama 0.9-1.1 em valores médios


def test_multi_image_sample_uses_same_parameters():
    variant = get_variant("R2.0_S512_A2-G2")
    patch = textured_patch(variant, 4)
    first, second = Augmenter(variant, train=True)([patch, patch.copy()], np.random.default_rng(5))
    assert np.array_equal(first, second)


def test_registry_rejects_invalid_and_loads_all_ids():
    ids = list_variant_ids()
    assert ids == sorted(set(ids), key=ids.index)
    for variant_id in ids:
        Augmenter(variant_id, train=True)
    assert not any(v.startswith("R3.0_") and ("G2" in v or "G3" in v) for v in ids)
    with pytest.raises(ValueError):
        get_variant("R3.0_S512_G2")
    with pytest.raises(ValueError):
        get_variant("R2.0_S512_cutout")
    assert "hue_shift" in load_registry()["prohibited"]


def test_resize_is_antialiased():
    """Um padrão acima da nova frequência de Nyquist deve sumir, não virar padrão falso."""
    stripes = np.zeros((1000, 1000, 3), np.uint8)
    stripes[:, ::2] = 255
    out = resize_antialiased(stripes, 384)
    assert abs(float(out.mean()) - 127.5) < 2
    assert float(out.std()) < 25


# ---- (f) sensibilidade da auditoria ----

def synthetic_roi(native: int = 936, seed: int = 0, dd: float = 234.0) -> tuple[np.ndarray, np.ndarray]:
    """Fundo suave com manchas claras do tamanho de drusas e de baixo contraste."""
    rng = np.random.default_rng(seed)
    image = np.full((native, native), 90.0, np.float32)
    background = cv2.resize(rng.normal(0, 8, (9, 9)).astype(np.float32), (native, native), interpolation=cv2.INTER_CUBIC)
    image += background
    spots = np.zeros_like(image)
    for _ in range(350):
        x, y = rng.integers(40, native - 40, 2)
        diameter = rng.uniform(DIAMETERS_DD[0], DIAMETERS_DD[-1]) * dd
        cv2.circle(spots, (int(x), int(y)), max(1, int(diameter / 2)), float(rng.uniform(8, 25)), -1, cv2.LINE_AA)
    image += cv2.GaussianBlur(spots, (0, 0), 1.5)
    image += rng.normal(0, 0.6, image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8), np.ones((native, native), np.uint8)


@pytest.fixture(scope="module")
def audit_setup():
    from macdeg.audit import dog_responses

    native, valid = synthetic_roi()
    dd = 234.0
    thresholds = fit_thresholds([[r[r > 0] for r in dog_responses(native, dd)]])
    reference = make_reference(native, valid, dd, thresholds)
    side = 512
    scale = side / native.shape[0]
    matrix = np.array([[scale, 0, 0.5 * scale - 0.5], [0, scale, 0.5 * scale - 0.5], [0, 0, 1.0]])
    base = cv2.resize(native, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = np.ones((side, side), np.uint8)

    def run(image):
        return evaluate(reference, image, mask, scale * dd, matrix, thresholds)

    return base, run


def test_f_audit_baseline_retains_most_structures(audit_setup):
    base, run = audit_setup
    result = run(base)
    assert result["n_usable"] > 50
    assert result["retention"] > 0.6
    assert result["roi_area_lost"] < 1e-3


def test_f_audit_flags_blur(audit_setup):
    base, run = audit_setup
    clean, blurred = run(base), run(cv2.GaussianBlur(base, (0, 0), 1.5))
    assert blurred["retention"] < clean["retention"] - 0.05
    assert blurred["energy_ratio"] < clean["energy_ratio"] * 0.9


def test_f_audit_flags_noise(audit_setup):
    base, run = audit_setup
    noisy = base + np.random.default_rng(0).normal(0, 0.02 * 255, base.shape).astype(np.float32)
    assert run(noisy)["spurious_rate"] > run(base)["spurious_rate"] + 0.02


def test_f_audit_flags_downscale(audit_setup):
    base, run = audit_setup
    small = cv2.resize(base, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, base.shape[::-1], interpolation=cv2.INTER_CUBIC)
    clean, degraded = run(base), run(restored)
    assert degraded["retention"] < clean["retention"]
    assert degraded["energy_ratio"] < clean["energy_ratio"]


def test_f_audit_flags_jpeg(audit_setup):
    base, run = audit_setup
    _, data = cv2.imencode(".jpg", base.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 75])
    degraded = run(cv2.imdecode(data, cv2.IMREAD_GRAYSCALE).astype(np.float32))
    assert degraded["retention"] < run(base.astype(np.uint8).astype(np.float32))["retention"]


# ---- Checagens no cache real, se ele já foi gerado ----

needs_cache = pytest.mark.skipif(not (CACHE / "metadata.csv").exists(), reason="cache ainda não gerado")


@needs_cache
def test_cache_patch_sizes_and_no_black_outside_flagged():
    metadata = pd.read_csv(CACHE / "metadata.csv", dtype={"patient_id": str})
    rows = metadata[metadata["roi"].isin(["R1.5", "R2.0"]) & ~metadata["margin_black"]].sample(60, random_state=0)
    for _, row in rows.iterrows():
        variant = get_variant(f"{row['roi']}_S512_A0")
        patch = load_patch(CACHE, variant, row["file"])
        assert patch.shape[0] == patch_side(512)
        crop = center_crop(patch, variant.side)
        for corner in (crop[0, 0], crop[0, -1], crop[-1, 0], crop[-1, -1]):
            assert corner.max() > 0


@needs_cache
def test_cache_excludes_side_anomalies():
    metadata = pd.read_csv(CACHE / "metadata.csv", dtype={"patient_id": str})
    excluded = pd.read_csv(CACHE / "excluded.csv", dtype={"patient_id": str})
    assert not metadata["file"].isin(excluded["file"]).any()
