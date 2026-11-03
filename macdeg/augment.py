"""Augmentation por ID de variante, sobre os patches do cache.

Ordem: fotometria -> geometria -> recorte central -> qualidade (só controles negativos).
Validação e teste só normalizam. Uma amostra pode ter várias imagens do mesmo olho: todas
recebem os mesmos parâmetros.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "variants.yaml"
ROI_PATCH_MARGIN = 1.55  # cobre rotação, janela de 1.05x e deslocamento de 2%
FULL_FIELD_PATCH_MARGIN = 1.10
ROTATING_GEOMETRIES = {"rotate"}


def load_registry(path: Path | str = CONFIG_PATH, flag_summary: Path | str | None = None) -> dict:
    """Carrega o YAML. Com `flag_summary`, ROIs cuja margem não coube ficam só com G0 e G1."""
    registry = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if flag_summary is not None and Path(flag_summary).exists():
        allowed = json.loads(Path(flag_summary).read_text(encoding="utf-8"))["rotation_allowed"]
        for roi, is_allowed in allowed.items():
            if roi in registry["rois"] and not is_allowed:
                registry["rois"][roi]["rotation"] = False
    prohibited = set(registry["prohibited"])
    for section in ("geometry", "photometry", "quality"):
        for name, spec in registry[section].items():
            if spec["type"] in prohibited:
                raise ValueError(f"operação proibida no registro: {name}")
    return registry


def patch_side(side: int, full_field: bool = False) -> int:
    """Lado do patch no cache. A diferença para `side` é par, para o recorte central ser exato."""
    margin = FULL_FIELD_PATCH_MARGIN if full_field else ROI_PATCH_MARGIN
    return side + 2 * int(round(side * (margin - 1) / 2))


@dataclass(frozen=True)
class Variant:
    id: str
    roi: str
    resolution: str
    recipe: str
    side: int
    patch: int
    full_field: bool
    half_dd: float | None
    geometry: dict
    photometry: dict
    quality: dict
    negative_control: bool


def list_variant_ids(registry: dict | None = None) -> list[str]:
    """Todos os IDs válidos. ROIs sem rotação não geram receitas com rotação."""
    registry = registry or load_registry()
    ids = []
    for roi, roi_spec in registry["rois"].items():
        for resolution in registry["resolutions"]:
            for recipe, spec in registry["recipes"].items():
                rotates = registry["geometry"][spec["geometry"]]["type"] in ROTATING_GEOMETRIES
                if rotates and not roi_spec.get("rotation", False):
                    continue
                ids.append(f"{roi}_{resolution}_{recipe}")
    return ids


def get_variant(variant_id: str, registry: dict | None = None) -> Variant:
    registry = registry or load_registry()
    try:
        roi, resolution, recipe = variant_id.split("_")
        roi_spec, res_spec, spec = registry["rois"][roi], registry["resolutions"][resolution], registry["recipes"][recipe]
    except (ValueError, KeyError) as error:
        raise ValueError(f"ID de variante inválido: {variant_id}") from error
    geometry = registry["geometry"][spec["geometry"]]
    if geometry["type"] in ROTATING_GEOMETRIES and not roi_spec.get("rotation", False):
        raise ValueError(f"{roi} só aceita G0 e G1: {variant_id}")

    def expand(section: str, name: str) -> dict:
        item = dict(registry[section][name])
        if item["type"] == "one_of":
            item["options"] = {o: registry[section][o] for o in item["options"]}
        return {"name": name, **item}

    full_field = bool(roi_spec.get("full_field", False))
    return Variant(
        id=variant_id, roi=roi, resolution=resolution, recipe=recipe, side=res_spec["side"],
        patch=patch_side(res_spec["side"], full_field), full_field=full_field, half_dd=roi_spec.get("half_dd"),
        geometry={"name": spec["geometry"], **geometry},
        photometry=expand("photometry", spec["photometry"]),
        quality=expand("quality", spec["quality"]),
        negative_control=bool(spec.get("negative_control", False)),
    )


def cache_path(cache_root: Path | str, roi: str, resolution: str, file: str) -> Path:
    return Path(cache_root) / roi / resolution / file


def load_patch(cache_root: Path | str, variant: Variant, file: str) -> np.ndarray:
    """Patch do cache em RGB uint8."""
    path = cache_path(cache_root, variant.roi, variant.resolution, file)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr[..., ::-1].copy()


def geometry_matrix(params: dict, patch: int, side: int) -> np.ndarray:
    """Matriz 3x3 que leva coordenadas do patch às coordenadas da saída."""
    cp, cs = (patch - 1) / 2, (side - 1) / 2
    if params["kind"] == "d4":
        # Mesmo efeito de fliplr seguido de rot90(k), em coordenadas centradas com y para baixo.
        linear = np.eye(2)
        if params["flip"]:
            linear = np.array([[-1.0, 0.0], [0.0, 1.0]]) @ linear
        rot = np.array([[0.0, 1.0], [-1.0, 0.0]])
        linear = np.linalg.matrix_power(rot, params["k"] % 4) @ linear
        offset = np.zeros(2)
    elif params["kind"] == "rotate":
        theta = np.radians(params["angle"])
        linear = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) / params["window"]
        offset = np.array([params["shift_x"], params["shift_y"]]) * side
    else:
        linear, offset = np.eye(2), np.zeros(2)
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = linear @ (-(np.array([cp, cp]) + offset)) + np.array([cs, cs])
    return matrix


def transform_identity_params() -> dict:
    return {"kind": "none"}


def center_crop(image: np.ndarray, side: int) -> np.ndarray:
    start = (image.shape[0] - side) // 2
    return image[start:start + side, start:start + side]


def _valid(image: np.ndarray) -> np.ndarray:
    return image.max(axis=2) > 0


def apply_photometry(image: np.ndarray, params: dict | None) -> np.ndarray:
    """`image` é float32 em [0, 1]. Pixels fora do campo (pretos) não entram nem mudam."""
    if params is None:
        return image
    valid = _valid(image)
    if params["kind"] == "brightness_contrast":
        mean = image[valid].mean() if valid.any() else 0.0
        out = ((image - mean) * params["contrast"] + mean) * params["brightness"]
    elif params["kind"] == "gamma":
        out = np.power(image, params["gamma"])
    elif params["kind"] == "saturation":
        gray = (image @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
        out = gray + (image - gray) * params["saturation"]
    else:
        raise ValueError(params["kind"])
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    out[~valid] = 0.0
    return out


def apply_geometry(image: np.ndarray, params: dict, side: int) -> np.ndarray:
    """Devolve o recorte central com a geometria aplicada, sem reamostrar nos casos exatos."""
    if params["kind"] == "none":
        return center_crop(image, side).copy()
    if params["kind"] == "d4":
        out = np.fliplr(image) if params["flip"] else image
        return center_crop(np.rot90(out, params["k"]), side).copy()
    matrix = geometry_matrix(params, image.shape[0], side)
    out = cv2.warpAffine(image, matrix[:2], (side, side), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.clip(out, 0.0, 1.0)


def apply_quality(image: np.ndarray, params: dict | None, rng: np.random.Generator) -> np.ndarray:
    if params is None:
        return image
    kind = params["kind"]
    if kind == "gaussian_blur":
        return cv2.GaussianBlur(image, (0, 0), params["sigma"])
    if kind == "gaussian_noise":
        noise = rng.normal(0.0, params["std"], image.shape).astype(np.float32)
        return np.clip(image + noise, 0.0, 1.0)
    if kind == "downscale":
        h, w = image.shape[:2]
        small = cv2.resize(image, (max(1, round(w * params["factor"])), max(1, round(h * params["factor"]))),
                           interpolation=cv2.INTER_AREA)
        return np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    if kind == "jpeg":
        rgb8 = (image * 255.0 + 0.5).astype(np.uint8)
        _, data = cv2.imencode(".jpg", rgb8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, params["quality"]])
        return cv2.imdecode(data, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    raise ValueError(kind)


def _sample_photometry(spec: dict, rng: np.random.Generator) -> dict | None:
    kind = spec["type"]
    if kind == "none":
        return None
    if kind == "one_of":
        if rng.random() >= spec["p"]:
            return None
        names = list(spec["options"])
        return _sample_photometry(spec["options"][names[int(rng.integers(len(names)))]], rng)
    if kind == "brightness_contrast":
        limit = spec["limit"]
        return {"kind": kind, "brightness": float(rng.uniform(1 - limit, 1 + limit)),
                "contrast": float(rng.uniform(1 - limit, 1 + limit))}
    if kind == "gamma":
        return {"kind": kind, "gamma": float(rng.uniform(*spec["range"]))}
    if kind == "saturation":
        return {"kind": kind, "saturation": float(rng.uniform(1 - spec["limit"], 1 + spec["limit"]))}
    raise ValueError(kind)


def _sample_quality(spec: dict, rng: np.random.Generator) -> dict | None:
    kind = spec["type"]
    if kind == "none":
        return None
    if kind == "one_of":
        if rng.random() >= spec["p"]:
            return None
        names = list(spec["options"])
        return _sample_quality(spec["options"][names[int(rng.integers(len(names)))]], rng)
    if kind == "gaussian_blur":
        return {"kind": kind, "sigma": float(rng.uniform(*spec["sigma"]))}
    if kind == "gaussian_noise":
        return {"kind": kind, "std": float(rng.uniform(*spec["std"]))}
    if kind == "downscale":
        return {"kind": kind, "factor": float(rng.uniform(*spec["factor"]))}
    if kind == "jpeg":
        return {"kind": kind, "quality": int(rng.integers(spec["quality"][0], spec["quality"][1] + 1))}
    raise ValueError(kind)


def _sample_geometry(spec: dict, rng: np.random.Generator, rotation_ok: bool) -> dict:
    kind = spec["type"]
    if kind == "rotate" and not rotation_ok:
        kind = "d4"  # a margem desta imagem não comporta rotação: evita criar preto
    if kind == "none":
        return {"kind": "none"}
    if kind == "d4":
        return {"kind": "d4", "flip": bool(rng.integers(2)), "k": int(rng.integers(4))}
    limit = spec.get("shift_frac", 0.0)
    return {"kind": "rotate", "angle": float(rng.uniform(*spec["angle_deg"])),
            "window": float(rng.uniform(*spec.get("window", [1.0, 1.0]))),
            "shift_x": float(rng.uniform(-limit, limit)), "shift_y": float(rng.uniform(-limit, limit))}


class Augmenter:
    """Aplica uma variante. Com `train=False`, nenhuma transformação aleatória é usada."""

    def __init__(self, variant: Variant | str, train: bool = True, registry: dict | None = None):
        registry = registry or load_registry()
        self.variant = get_variant(variant, registry) if isinstance(variant, str) else variant
        self.train = train
        self.normalization = registry["normalization"]

    def sample_params(self, rng: np.random.Generator, rotation_ok: bool = True) -> dict:
        v = self.variant
        if not self.train:
            return {"photometry": None, "geometry": {"kind": "none"}, "quality": None}
        return {"photometry": _sample_photometry(v.photometry, rng),
                "geometry": _sample_geometry(v.geometry, rng, rotation_ok),
                "quality": _sample_quality(v.quality, rng)}

    def apply_one(self, patch: np.ndarray, params: dict, rng: np.random.Generator | None = None) -> np.ndarray:
        """`patch` é RGB uint8 com lado `variant.patch`. Saída: float32 em [0, 1], sem normalizar."""
        if patch.shape[0] != self.variant.patch or patch.shape[1] != self.variant.patch:
            raise ValueError("patch com lado diferente do esperado pela variante")
        image = patch.astype(np.float32) / 255.0
        image = apply_photometry(image, params["photometry"])
        image = apply_geometry(image, params["geometry"], self.variant.side)
        rng = rng if rng is not None else np.random.default_rng(0)
        return apply_quality(image, params["quality"], rng)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        mean, std = self.normalization["mean"], self.normalization["std"]
        if mean is None or std is None:
            return image
        return (image - np.array(mean, np.float32)) / np.array(std, np.float32)

    def __call__(self, images: np.ndarray | list[np.ndarray], rng: np.random.Generator | None = None,
                 rotation_ok: bool = True) -> np.ndarray | list[np.ndarray]:
        """Uma imagem ou uma lista (visitas do mesmo olho), sempre com os mesmos parâmetros."""
        rng = rng if rng is not None else np.random.default_rng()
        params = self.sample_params(rng, rotation_ok)
        single = isinstance(images, np.ndarray)
        batch = [images] if single else images
        out = [self.normalize(self.apply_one(p, params, rng)) for p in batch]
        return out[0] if single else out
