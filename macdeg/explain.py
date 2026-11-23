"""Explicabilidade: para onde a rede olha. Grad-CAM e mapas de atenção, com números agregados.

Em cada tarefa (macdeg.train.run, opção `explain`, padrão ligado), o modelo escolhido gera mapas para as
imagens do teste externo, que não influenciaram nada no treino, e grava `explain_maps.npz`:
sample_id + um mapa por imagem, normalizado para somar 1 (uma distribuição sobre a imagem):
  dmri      Grad-CAM da saída de DMRI no mapa mais profundo (16 x 16 na V2-S em 512 px)
  drusas    Grad-CAM da saída de drusas (com aux_drusas)
  atencao   softmax dos escores da atenção da híbrida (hibrida_v2s)
  macula    Grad-CAM da saída de DMRI no ramo da mácula (v2s_macula); cobre só o recorte central
Nada de imagem: só números. A análise (macdeg.analysis) agrega por tipo de olho e por acerto/erro.

Pranchas para olhar (imagem + mapa), SÓ no cluster ou no computador local, nunca no GitHub:
    python -m macdeg.explain pranchas ~/macdeg_runs/<lote> [--n 8]
grava <lote>/explicabilidade/<configuração>/<grupo>.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

MAP_SIDE = 32
MACULA_RADIUS = 0.23     # raio da grade do AREDS (3 mm) sobre ~13 mm de retina, em frações do lado
FILE = "explain_maps.npz"


def _deepest_4d(model: nn.Module, images: torch.Tensor) -> nn.Module | None:
    """Último módulo executado que devolve um mapa 4D com grade >= 2 x 2 (o mapa mais profundo)."""
    found, hooks = [], []
    for module in model.modules():
        if len(list(module.children())):
            continue
        hooks.append(module.register_forward_hook(
            lambda m, i, o: found.append(m) if torch.is_tensor(o) and o.ndim == 4 and min(o.shape[-2:]) >= 2 else None))
    with torch.no_grad():
        model(images[:1])
    for hook in hooks:
        hook.remove()
    return found[-1] if found else None


def cam_layers(model: nn.Module, images: torch.Tensor) -> dict[str, nn.Module]:
    """Camada do Grad-CAM de cada mapa. Híbrida e mácula: as camadas conhecidas; as outras: o mapa 4D
    mais profundo (ResNet: layer4; EfficientNet: bn2; ViT não tem mapa 4D e fica sem Grad-CAM)."""
    from macdeg.hybrid import HybridV2S, MaculaV2S
    if isinstance(model, MaculaV2S):
        return {"dmri": model.backbone.bn2, "macula": model.macula.bn2}
    if isinstance(model, HybridV2S):
        return {"dmri": model.backbone.bn2}
    layer = _deepest_4d(model, images)
    return {"dmri": layer} if layer is not None else {}


def _normalize(maps: torch.Tensor) -> np.ndarray:
    maps = F.interpolate(maps.detach()[:, None].float(), size=(MAP_SIDE, MAP_SIDE), mode="bilinear",
                         align_corners=False)[:, 0]
    maps = maps.clamp(min=0)
    total = maps.flatten(1).sum(1).clamp(min=1e-12)
    return (maps / total[:, None, None]).cpu().numpy().astype(np.float16)


def grad_cam(model: nn.Module, layer: nn.Module, images: torch.Tensor, column: int) -> np.ndarray:
    """Grad-CAM (Selvaraju et al.): ativações ponderadas pela média do gradiente da saída `column`."""
    store = {}
    hook = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        with torch.enable_grad():
            logits = model(images)
            activation = store["a"]
            grad, = torch.autograd.grad(logits[:, column].float().sum(), activation)
    finally:
        hook.remove()
    weights = grad.float().mean(dim=(2, 3), keepdim=True)
    return _normalize((weights * activation.float()).sum(1))


def explain(model: nn.Module, loader, device, sample_ids: list[str], outputs: dict[str, int]) -> dict[str, np.ndarray]:
    """outputs: {nome do mapa: coluna da saída}, ex. {"dmri": 0, "drusas": 2}. Modelo em eval()."""
    from macdeg.hybrid import HybridV2S
    model.eval()
    layers = None
    maps: dict[str, list] = {}
    for images, *_ in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        if layers is None:
            layers = cam_layers(model, images)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            for name, column in outputs.items():
                if "dmri" in layers:
                    maps.setdefault(name, []).append(grad_cam(model, layers["dmri"], images, column))
            if "macula" in layers:
                maps.setdefault("macula", []).append(grad_cam(model, layers["macula"], images, 0))
            if isinstance(model, HybridV2S):
                with torch.no_grad():
                    model(images)
                scores = model.last_attention.float()
                maps.setdefault("atencao", []).append(_normalize(scores.flatten(1).softmax(1).reshape(scores.shape)))
    result = {name: np.concatenate(parts) for name, parts in maps.items()}
    result["sample_id"] = np.array(sample_ids)
    return result


def _grid() -> tuple[np.ndarray, np.ndarray]:
    centers = (np.arange(MAP_SIDE) + 0.5) / MAP_SIDE - 0.5
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    return yy, xx


def map_stats(maps: np.ndarray, region: float = 1.0) -> pd.DataFrame:
    """Por mapa: massa dentro da mácula (raio da grade do AREDS em torno do centro), massa fora da retina
    (fora do círculo inscrito: bordas pretas) e área que concentra metade da massa (espalhamento).
    `region`: fração do lado coberta pelo mapa (o da mácula cobre só o recorte central)."""
    yy, xx = _grid()
    radius = np.hypot(yy, xx) * region
    maps = maps.astype(np.float32).reshape(len(maps), -1)
    ordered = -np.sort(-maps, axis=1)
    half_area = (np.cumsum(ordered, axis=1) < 0.5).sum(axis=1) / maps.shape[1] * region ** 2
    return pd.DataFrame({"massa_macula": maps[:, radius.ravel() <= MACULA_RADIUS].sum(1),
                         "massa_fora_retina": maps[:, radius.ravel() > 0.5].sum(1),
                         "area_metade": half_area})


def summarize(maps_by_task: list[dict], predictions: pd.DataFrame, details: pd.DataFrame | None) -> dict:
    """Média dos mapas de cada imagem entre as tarefas em que ela foi teste; números por grupo:
    tipo de olho (DMRI precoce/tardia, normal, DR, outras doenças, drusas) e erros (falso negativo/positivo)."""
    from macdeg.hybrid import MACULA_FRACTION
    names = sorted({k for m in maps_by_task for k in m if k != "sample_id"})
    groups = predictions[["sample_id", "label", "kind", "disease_group"]].copy()
    if "pred" in predictions:
        groups["pred"] = predictions["pred"]
    if details is not None:
        groups = groups.merge(details[["sample_id", "label_detail"]], on="sample_id", how="left")
    result = {}
    for name in names:
        frames = [pd.DataFrame(m[name].reshape(len(m[name]), -1).astype(np.float32)).assign(sample_id=m["sample_id"])
                  for m in maps_by_task if name in m]
        mean = pd.concat(frames).groupby("sample_id").mean()
        stats = map_stats(mean.to_numpy().reshape(-1, MAP_SIDE, MAP_SIDE),
                          region=MACULA_FRACTION if name == "macula" else 1.0).assign(sample_id=mean.index.to_numpy())
        table = stats.merge(groups, on="sample_id", how="left")
        labels = {
            "dmri": table["label"] == 1,
            "dmri_precoce": table.get("label_detail", pd.Series(dtype=str)).eq("dmri_precoce"),
            "dmri_intermediaria_tardia": table.get("label_detail", pd.Series(dtype=str)).eq("dmri_intermediaria_tardia"),
            "normal": table["kind"] == "normal",
            "retinopatia_diabetica": table["disease_group"].fillna("").astype(str).str.contains("retinopatia_diabetica"),
            "outras_doencas": (table["kind"] == "outra"),
            "drusas": table["kind"] == "drusas",
        }
        if "pred" in table:
            labels["falso_negativo"] = (table["label"] == 1) & (table["pred"] == 0)
            labels["falso_positivo"] = (table["label"] == 0) & (table["pred"] == 1)
        rows = []
        for group, mask in labels.items():
            part = table[mask.reindex(table.index, fill_value=False).astype(bool)]
            if len(part):
                rows.append({"grupo": group, "n": len(part),
                             **{c: round(float(part[c].mean()), 4) for c in ("massa_macula", "massa_fora_retina", "area_metade")}})
        result[name] = rows
    return result


def boards(batch: Path, n: int = 8) -> None:
    """Pranchas por configuração e grupo: imagem (512 px) com o mapa médio por cima. Só local/cluster."""
    import json

    import cv2
    from macdeg.data.sources import resolve
    lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    root = resolve(lote["configuracoes"][0]["config"]["dataset"])
    manifest = pd.read_csv(root / "manifest.csv", usecols=["sample_id", "image", "label", "label_detail", "disease"])
    for config in lote["configuracoes"]:
        name = config["nome"]
        tasks = sorted((batch / "tarefas" / name).glob(f"outer*_inner*/{FILE}"))
        if not tasks:
            continue
        loaded = [dict(np.load(t, allow_pickle=False)) for t in tasks]
        for map_name in sorted({k for m in loaded for k in m if k != "sample_id"}):
            frames = [pd.DataFrame(m[map_name].reshape(len(m[map_name]), -1).astype(np.float32)).assign(sample_id=m["sample_id"])
                      for m in loaded if map_name in m]
            mean = pd.concat(frames).groupby("sample_id").mean()
            table = manifest[manifest["sample_id"].isin(mean.index)]
            groups = {"dmri_precoce": table["label_detail"].eq("dmri_precoce"),
                      "dmri": table["label"].eq(1), "normal": table["label_detail"].eq("normal"),
                      "retinopatia_diabetica": table["disease"].fillna("").str.contains("retinopatia_diabetica")}
            out = batch / "explicabilidade" / name
            out.mkdir(parents=True, exist_ok=True)
            for group, mask in groups.items():
                chosen = table[mask].sample(min(n, int(mask.sum())), random_state=0) if mask.any() else table.head(0)
                tiles = []
                for _, row in chosen.iterrows():
                    image = cv2.resize(cv2.imread(str(root / row["image"])), (512, 512), interpolation=cv2.INTER_AREA)
                    heat = mean.loc[row["sample_id"]].to_numpy().reshape(MAP_SIDE, MAP_SIDE)
                    heat = cv2.resize(heat / max(heat.max(), 1e-12), (512, 512), interpolation=cv2.INTER_CUBIC)
                    color = cv2.applyColorMap(np.uint8(255 * heat.clip(0, 1)), cv2.COLORMAP_JET)
                    tiles.append(np.hstack([image, cv2.addWeighted(image, 0.55, color, 0.45, 0)]))
                if tiles:
                    cv2.imwrite(str(out / f"{map_name}_{group}.png"), np.vstack(tiles))
        print(f"{name}: pranchas em {batch / 'explicabilidade' / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pranchas")
    p.add_argument("lote", type=Path)
    p.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    boards(args.lote.expanduser(), args.n)


if __name__ == "__main__":
    main()
