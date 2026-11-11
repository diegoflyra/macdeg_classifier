"""Modelos no Hugging Face: um repositório PRIVADO com uma pasta por configuração publicada.

  <usuário>/macdeg_classifier
    README.md                       ranking das configurações publicadas, do pior para o melhor
    ranking.json                    o mesmo ranking, para o código ler
    <configuração>/config.json      arquitetura, tamanho da imagem, normalização, limiar, métricas
    <configuração>/dobra<k>.safetensors   os 5 modelos do ensemble (float16)
Nenhuma imagem, previsão por imagem ou dado de paciente vai para o Hugging Face.

Usar um modelo publicado:
    from macdeg.hub import load_ensemble
    modelo = load_ensemble("<usuário>/macdeg_classifier", "referencia-00")
    prob = modelo.predict(imagem_bgr)        # média das 5 redes; DMRI se prob >= modelo.threshold
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from macdeg.ambiente import secret
from macdeg.cluster import task_dir

REPO_NAME = os.environ.get("HF_REPO", "macdeg_classifier")  # repositório do tipo "model"


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=secret("HF_TOKEN") or None)


def repo_id(api=None) -> str:
    api = api or _api()
    return f"{api.whoami()['name']}/{REPO_NAME}"


def _source_metrics(batch: Path, lote: dict, name: str) -> dict:
    """Métricas da validação cruzada aninhada da configuração (lote de origem do treino final)."""
    origin = Path(lote["lote_de_origem"]) / "analise" / "relatorio.json"
    if not origin.exists():
        return {}
    r = json.loads(origin.read_text())["configuracoes"].get(name, {})
    keys = ("f2_media_bases", "f2_media_bases_ic95", "auroc", "sensibilidade", "especificidade_normais",
            "especificidade_outras_doencas", "regua_origem_auroc")
    return {k: r.get(k) for k in keys} | {"por_base": {b: {"f2": v["f2"], "auroc": v["auroc"]}
                                                       for b, v in r.get("por_base", {}).items()}}


def upload_final(batch: Path | str, name: str, final: dict) -> str:
    """Envia os 5 modelos do treino final de `name`, com config e limiar, e atualiza o ranking."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    batch = Path(batch)
    lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    config = next(c for c in lote["configuracoes"] if c["nome"] == name)
    api = _api()
    repo = repo_id(api)
    api.create_repo(repo, private=True, exist_ok=True)
    metrics = _source_metrics(batch, lote, name)
    card = {"configuracao": name, "lote_final": lote["nome"], "commit": lote["commit"],
            "assinatura_dados": lote["assinatura_dados"], "modelo": config["config"]["model"],
            "image_size": config["config"]["image_size"], "normalizacao": "ImageNet (média 0.485 0.456 0.406; dp 0.229 0.224 0.225)",
            "entrada": "retina inteira num quadrado (bordas pretas completam), RGB", "saida": "1 logit; sigmoide = P(DMRI)",
            "ensemble": "média das probabilidades das 5 dobras", "limiar": final["limiar"],
            "mudancas": config["mudancas"], "metricas_validacao_aninhada": metrics,
            "publicado": time.strftime("%Y-%m-%d %H:%M")}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / name
        folder.mkdir()
        for task in lote["tarefas"]:
            if task["configuracao"] == name:
                state = torch.load(task_dir(batch, task) / "best.pt", map_location="cpu")
                save_file({k: v.half().contiguous() if v.is_floating_point() else v.contiguous() for k, v in state.items()},
                          folder / f"dobra{task['outer']}.safetensors")
        (folder / "config.json").write_text(json.dumps(card, indent=1, ensure_ascii=False, default=float))
        try:
            ranking = json.loads(Path(hf_hub_download(repo, "ranking.json", token=secret("HF_TOKEN"))).read_text())
        except Exception:
            ranking = []
        ranking = [r for r in ranking if r["configuracao"] != name] + [
            {"configuracao": name, "f2_media_bases": metrics.get("f2_media_bases"),
             "f2_ic95": metrics.get("f2_media_bases_ic95"), "auroc": metrics.get("auroc"),
             "limiar": final["limiar"], "publicado": card["publicado"]}]
        ranking.sort(key=lambda r: (r["f2_media_bases"] is None, r["f2_media_bases"] or 0))
        (Path(tmp) / "ranking.json").write_text(json.dumps(ranking, indent=1, default=float))
        (Path(tmp) / "README.md").write_text(_readme(ranking))
        api.upload_folder(repo_id=repo, folder_path=tmp, commit_message=f"publica {name}")
    return f"https://huggingface.co/{repo}/tree/main/{name}"


def _readme(ranking: list) -> str:
    lines = ["---", "license: other", "tags: [fundus, amd, dmri, resnet]", "---", "",
             "# Classificador de DMRI em retinografias (macdeg)", "",
             "Modelos privados de pesquisa. Cada pasta é uma configuração, com 5 redes (ensemble das dobras),",
             "o limiar e as métricas da validação cruzada aninhada. Nenhum dado de paciente é publicado aqui.", "",
             "Métrica principal: média do F2 dentro de cada base. Ranking do pior para o melhor:", "",
             "| # | configuração | F2 médio por base [IC95] | AUROC | limiar | publicado |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(ranking, 1):
        ci = r.get("f2_ic95") or [None, None]
        f2 = "—" if r["f2_media_bases"] is None else f"{r['f2_media_bases']:.3f}"
        ci_text = "" if ci[0] is None else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
        auc = "—" if r.get("auroc") is None else f"{r['auroc']:.3f}"
        lines.append(f"| {i} | `{r['configuracao']}` | {f2}{ci_text} | {auc} | {r['limiar']:.4f} | {r['publicado']} |")
    return "\n".join(lines) + "\n"


class Ensemble:
    def __init__(self, models: list, card: dict, device: torch.device):
        self.models, self.card, self.device = models, card, device
        self.threshold = float(card["limiar"])

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> float:
        """Probabilidade de DMRI de uma retinografia (BGR, como o cv2.imread lê)."""
        import cv2
        from macdeg.data.build import fov_square
        from macdeg.train import IMAGENET_MEAN, IMAGENET_STD
        square = fov_square(image_bgr, self.card["image_size"])
        x = torch.from_numpy(cv2.cvtColor(square, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        x = ((x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None])[None]
        x = x.to(self.device)
        return float(np.mean([torch.sigmoid(m(x)).item() for m in self.models]))


def load_ensemble(repo: str, name: str, device: str | None = None) -> Ensemble:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from macdeg.train import build_model
    folder = Path(snapshot_download(repo, allow_patterns=[f"{name}/*"], token=secret("HF_TOKEN") or None)) / name
    card = json.loads((folder / "config.json").read_text())
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = []
    for path in sorted(folder.glob("dobra*.safetensors")):
        model = build_model(card["modelo"], pretrained=False)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in load_file(path).items()})
        models.append(model.to(device).eval())
    return Ensemble(models, card, device)
