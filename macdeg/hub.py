"""Modelo final: exportado no cluster e, quando você decidir, enviado ao Hugging Face.

1. O treino final (`--final`) EXPORTA o ensemble para o próprio lote, sem enviar nada:
    ~/macdeg_runs/<lote>-final-<data>/modelo/<configuração>/
      dobra<k>.safetensors   os 5 modelos (float16)
      config.json            de onde veio (lote de origem, commit, dados), limiar e métricas da validação aninhada
    O cartão (config.json, sem pesos) também vai para a branch `resultados` do GitHub, como registro de qual
    modelo foi o melhor. Os pesos ficam só no cluster.
2. Para publicar depois (repositório PRIVADO):
    python -m macdeg.hub publicar ~/macdeg_runs/<lote final> <configuração>

Repositório no Hugging Face: uma pasta por configuração publicada.

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


def export_final(batch: Path | str, name: str, final: dict) -> Path:
    """Grava o ensemble do treino final de `name` em <lote>/modelo/<name>/ (5 x float16 + cartão) e apaga
    os best.pt (float32, o dobro do tamanho). Nada sai do cluster."""
    from safetensors.torch import save_file
    batch = Path(batch)
    lote = json.loads((batch / "lote.json").read_text(encoding="utf-8"))
    config = next(c for c in lote["configuracoes"] if c["nome"] == name)
    metrics = _source_metrics(batch, lote, name)
    card = {"configuracao": name, "lote_final": lote["nome"], "commit": lote["commit"],
            "assinatura_dados": lote["assinatura_dados"], "modelo": config["config"]["model"],
            "image_size": config["config"]["image_size"], "normalizacao": "ImageNet (média 0.485 0.456 0.406; dp 0.229 0.224 0.225)",
            "entrada": "retina inteira num quadrado (bordas pretas completam), RGB", "saida": "logit 0; sigmoide = P(DMRI) (logit 1, se houver, é a cabeça auxiliar de DR)",
            "saidas": 2 if config["config"].get("aux_dr") else 1, "preprocess": config["config"].get("preprocess"),
            "ensemble": "média das probabilidades das 5 dobras", "limiar": final["limiar"],
            "mudancas": config["mudancas"], "lote_de_origem": Path(lote["lote_de_origem"] or "").name,
            "metricas_validacao_aninhada": metrics, "exportado": time.strftime("%Y-%m-%d %H:%M")}
    folder = batch / "modelo" / name
    folder.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in lote["tarefas"] if t["configuracao"] == name]
    for task in tasks:
        weights = task_dir(batch, task) / "best.pt"
        target = folder / f"dobra{task['outer']}.safetensors"
        if weights.exists():
            state = torch.load(weights, map_location="cpu")
            save_file({k: v.half().contiguous() if v.is_floating_point() else v.contiguous() for k, v in state.items()},
                      target)
            weights.unlink()
        elif not target.exists():
            raise FileNotFoundError(f"faltam os pesos da dobra {task['outer']} de {name}")
    card["dobras"] = sorted(t["outer"] for t in tasks)
    (folder / "config.json").write_text(json.dumps(card, indent=1, ensure_ascii=False, default=float))
    return folder


def upload_final(batch: Path | str, name: str) -> str:
    """Envia ao Hugging Face (privado) o ensemble já exportado de `name` e atualiza o ranking do repositório."""
    import shutil
    from huggingface_hub import hf_hub_download
    folder = Path(batch).expanduser() / "modelo" / name
    if not (folder / "config.json").exists():
        raise SystemExit(f"não achei {folder}/config.json: rode o treino final (--final) antes")
    card = json.loads((folder / "config.json").read_text())
    metrics = card.get("metricas_validacao_aninhada", {})
    final = {"limiar": card["limiar"]}
    api = _api()
    repo = repo_id(api)
    api.create_repo(repo, private=True, exist_ok=True)
    card["publicado"] = time.strftime("%Y-%m-%d %H:%M")
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(folder, Path(tmp) / name)
        (Path(tmp) / name / "config.json").write_text(json.dumps(card, indent=1, ensure_ascii=False, default=float))
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
        from macdeg.cache import clahe
        square = cv2.cvtColor(fov_square(image_bgr, self.card["image_size"]), cv2.COLOR_BGR2RGB)
        if self.card.get("preprocess") == "clahe":
            square = clahe(square)
        x = torch.from_numpy(square).permute(2, 0, 1).float() / 255
        x = ((x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None])[None]
        x = x.to(self.device)
        return float(np.mean([torch.sigmoid(m(x)[0, 0]).item() for m in self.models]))


def load_ensemble(repo: str, name: str, device: str | None = None) -> Ensemble:
    """`repo`: repositório do Hugging Face ("<usuário>/macdeg_classifier") ou pasta de um lote final local."""
    from safetensors.torch import load_file
    from macdeg.train import build_model
    local = Path(repo).expanduser() / "modelo" / name
    if local.exists():
        folder = local
    else:
        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(repo, allow_patterns=[f"{name}/*"], token=secret("HF_TOKEN") or None)) / name
    card = json.loads((folder / "config.json").read_text())
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = []
    for path in sorted(folder.glob("dobra*.safetensors")):
        model = build_model(card["modelo"], pretrained=False, outputs=card.get("saidas", 1))
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in load_file(path).items()})
        models.append(model.to(device).eval())
    return Ensemble(models, card, device)


def main() -> None:
    import argparse
    from macdeg.ambiente import load_dotenv
    ap = argparse.ArgumentParser(description="Publica no Hugging Face (privado) um modelo final já exportado.")
    ap.add_argument("acao", choices=["publicar"])
    ap.add_argument("lote", type=Path, help="pasta do lote final, ex.: ~/macdeg_runs/referencia-...-final-...")
    ap.add_argument("configuracao", help="ex.: referencia-00")
    args = ap.parse_args()
    load_dotenv()
    print(upload_final(args.lote, args.configuracao))


if __name__ == "__main__":
    main()
