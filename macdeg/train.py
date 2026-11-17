"""Treina o classificador DMRI x sem DMRI numa dobra da validação cruzada aninhada.

Uso: python -m macdeg.train configs/train/baseline.yaml --outer 0 [--inner 2] [--seed 0] [--set chave=valor ...]

- Com --inner j: treina nas outras 3 partes internas da dobra externa k, valida na parte j (escolhe a
  época pela AUROC) e, no fim, dá nota ao teste da dobra externa k, que não influenciou nada.
- Sem --inner ("final"): treina em tudo menos a dobra externa k e valida nela. Serve para o modelo
  final (ensemble das 5 dobras), não para medir desempenho.

Saída em <out>/:
  config.json            configuração efetiva
  history.csv            perda e AUROC de validação por época
  val_predictions.csv    nota de cada imagem de validação
  test_predictions.csv   nota de cada imagem do teste externo (só com --inner)
  extra_predictions.csv  nota das imagens com drusas sem diagnóstico (fora do treino)
  metrics.json           AUROC geral e por base (o F2 precisa das outras dobras: macdeg.analysis)
  last.pt / best.pt      pesos; last.pt permite retomar se a tarefa for interrompida e é apagado no fim;
                         best.pt fica só no treino final (ou com keep_weights: true)
As previsões trazem só o ID sequencial (sample_id), nunca o nome nem o hash do arquivo original.
Uma execução já concluída (metrics.json existe) é pulada.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from macdeg import cache as image_cache
from macdeg.data.sources import resolve
from macdeg.data.splits import kind
from macdeg.metrics import auroc, evaluate

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_config(path: Path, overrides: list[str]) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for item in overrides:
        key, value = item.split("=", 1)
        target = config
        *parents, leaf = key.split(".")
        for parent in parents:
            target = target[parent]
        target[leaf] = yaml.safe_load(value)
    return config


class FundusDataset(Dataset):
    """Lê do cache de imagens reduzidas (macdeg.cache) quando há; senão, do JPEG. Os pixels são os mesmos."""

    def __init__(self, frame: pd.DataFrame, root: Path, transform, cache: tuple[Path, dict] | None = None):
        self.paths = [str(root / p) for p in frame["image"]]
        self.labels = frame["label"].to_numpy(np.float32)
        self.transform = transform
        self.cache_path, lines = cache or (None, {})
        self.rows = [lines.get(p, -1) for p in frame["image"]]
        self._data = None                       # aberto em cada processo do DataLoader (mmap)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        row = self.rows[index]
        if row >= 0:
            if self._data is None:
                self._data = np.load(self.cache_path, mmap_mode="r")
            image = np.array(self._data[row])
        else:
            image = image_cache.read_rgb(self.paths[index])
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        return self.transform(tensor), self.labels[index]


def transforms(config: dict, train: bool):
    size, aug = config["image_size"], config.get("augment") or {}
    steps = []
    if train and aug:  # augment vazio ou null = sem augmentation: só redimensiona
        steps.append(v2.RandomResizedCrop(size, scale=tuple(aug.get("crop_scale", (0.85, 1.0))), ratio=(0.95, 1.05),
                                          antialias=True))
        if aug.get("hflip", True):
            steps.append(v2.RandomHorizontalFlip())
        if aug.get("rotate_deg", 0):
            steps.append(v2.RandomRotation(aug["rotate_deg"]))
        if aug.get("color", 0):
            steps.append(v2.ColorJitter(brightness=aug["color"], contrast=aug["color"]))
    else:
        steps.append(v2.Resize(size, antialias=True))
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(steps)


def build_model(name: str, pretrained: bool, *, freeze_first_convs: int = 0,
                head_init: str | None = None) -> nn.Module:
    if head_init not in (None, "xavier"):
        raise ValueError("head_init deve ser null ou xavier.")
    if head_init is not None and name != "alexnet":
        raise ValueError("head_init só é suportado para model: alexnet.")
    if type(freeze_first_convs) is not int or not 0 <= freeze_first_convs <= 5:
        raise ValueError("freeze_first_convs deve ser um inteiro entre 0 e 5.")
    if freeze_first_convs and name != "alexnet":
        raise ValueError("freeze_first_convs só é suportado para model: alexnet.")
    if name == "alexnet":
        from macdeg.models import AlexNet, freeze_alexnet_features, load_alexnet_imagenet_weights
        model = AlexNet()
        if pretrained:
            load_alexnet_imagenet_weights(model)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        if head_init == "xavier":
            nn.init.xavier_uniform_(model.classifier[-1].weight, gain=1.0)
            nn.init.zeros_(model.classifier[-1].bias)
        freeze_alexnet_features(model, freeze_first_convs)
        return model
    if name in ("resnet18", "resnet34", "resnet50"):
        import torchvision.models as tvm
        weights = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1, "resnet34": tvm.ResNet34_Weights.IMAGENET1K_V1,
                   "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2}[name] if pretrained else None
        model = getattr(tvm, name)(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=1)


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Taxa opcional para a saída nova; demais parâmetros mantêm config['lr']."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    head_lr = config.get("head_lr")
    if head_lr is not None:
        if config.get("model") != "alexnet":
            raise ValueError("head_lr só é suportado para model: alexnet.")
        if not math.isfinite(head_lr) or head_lr <= 0:
            raise ValueError("head_lr deve ser positivo e finito.")
        head = [p for p in model.classifier[-1].parameters() if p.requires_grad]
        head_ids = {id(p) for p in head}
        parameters = [{"params": [p for p in parameters if id(p) not in head_ids], "lr": config["lr"]},
                      {"params": head, "lr": head_lr}]
    return torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))


PREDICTION_COLUMNS = ["sample_id", "source", "group_id", "label", "kind", "disease_group"]


def split_frames(config: dict, outer: int, inner: int | None, seed: int):
    """(treino, validação, teste externo ou None, drusas ou None, raiz do conjunto)."""
    root = resolve(config["dataset"])
    manifest = pd.read_csv(root / "manifest.csv", dtype={"patient_id": str})
    splits = pd.read_csv(root / config.get("splits", "splits_nested_seed0.csv"))
    table = manifest.merge(splits, on="uid", validate="one_to_one").assign(kind=lambda t: kind(t))
    if config.get("sources"):
        table = table[table["source"].isin(config["sources"])]
    if not config.get("other_diseases", True):
        table = table[table["kind"] != "outra"]
    rest, test = table[table["outer"] != outer], table[table["outer"] == outer]
    if inner is None:
        train, val, test = rest.copy(), test.copy(), None
    else:
        column = rest[f"inner_{outer}"]
        train, val = rest[column != inner].copy(), rest[column == inner].copy()
    if config.get("shuffle_labels"):
        # Controle negativo: rótulos embaralhados no treino. A AUROC de validação deve ficar perto de 0,5.
        train["label"] = np.random.default_rng(seed).permutation(train["label"].to_numpy())
    for name, frame in (("treino", train), ("validação", val)):
        if frame["label"].nunique() < 2:
            raise SystemExit(f"O conjunto de {name} só tem uma classe: faltam imagens com ou sem DMRI.")
    extra = None
    if (root / "extras_drusas.csv").exists():
        extra = pd.read_csv(root / "extras_drusas.csv").assign(label=-1, kind="drusas", disease_group=None)
        if config.get("sources"):
            extra = extra[extra["source"].isin(config["sources"])]
    if config.get("limit"):
        n = config["limit"]
        train, val = train.sample(min(n, len(train)), random_state=seed), val.sample(min(n, len(val)), random_state=seed)
        test = test.sample(min(n, len(test)), random_state=seed) if test is not None else None
        extra = extra.sample(min(n, len(extra)), random_state=seed) if extra is not None else None
    reset = lambda f: f.reset_index(drop=True) if f is not None else None
    return reset(train), reset(val), reset(test), reset(extra), root


# Memória por imagem da ResNet-50 em 512 px com AMP, medida (4,2 GB com lote 16), com folga.
GB_PER_IMAGE_512 = 0.3


def micro_batch(config: dict, device: torch.device) -> int:
    """Pedaço do lote que cabe na GPU. O lote efetivo (batch_size) não muda: os gradientes dos pedaços
    se acumulam antes de cada passo, então o resultado não depende da GPU sorteada pela fila."""
    batch = config["batch_size"]
    if config.get("micro_batch"):
        return int(config["micro_batch"])
    if device.type != "cuda":
        return batch
    free = torch.cuda.get_device_properties(device).total_memory / 1e9 * 0.75
    per_image = GB_PER_IMAGE_512 * (config["image_size"] / 512) ** 2
    fits = max(1, int(free // per_image))
    return max(d for d in range(1, batch + 1) if batch % d == 0 and d <= fits)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for images, _ in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last))
        probs.append(torch.sigmoid(logits.float()).squeeze(1).cpu().numpy())
    return np.concatenate(probs)


def run(config: dict, outer: int, inner: int | None, seed: int, out: Path, log=None, on_model=None) -> dict | None:
    """Treina uma dobra. `log(dict)` recebe as métricas de cada época e `on_model(model)` recebe o modelo
    antes do treino (ex.: W&B, para os histogramas de pesos e gradientes)."""
    if (out / "metrics.json").exists():
        print(f"{out}: já concluído, pulando")
        return None
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    train, val, test, extra, root = split_frames(config, outer, inner, seed)
    (out / "config.json").write_text(json.dumps({**config, "outer": outer, "inner": inner, "seed": seed}, indent=1),
                                     encoding="utf-8")
    print(f"treino: {len(train)} imagens ({int(train['label'].sum())} DMRI); "
          f"validação: {len(val)} ({int(val['label'].sum())} DMRI); dispositivo: {device}", flush=True)

    workers = config.get("num_workers", 6)
    cached = None
    if config.get("cache", True):   # mesmos pixels, sem decodificar 1024 px a cada época (macdeg.cache)
        frames = [f for f in (train, val, test, extra) if f is not None]
        cached = image_cache.build(root, [i for f in frames for i in f["image"]], config["image_size"])
    torch.backends.cudnn.benchmark = True     # entrada de tamanho fixo: o cuDNN escolhe o algoritmo mais rápido
    micro = micro_batch(config, device)
    accumulate = config["batch_size"] // micro
    print(f"lote efetivo {config['batch_size']} = {accumulate} x {micro} imagens"
          + (f" | GPU: {torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""), flush=True)
    train_loader = DataLoader(FundusDataset(train, root, transforms(config, True), cached), batch_size=micro,
                              shuffle=True, num_workers=workers, pin_memory=device.type == "cuda", drop_last=True,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(FundusDataset(val, root, transforms(config, False), cached), batch_size=micro,
                            num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)

    model = build_model(config["model"], config.get("pretrained", True),
                        freeze_first_convs=config.get("freeze_first_convs", 0),
                        head_init=config.get("head_init")).to(device).to(memory_format=torch.channels_last)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"parâmetros treináveis: {trainable}; congelados: {frozen}", flush=True)
    if on_model is not None:
        on_model(model)
    positives = float(train["label"].sum())
    pos_weight = (len(train) - positives) / positives if config.get("balance") == "pos_weight" else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = build_optimizer(model, config)
    epochs, steps_per_epoch = config["epochs"], len(train_loader) // accumulate
    warmup = steps_per_epoch * config.get("warmup_epochs", 1)
    total = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 *
                                                  (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    start, best, stale, history = 0, -1.0, 0, []
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, stale, history = state["epoch"] + 1, state["best"], state["stale"], state["history"]
        print(f"retomando da época {start}")

    for epoch in range(start, epochs):
        model.train()
        began, losses, waiting = time.time(), [], 0.0
        optimizer.zero_grad(set_to_none=True)
        step = 0
        asked = time.time()
        for k, (images, labels) in enumerate(train_loader):
            waiting += time.time() - asked    # tempo parado esperando o DataLoader (gargalo de CPU/disco)
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(images).squeeze(1).float(), labels)
            scaler.scale(loss / accumulate).backward()
            losses.append(loss.item())
            asked = time.time()
            if (k + 1) % accumulate:
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if config.get("max_steps") and step >= config["max_steps"]:
                break
            asked = time.time()
        train_seconds = time.time() - began
        probs = predict(model, val_loader, device)
        val_auroc = auroc(val["label"], probs)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auroc": val_auroc,
               "seconds": round(time.time() - began, 1), "train_seconds": round(train_seconds, 1),
               "data_wait_seconds": round(waiting, 1), "lr": scheduler.get_last_lr()[0]}
        if config.get("head_lr") is not None:
            row["head_lr"] = scheduler.get_last_lr()[-1]
        history.append(row)
        print(json.dumps(row), flush=True)
        if log is not None:
            log({"epoca": epoch, "perda_treino": row["train_loss"], "auroc_val": val_auroc, "lr": row["lr"],
                 "segundos_epoca": row["seconds"], "segundos_esperando_dados": row["data_wait_seconds"],
                 **({"lr_saida": row["head_lr"]} if "head_lr" in row else {})})
        if val_auroc is not None and val_auroc > best:
            best, stale = val_auroc, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "stale": stale, "history": history},
                   out / "last.pt")
        if stale >= config.get("patience", 1000):
            print(f"parada antecipada: {stale} épocas sem melhorar")
            break

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    predictions = val[PREDICTION_COLUMNS].assign(prob=predict(model, val_loader, device))
    predictions.to_csv(out / "val_predictions.csv", index=False)
    for name, frame in (("test", test), ("extra", extra)):
        if frame is not None and len(frame):
            loader = DataLoader(FundusDataset(frame.assign(label=frame["label"].clip(lower=0)), root,
                                              transforms(config, False), cached),
                                batch_size=micro, num_workers=workers, pin_memory=device.type == "cuda")
            frame[PREDICTION_COLUMNS].assign(prob=predict(model, loader, device)).to_csv(
                out / f"{name}_predictions.csv", index=False)
    metrics = evaluate(predictions, train)
    metrics.update(trainable_parameters=trainable, frozen_parameters=frozen,
                   micro_batch=micro, gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                   best_epoch=int(np.nanargmax([h["val_auroc"] or -1 for h in history])),
                   epochs_run=len(history), seconds_per_epoch=float(np.mean([h["seconds"] for h in history])),
                   peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    # Pesos ocupam muito (~400 MB por tarefa na ResNet-50): last.pt só serve para retomar, e best.pt só
    # é preciso no treino final (vai para o Hugging Face). As previsões já estão salvas.
    (out / "last.pt").unlink(missing_ok=True)
    if inner is not None and not config.get("keep_weights"):
        (out / "best.pt").unlink(missing_ok=True)
    print(json.dumps({k: metrics[k] for k in ("auroc", "auroc_ic95", "auprc", "regua_origem_auroc")}, indent=1))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--outer", type=int, default=0)
    ap.add_argument("--inner", type=int, default=None, help="vazio = treino final da dobra externa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="sobrescreve chaves da configuração, ex.: lr=1e-4")
    ap.add_argument("--out", type=Path, default=Path("outputs/runs"))
    args = ap.parse_args()
    config = load_config(args.config, args.set)
    tag = f"outer{args.outer}_" + (f"inner{args.inner}" if args.inner is not None else "final") + f"_seed{args.seed}"
    run(config, args.outer, args.inner, args.seed, args.out / config["name"] / tag)


if __name__ == "__main__":
    main()
