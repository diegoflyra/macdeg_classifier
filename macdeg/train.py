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

from macdeg import cache as image_cache
from macdeg.convnext import CONVNEXT_MODELS
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

    def __init__(self, frame: pd.DataFrame, root: Path, transform, cache: tuple[Path, dict] | None = None,
                 *, aux_dr: bool = False):
        self.paths = [str(root / p) for p in frame["image"]]
        self.labels = frame["label"].to_numpy(np.float32)
        self.dr_labels = dr_target(frame) if aux_dr else None
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
        sample = (self.transform(tensor), self.labels[index])
        if self.dr_labels is not None:
            return (*sample, self.dr_labels[index])
        return sample


def dr_target(frame: pd.DataFrame) -> np.ndarray:
    """Rótulo auxiliar da rodada 002. Nos olhos com DMRI o estado da DR é desconhecido."""
    if "disease" not in frame:
        raise ValueError("aux_dr requer a coluna disease no manifesto.")
    without_dmri = frame["label"].to_numpy() == 0
    diseases = frame.loc[without_dmri, "disease"].fillna("").astype(str).str.split("|")
    target = np.full(len(frame), -1.0, np.float32)
    target[without_dmri] = diseases.map(lambda names: "retinopatia_diabetica" in names).to_numpy(np.float32)
    return target


def dr_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE média dos alvos conhecidos no microbatch. Vale zero quando o microbatch não tem nenhum."""
    known = target >= 0
    masked_logits = logits.masked_fill(~known, 0)
    loss = nn.functional.binary_cross_entropy_with_logits(
        masked_logits, target.clamp_min(0), weight=known.to(logits.dtype), reduction="sum")
    return loss / known.sum().clamp_min(1)


def aux_dr_weight(config: dict) -> float:
    weight = config.get("aux_dr")
    if weight is None:
        return 0.0
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("aux_dr deve ser null ou um peso finito e não negativo.")
    return float(weight)


def outputs_for(config: dict) -> int:
    return 2 if aux_dr_weight(config) > 0 else 1


def transforms(config: dict, train: bool):
    from torchvision.transforms import v2

    size, aug = config["image_size"], config.get("augment") or {}
    steps = []
    if train and aug:  # augment vazio ou null = sem augmentation: só redimensiona
        steps.append(v2.RandomResizedCrop(size, scale=tuple(aug.get("crop_scale", (0.85, 1.0))), ratio=(0.95, 1.05),
                                          antialias=True))
        if aug.get("hflip", True):
            steps.append(v2.RandomHorizontalFlip())
        if aug.get("rotate_deg", 0):
            interpolation = aug.get("rotate_interpolation", "nearest")
            if interpolation not in ("nearest", "bilinear"):
                raise ValueError("rotate_interpolation deve ser nearest ou bilinear.")
            steps.append(v2.RandomRotation(aug["rotate_deg"], interpolation=v2.InterpolationMode(interpolation)))
        if aug.get("color", 0):
            steps.append(v2.ColorJitter(brightness=aug["color"], contrast=aug["color"]))
    else:
        steps.append(v2.Resize(size, antialias=True))
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(steps)


def build_model(name: str, pretrained: bool, outputs: int = 1, *, freeze_first_convs: int = 0,
                head_init: str | None = None, freeze_first_stages: int = 0,
                stochastic_depth_prob: float | None = None, gradient_checkpointing: bool = False) -> nn.Module:
    """O logit 0 é o de DMRI. O logit 1, quando habilitado, supervisiona DR."""
    if type(outputs) is not int or outputs not in (1, 2):
        raise ValueError("outputs deve ser 1 ou 2.")
    is_convnext = name in CONVNEXT_MODELS
    if head_init not in (None, "xavier"):
        raise ValueError("head_init deve ser null ou xavier.")
    if head_init is not None and name != "alexnet" and not is_convnext:
        raise ValueError("head_init só é suportado para alexnet e convnext.")
    if type(freeze_first_convs) is not int or not 0 <= freeze_first_convs <= 5:
        raise ValueError("freeze_first_convs deve ser um inteiro entre 0 e 5.")
    if freeze_first_convs and name != "alexnet":
        raise ValueError("freeze_first_convs só é suportado para model: alexnet.")
    if type(freeze_first_stages) is not int or not 0 <= freeze_first_stages <= 4:
        raise ValueError("freeze_first_stages deve ser um inteiro entre 0 e 4.")
    if freeze_first_stages and not is_convnext:
        raise ValueError("freeze_first_stages só é suportado para convnext.")
    if type(gradient_checkpointing) is not bool:
        raise ValueError("gradient_checkpointing deve ser true ou false.")
    if not is_convnext and (gradient_checkpointing or stochastic_depth_prob is not None):
        raise ValueError("gradient_checkpointing e stochastic_depth_prob só são suportados para convnext.")
    if name == "alexnet":
        from macdeg.layers import Linear
        from macdeg.models import AlexNet, freeze_alexnet_features, load_alexnet_imagenet_weights
        model = AlexNet()
        if pretrained:
            load_alexnet_imagenet_weights(model)
        model.classifier[-1] = Linear(model.classifier[-1].in_features, outputs)
        if head_init == "xavier":
            nn.init.xavier_uniform_(model.classifier[-1].weight, gain=1.0)
            nn.init.zeros_(model.classifier[-1].bias)
        freeze_alexnet_features(model, freeze_first_convs)
        return model
    if is_convnext:
        from macdeg.convnext import (freeze_convnext_stages, load_convnext_imagenet_weights,
                                     load_convnext_tiny_imagenet_weights)
        from macdeg.layers import Linear
        model = CONVNEXT_MODELS[name](stochastic_depth_prob=stochastic_depth_prob,
                                      gradient_checkpointing=gradient_checkpointing)
        if pretrained:
            loader = load_convnext_tiny_imagenet_weights if name == "convnext_tiny" else load_convnext_imagenet_weights
            loader(model)
        model.classifier[-1] = Linear(model.classifier[-1].in_features, outputs)
        if head_init == "xavier":
            nn.init.xavier_uniform_(model.classifier[-1].weight, gain=1.0)
            nn.init.zeros_(model.classifier[-1].bias)
        freeze_convnext_stages(model, freeze_first_stages)
        return model
    if name in ("resnet18", "resnet34", "resnet50"):
        import torchvision.models as tvm
        weights = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1, "resnet34": tvm.ResNet34_Weights.IMAGENET1K_V1,
                   "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2}[name] if pretrained else None
        model = getattr(tvm, name)(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, outputs)
        return model
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=outputs)


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Taxa opcional para a saída nova; demais parâmetros mantêm config['lr']."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    head_lr = config.get("head_lr")
    if head_lr is not None:
        name = config.get("model")
        if name != "alexnet" and name not in CONVNEXT_MODELS:
            raise ValueError("head_lr só é suportado para alexnet e convnext.")
        if not math.isfinite(head_lr) or head_lr <= 0:
            raise ValueError("head_lr deve ser positivo e finito.")
        head = [p for p in model.classifier[-1].parameters() if p.requires_grad]
        head_ids = {id(p) for p in head}
        parameters = [{"params": [p for p in parameters if id(p) not in head_ids], "lr": config["lr"]},
                      {"params": head, "lr": head_lr}]
    if config.get("exclude_norm_bias_from_weight_decay", False):
        # Escalas, vieses e LayerScale não recebem penalização nesta opção explícita.
        head_ids = {id(p) for p in model.classifier[-1].parameters()} if head_lr is not None else set()
        groups = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            is_head = id(parameter) in head_ids
            no_decay = parameter.ndim <= 1 or name.endswith(".bias") or name.endswith(".layer_scale")
            key = (is_head, no_decay)
            if key not in groups:
                groups[key] = {"params": [], "lr": head_lr if is_head else config["lr"],
                               "weight_decay": 0.0 if no_decay else config.get("weight_decay", 1e-4)}
            groups[key]["params"].append(parameter)
        # Mantém os grupos da saída por último, para histórico e retomada.
        parameters = [groups[key] for key in sorted(groups)]
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
    if type(batch) is not int or batch <= 0:
        raise ValueError("batch_size deve ser um inteiro positivo.")
    if config.get("micro_batch") is not None:
        micro = config["micro_batch"]
        if type(micro) is not int or micro <= 0 or batch % micro:
            raise ValueError("micro_batch deve ser um inteiro positivo que divide batch_size.")
        return micro
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
    for images, *_ in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last))
        probs.append(torch.sigmoid(logits[:, 0].float()))
    return torch.cat(probs).cpu().numpy()


def run(config: dict, outer: int, inner: int | None, seed: int, out: Path, log=None, on_model=None) -> dict | None:
    """Treina uma dobra. `log(dict)` recebe as métricas de cada época e `on_model(model)` recebe o modelo
    antes do treino (ex.: W&B, para os histogramas de pesos e gradientes)."""
    if (out / "metrics.json").exists():
        print(f"{out}: já concluído, pulando")
        return None
    task_started = time.perf_counter()
    aux_weight = aux_dr_weight(config)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if config.get("torch_num_threads") is not None:
        threads = config["torch_num_threads"]
        if type(threads) is not int or threads <= 0:
            raise ValueError("torch_num_threads deve ser um inteiro positivo.")
        torch.set_num_threads(threads)
    max_grad_norm = config.get("max_grad_norm")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("max_grad_norm deve ser positivo e finito.")
    train, val, test, extra, root = split_frames(config, outer, inner, seed)
    (out / "config.json").write_text(json.dumps({**config, "outer": outer, "inner": inner, "seed": seed}, indent=1),
                                     encoding="utf-8")
    print(f"treino: {len(train)} imagens ({int(train['label'].sum())} DMRI); "
          f"validação: {len(val)} ({int(val['label'].sum())} DMRI); dispositivo: {device}", flush=True)

    workers = config.get("num_workers", 6)
    train_transform = transforms(config, True)
    eval_transform = transforms(config, False)
    cached = None
    if config.get("cache", True):   # mesmos pixels, sem decodificar 1024 px a cada época (macdeg.cache)
        frames = [f for f in (train, val, test, extra) if f is not None]
        cached = image_cache.build(root, [i for f in frames for i in f["image"]], config["image_size"])
    torch.backends.cudnn.benchmark = True     # entrada de tamanho fixo: o cuDNN escolhe o algoritmo mais rápido
    micro = micro_batch(config, device)
    accumulate = config["batch_size"] // micro
    print(f"lote efetivo {config['batch_size']} = {accumulate} x {micro} imagens"
          + (f" | GPU: {torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""), flush=True)
    train_loader = DataLoader(FundusDataset(train, root, train_transform, cached, aux_dr=aux_weight > 0), batch_size=micro,
                              shuffle=True, num_workers=workers, pin_memory=device.type == "cuda", drop_last=True,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(FundusDataset(val, root, eval_transform, cached), batch_size=micro,
                            num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)

    model = build_model(config["model"], config.get("pretrained", True), outputs_for(config),
                        freeze_first_convs=config.get("freeze_first_convs", 0),
                        head_init=config.get("head_init"),
                        freeze_first_stages=config.get("freeze_first_stages", 0),
                        stochastic_depth_prob=config.get("stochastic_depth_prob"),
                        gradient_checkpointing=config.get("gradient_checkpointing", False)).to(device).to(memory_format=torch.channels_last)
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
    if steps_per_epoch < 1:
        raise ValueError("O treino precisa de ao menos batch_size amostras para uma atualização completa.")
    warmup = steps_per_epoch * config.get("warmup_epochs", 1)
    total = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (min(1.0, (s + 1) / warmup) if warmup else 1.0) * 0.5 *
                                                  (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    start, best, stale, history = 0, -1.0, 0, []
    if (out / "last.pt").exists():
        # Não mantém uma segunda cópia inteira do checkpoint na VRAM ao retomar.
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, stale, history = state["epoch"] + 1, state["best"], state["stale"], state["history"]
        del state
        print(f"retomando da época {start}")

    for epoch in range(start, epochs):
        model.train()
        began, losses, waiting = time.time(), [], 0.0
        optimizer.zero_grad(set_to_none=True)
        step = 0
        asked = time.time()
        for k, (images, labels, *auxiliary) in enumerate(train_loader):
            waiting += time.time() - asked    # tempo parado esperando o DataLoader (gargalo de CPU/disco)
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images).float()
                loss = criterion(logits[:, 0], labels)
                if aux_weight:
                    targets = auxiliary[0].to(device, non_blocking=True)
                    loss = loss + aux_weight * dr_loss(logits[:, 1], targets)
            scaler.scale(loss / accumulate).backward()
            losses.append(loss.detach())  # transfere as perdas juntas no fim da época
            asked = time.time()
            if (k + 1) % accumulate:
                continue
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if config.get("max_steps") and step >= config["max_steps"]:
                break
            asked = time.time()
        train_loss = float(np.mean(torch.stack(losses).tolist()))
        train_seconds = time.time() - began
        probs = predict(model, val_loader, device)
        val_auroc = auroc(val["label"], probs)
        row = {"epoch": epoch, "train_loss": train_loss, "val_auroc": val_auroc,
               "optimizer_steps": step,
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
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=True))
    predictions = val[PREDICTION_COLUMNS].assign(prob=predict(model, val_loader, device))
    predictions.to_csv(out / "val_predictions.csv", index=False)
    for name, frame in (("test", test), ("extra", extra)):
        if frame is not None and len(frame):
            loader = DataLoader(FundusDataset(frame.assign(label=frame["label"].clip(lower=0)), root,
                                              eval_transform, cached),
                                batch_size=micro, num_workers=workers, pin_memory=device.type == "cuda")
            frame[PREDICTION_COLUMNS].assign(prob=predict(model, loader, device)).to_csv(
                out / f"{name}_predictions.csv", index=False)
    metrics = evaluate(predictions, train)
    metrics.update(trainable_parameters=trainable, frozen_parameters=frozen,
                   gradient_checkpointing=config.get("gradient_checkpointing", False),
                   effective_batch_size=config["batch_size"],
                   micro_batch=micro, gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                   best_epoch=int(np.nanargmax([h["val_auroc"] or -1 for h in history])),
                   epochs_run=len(history), seconds_per_epoch=float(np.mean([h["seconds"] for h in history])),
                   task_seconds=round(time.perf_counter() - task_started, 1),
                   peak_gpu_gb=None, peak_reserved_gpu_gb=None, gpu_total_gb=None)
    if device.type == "cuda":
        metrics.update(peak_gpu_gb=round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
                       peak_reserved_gpu_gb=round(torch.cuda.max_memory_reserved(device) / 1e9, 2),
                       gpu_total_gb=round(torch.cuda.get_device_properties(device).total_memory / 1e9, 2))
    # A fila compartilhada só vê a tarefa como concluída quando o JSON está inteiro.
    metrics_tmp = out / "metrics.tmp"
    metrics_tmp.write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    metrics_tmp.replace(out / "metrics.json")
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
