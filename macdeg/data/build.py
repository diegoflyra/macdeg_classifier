"""Monta o conjunto único: lê as bases, remove duplicatas e grava as imagens reduzidas.

Uso: python -m macdeg.data.build [--sources hyamd brset ...] [--out ~/macdeg_data/dataset_1024] [--size 1024]

Em --out:
  manifest.csv   uma linha por imagem mantida (rótulo 0 ou 1)
  dropped.csv    imagens descartadas e o motivo
  summary.json   contagens agregadas, sem identificar imagens
  images/        retina inteira num quadrado de lado --size (bordas pretas completam), JPEG;
                 o nome é o hash do arquivo original
  signatures/    assinaturas de detalhes usadas para confirmar cópias

Duplicatas:
- exatas (mesmo conteúdo do arquivo): fica uma cópia; se os rótulos divergem, saem todas;
- cópias entre bases diferentes (a mesma foto republicada, redimensionada, recomprimida ou com as
  bordas cortadas): fica a da base listada primeiro no registro; com rótulos divergentes, saem as duas;
- cópias dentro da mesma base: ficam todas, no mesmo grupo, para nunca caírem uma no treino e
  outra no teste. Fotos diferentes do mesmo olho não são cópias: só ficam juntas se a base
  informar o paciente.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from macdeg.data.select import OTHER_DETAIL, select_other_diseases
from macdeg.data.sources import load_registry, read_source, resolve

# Regra de cópia: o hash (até CANDIDATE_BITS de diferença) só aponta candidatos; a cópia é
# confirmada pela correlação dos detalhes finos (vasos), de pelo menos DETAIL_CORR_MIN.
# Calibração (docs/decisoes.md): na HYAMD nenhum par de pacientes diferentes passa de 0,93; na
# ODIR, entre 57.144 pares candidatos de pacientes diferentes, a correlação forma uma cauda
# contínua até ~0,97, e só 2 pares chegam a 1,0 (cópias reais com IDs diferentes). Com 0,95:
# pega quase todas as cópias reduzidas à metade ou cortadas até o círculo, menos da metade das
# reduzidas a 1/4 e poucas das que tiveram brilho ou contraste alterados.
CANDIDATE_BITS = 24
DETAIL_CORR_MIN = 0.95
JPEG_QUALITY = 95


def fov_square(image: np.ndarray, size: int) -> np.ndarray:
    """Recorta até o círculo da retina, completa com preto até virar quadrado e reduz para `size`.
    Nada da retina é cortado: na BRSET, por exemplo, o círculo é 25% mais largo que a altura da
    foto, e um quadrado central perderia as bordas laterais."""
    crop = fov_crop(image)
    height, width = crop.shape[:2]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    square = cv2.copyMakeBorder(crop, top, side - height - top, left, side - width - left,
                                cv2.BORDER_CONSTANT, value=0)
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(square, (size, size), interpolation=interpolation)


def fov_crop(image: np.ndarray) -> np.ndarray:
    """Recorta até o círculo da retina, para que cópias com bordas pretas cortadas se alinhem."""
    gray = image.max(axis=2) if image.ndim == 3 else image
    scale = min(1.0, 512 / max(gray.shape))  # mesma precisão relativa em qualquer tamanho
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    bright = small > 20
    rows = np.nonzero(bright.mean(1) > 0.02)[0]
    cols = np.nonzero(bright.mean(0) > 0.02)[0]
    if len(rows) < 2 or len(cols) < 2:
        return image
    y0, y1 = int(rows[0] / scale), int(np.ceil((rows[-1] + 1) / scale))
    x0, x1 = int(cols[0] / scale), int(np.ceil((cols[-1] + 1) / scale))
    return image[y0:y1, x0:x1]


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def dhash(image: np.ndarray) -> np.ndarray:
    """Hash de 256 bits (4 inteiros de 64 bits) das formas grandes, dentro do círculo da retina."""
    gray = cv2.resize(_gray(fov_crop(image)), (128, 128), interpolation=cv2.INTER_AREA)
    small = cv2.resize(cv2.GaussianBlur(gray, (0, 0), 2), (17, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits((small[:, 1:] > small[:, :-1]).reshape(-1)).view(">u8").astype(np.uint64)


def detail_signature(image: np.ndarray) -> np.ndarray:
    """Detalhes finos (vasos), normalizados: a correlação entre duas assinaturas confirma a cópia."""
    gray = cv2.resize(_gray(fov_crop(image)), (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    detail = (cv2.GaussianBlur(gray, (0, 0), 1) - cv2.GaussianBlur(gray, (0, 0), 6))[16:-16, 16:-16]
    detail -= detail.mean()
    detail /= detail.std() + 1e-6
    return detail.reshape(-1).astype(np.float16)


def process_image(job: tuple[str, int, str]) -> dict:
    path, size, out = job
    try:
        data = Path(path).read_bytes()
    except OSError:
        return {"error": "ilegivel"}
    digest = hashlib.sha1(data).hexdigest()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"sha1": digest, "error": "ilegivel"}
    uid = digest[:20]
    target = Path(out) / "images" / f"{uid}.jpg"
    if not target.exists():
        cv2.imwrite(str(target), fov_square(image, size), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    signature = Path(out) / "signatures" / f"{uid}.npy"
    if not signature.exists():
        np.save(signature, detail_signature(image))
    return {"sha1": digest, "uid": uid, "height": image.shape[0], "width": image.shape[1],
            **{f"h{i}": int(w) for i, w in enumerate(dhash(image))}, "error": None}


def is_copy(distance: int, load_signature, uid_a: str, uid_b: str) -> bool:
    if distance > CANDIDATE_BITS:
        return False
    a = load_signature(uid_a).astype(np.float32)
    b = load_signature(uid_b).astype(np.float32)
    return float(a @ b / a.size) >= DETAIL_CORR_MIN


def hamming_pairs(hashes: np.ndarray, max_bits: int, chunk: int = 64) -> list[tuple[int, int, int]]:
    """Pares (i, j, distância) com i < j e distância <= max_bits."""
    pairs = []
    for start in range(0, len(hashes), chunk):
        block = hashes[start:start + chunk]
        distance = np.bitwise_count(block[:, None, :] ^ hashes[None, :, :]).sum(-1)
        rows, cols = np.nonzero(distance <= max_bits)
        for r, c in zip(rows, cols):
            i = start + r
            if c > i:
                pairs.append((int(i), int(c), int(distance[r, c])))
    return pairs


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def deduplicate(table: pd.DataFrame, load_signature) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Recebe as imagens legíveis, na ordem do registro. Devolve (mantidas, descartadas, estatísticas).
    `load_signature(uid)` devolve a assinatura de detalhes da imagem."""
    table = table.reset_index(drop=True)
    table["_pos"] = np.arange(len(table))
    dropped = []

    # Duplicatas exatas: prefere a cópia com rótulo definido e, depois, a da base listada primeiro.
    keep = np.ones(len(table), bool)
    for _, group in table.groupby("sha1", sort=False):
        if len(group) == 1:
            continue
        labels = set(group["label"].dropna())
        if len(labels) > 1:
            keep[group.index] = False
            dropped += [(i, "rotulo_conflitante_entre_copias") for i in group.index]
            continue
        best = group.assign(_unlabeled=group["label"].isna()).sort_values(["_unlabeled", "order"]).index[0]
        for i in group.index:
            if i != best:
                keep[i] = False
                dropped.append((i, "duplicata_exata"))

    # Fora do conjunto: outras doenças ou rótulos ambíguos.
    for i in np.nonzero(keep & table["label"].isna().to_numpy())[0]:
        keep[i] = False
        dropped.append((int(i), "fora_do_conjunto"))

    kept = table[keep].reset_index(drop=True)
    hashes = kept[[f"h{i}" for i in range(4)]].to_numpy(np.uint64)
    uids = kept["uid"].to_numpy()
    pairs = [(i, j, d) for i, j, d in hamming_pairs(hashes, CANDIDATE_BITS)
             if is_copy(d, load_signature, uids[i], uids[j])]
    sources = kept["source"].to_numpy()
    cross = [(i, j, d) for i, j, d in pairs if sources[i] != sources[j]]
    within = [(i, j, d) for i, j, d in pairs if sources[i] == sources[j]]

    # Quase iguais entre bases: fica a de menor ordem. Rótulos divergentes: saem as duas.
    remove: dict[int, str] = {}
    for i, j, _ in cross:
        if kept.at[i, "label"] != kept.at[j, "label"]:
            remove[i] = remove[j] = "rotulo_conflitante_entre_copias"
        else:
            later = j if kept.at[j, "order"] > kept.at[i, "order"] else i
            remove.setdefault(later, "quase_duplicata_entre_bases")
    index_map = kept["_pos"].to_numpy()
    dropped += [(int(index_map[i]), reason) for i, reason in remove.items()]

    # Grupos para a divisão: mesmo paciente ou fotos quase iguais da mesma base.
    finder = UnionFind(len(kept))
    for i, j, _ in within:
        finder.union(i, j)
    first_of_patient: dict[str, int] = {}
    for i, patient in enumerate(kept["patient_id"]):
        if isinstance(patient, str):
            finder.union(i, first_of_patient.setdefault(patient, i))
    kept["group_id"] = [f"g{finder.find(i)}" for i in range(len(kept))]
    kept = kept.drop(index=list(remove)).reset_index(drop=True)

    sizes = kept["group_id"].value_counts()
    stats = {"pares_copia_mesma_base": len(within), "pares_copia_entre_bases": len(cross),
             "maior_grupo": int(sizes.max()) if len(sizes) else 0, "grupos": int(len(sizes))}
    gone = pd.DataFrame(dropped, columns=["_pos", "reason"])
    return kept, table.loc[gone["_pos"]].assign(reason=gone["reason"].to_numpy()), stats


def summarize(manifest: pd.DataFrame, dropped: pd.DataFrame, raw: pd.DataFrame, stats: dict) -> dict:
    by_source = {}
    for source, group in manifest.groupby("source"):
        by_source[source] = {
            "dmri": int((group["label"] == 1).sum()), "normal": int(((group["label"] == 0) & (group["label_detail"] != OTHER_DETAIL)).sum()),
            "outra_doenca": int((group["label_detail"] == OTHER_DETAIL).sum()),
            "com_paciente": round(float(group["patient_id"].notna().mean()), 3),
            "pacientes": int(group["patient_id"].nunique()), "grupos": int(group["group_id"].nunique())}
    outside = (dropped[dropped["reason"] == "fora_do_conjunto"].groupby(["source", "label_detail"]).size()
               .rename("n").reset_index().to_dict("records"))
    return {"total": int(len(manifest)), "dmri": int((manifest["label"] == 1).sum()),
            "sem_dmri": int((manifest["label"] == 0).sum()), "normal": int(((manifest["label"] == 0) & (manifest["label_detail"] != OTHER_DETAIL)).sum()),
            "outra_doenca": int((manifest["label_detail"] == OTHER_DETAIL).sum()), "por_base": by_source,
            "lidas_por_base": raw.groupby("source").size().astype(int).to_dict(),
            "descartadas": dropped.groupby(["source", "reason"]).size().rename("n").reset_index().to_dict("records"),
            "fora_do_conjunto_por_rotulo": outside, "duplicatas": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--sources", nargs="*", default=None, help="padrão: todas as do registro")
    ap.add_argument("--out", default="~/macdeg_data/dataset_1024")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    registry = load_registry(args.registry) if args.registry else load_registry()
    summary = build(registry, args.sources or list(registry["sources"]), resolve(args.out), args.size, args.workers)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def build(registry: dict, names: list[str], out: Path, size: int = 512, workers: int = 6) -> dict:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "signatures").mkdir(parents=True, exist_ok=True)

    raw = pd.concat([read_source(n, registry["sources"][n]) for n in names], ignore_index=True)
    raw["order"] = raw["source"].map({n: k for k, n in enumerate(names)})
    raw["_row"] = np.arange(len(raw))
    print(f"lidas: {len(raw)} imagens de {len(names)} base(s)")

    missing = ~raw["path"].map(lambda p: Path(p).is_file())
    jobs = [(p, size, str(out)) for p in raw.loc[~missing, "path"]]
    results = []
    with ProcessPoolExecutor(workers) as pool:
        for k, result in enumerate(pool.map(process_image, jobs, chunksize=16), 1):
            results.append(result)
            if k % 500 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} imagens processadas", flush=True)
    found = raw[~missing].reset_index(drop=True).join(pd.DataFrame(results))
    unreadable = found[found["error"].notna()]
    readable = found[found["error"].isna()]

    # O sorteio vem antes das cópias: uma outra doença que seja cópia de uma DMRI de outra base cai
    # como rótulo conflitante.
    readable, selection = select_other_diseases(readable, registry.get("selection"))
    load = functools.lru_cache(maxsize=None)(lambda uid: np.load(out / "signatures" / f"{uid}.npy"))
    kept, dropped, stats = deduplicate(readable, load)
    dropped = pd.concat([raw[missing].assign(reason="arquivo_ausente"), unreadable.assign(reason="ilegivel"),
                         dropped], ignore_index=True)
    kept["image"] = "images/" + kept["uid"] + ".jpg"
    kept["label"] = kept["label"].astype(int)
    # ID sequencial nosso: é o que sai do cluster (previsões no GitHub). O uid (hash do arquivo original)
    # permitiria a quem tem a base achar a imagem; a ligação entre os dois fica só no manifest.
    kept = kept.sort_values("uid").reset_index(drop=True)
    kept["sample_id"] = [f"img_{i:05d}" for i in range(len(kept))]
    # Drusas sem diagnóstico de DMRI: fora do treino, mas o modelo dá nota a elas (distribuição das notas).
    drusen = dropped[(dropped["reason"] == "fora_do_conjunto")
                     & dropped["label_detail"].eq("drusas_sem_diagnostico_de_dmri") & dropped["uid"].notna()]
    drusen = drusen[~drusen["uid"].isin(kept["uid"])].drop_duplicates("uid").sort_values("uid").reset_index(drop=True)
    drusen = drusen.assign(sample_id=[f"dru_{i:05d}" for i in range(len(drusen))],
                           image="images/" + drusen["uid"] + ".jpg", group_id="dru_" + drusen["uid"])
    drusen[["uid", "sample_id", "source", "group_id", "image"]].to_csv(out / "extras_drusas.csv", index=False)
    columns = ["uid", "sample_id", "source", "label", "label_detail", "disease", "disease_group", "patient_id", "eye", "age", "sex", "camera", "group_id",
               "width", "height", "image", "path"]
    kept[columns].to_csv(out / "manifest.csv", index=False)
    dropped[["source", "path", "label_detail", "reason"]].to_csv(out / "dropped.csv", index=False)
    summary = summarize(kept, dropped, raw, stats)
    summary["sorteio_outras_doencas"] = selection
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
