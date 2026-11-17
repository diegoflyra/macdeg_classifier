"""Cache das imagens já reduzidas ao tamanho do treino, para a CPU não virar o gargalo.

Sem o cache, cada época decodifica cada JPEG de 1024 px e o reduz para 512 px na CPU. Isso é ~3x mais
lento do que a GPU treina (referência na A100: 117 s por época, quando a GPU sozinha faria ~35 s).
Com o cache, a redução é feita uma vez por job e guardada num arquivo .npy (uint8, N x S x S x 3)
que os DataLoaders leem por mmap: ler uma imagem vira copiar 0,8 MB da memória.

Os pixels são idênticos aos da leitura direta (mesma decodificação e o mesmo v2.Resize com antialias;
o Resize do transform vira identidade numa imagem que já tem o tamanho certo): os resultados não mudam.
Fica em $MACDEG_CACHE ou em <conjunto>/cache (no cluster, o disco local do nó, apagado no fim do job).
Imagens fora do cache (ou não quadradas) são lidas do JPEG, como antes.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import v2


# Um thread do OpenCV por processo: quem paraleliza são os processos do DataLoader. Sem isso, cada
# processo abre um thread por núcleo do nó e eles disputam a CPU.
cv2.setNumThreads(1)


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError("imagem ilegível no conjunto reduzido")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize(image: np.ndarray, size: int) -> np.ndarray:
    """O mesmo redimensionamento do transform de avaliação (v2.Resize com antialias), em HWC uint8."""
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    return v2.functional.resize(tensor, size, antialias=True).permute(1, 2, 0).contiguous().numpy()


PREPROCESS = (None, "clahe")


def clahe(image: np.ndarray) -> np.ndarray:
    """Equalização local de contraste (CLAHE) só na luminância (L do LAB), cores preservadas. Uniformiza
    iluminação e contraste entre câmeras. Grade de 8 x 8 blocos e limite 2,0, os valores usuais em retina."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def center_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    """Quadrado central com `fraction` do lado. As fotos são centradas na mácula: com 0,6, a fóvea fica
    dentro do recorte em ~100% das imagens (fóvea marcada no ADAM; detector clássico nas outras bases)."""
    if not 0 < fraction <= 1:
        raise ValueError(f"crop deve estar em (0, 1]: {fraction!r}")
    height, width = image.shape[:2]
    side = int(round(min(height, width) * fraction))
    top, left = (height - side) // 2, (width - side) // 2
    return image[top:top + side, left:left + side]


def prepare(image: np.ndarray, size: int, preprocess: str | None = None, crop: float | None = None) -> np.ndarray:
    """Imagem RGB original -> recorte central (se houver) -> tamanho do treino -> pré-processamento (se houver)."""
    if preprocess not in PREPROCESS:
        raise ValueError(f"preprocess desconhecido: {preprocess!r} (opções: {PREPROCESS})")
    if crop:
        image = center_crop(image, crop)
    small = resize(image, size)
    return clahe(small) if preprocess == "clahe" else small


def folder_for(root: Path) -> Path:
    return Path(os.environ.get("MACDEG_CACHE") or Path(root) / "cache").expanduser()


@contextmanager
def _lock(path: Path):
    with open(path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)     # dois processos (duas GPUs) não constroem o mesmo cache
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _fill(args) -> list[int]:
    """Processo auxiliar: reduz um pedaço das imagens e grava direto no .npy (mmap)."""
    data_path, root, size, preprocess, crop, chunk = args
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    data = np.load(data_path, mmap_mode="r+")
    done = []
    for row, image in chunk:
        small = prepare(read_rgb(str(Path(root) / image)), size, preprocess, crop)
        if small.shape == (size, size, 3):
            data[row] = small
            done.append(row)
    data.flush()
    return done


def build(root: Path, images: list[str], size: int, workers: int | None = None,
          preprocess: str | None = None, crop: float | None = None) -> tuple[Path, dict[str, int]] | None:
    """Garante o cache de `images` (caminhos relativos ao conjunto) em `size` px. Devolve (arquivo .npy,
    {imagem: linha}). Se já existe um cache com todas essas imagens, só o reaproveita. Se o disco não
    tem espaço, devolve None: o treino lê os JPEGs (mais devagar, mesmos pixels)."""
    folder = folder_for(root)
    folder.mkdir(parents=True, exist_ok=True)
    key = (f"{size}px" + (f"-{preprocess}" if preprocess else "")    # um cache por tamanho, recorte e
           + (f"-crop{crop:g}" if crop else ""))                       # pré-processamento
    index_path = folder / f"{key}.json"
    wanted = sorted(set(images))
    with _lock(folder / f"{key}.lock"):
        if index_path.exists():
            index = json.loads(index_path.read_text())
            known = set(index["pedidas"])
            missing = sum(image not in known for image in wanted)
            if missing <= 0.1 * len(wanted):     # poucas faltando (ex.: modo de teste): essas vêm do JPEG
                return folder / index["arquivo"], index["linhas"]
            wanted = sorted(set(wanted) | known)    # amplia: refaz com a união
        # Nome novo a cada conteúdo: quem já lê o arquivo anterior (a outra GPU) não é afetado.
        need = len(wanted) * size * size * 3
        free = shutil.disk_usage(folder).free
        if free < need * 1.05 + 2e9:     # folga de 2 GB para o resto do job
            print(f"cache: AVISO, sem espaço em {folder} ({free / 1e9:.1f} GB livres, precisa de "
                  f"{need / 1e9:.1f} GB): o treino vai ler os JPEGs, mais devagar", flush=True)
            return None
        digest = hashlib.sha1("\n".join(wanted).encode()).hexdigest()[:10]
        data_path = folder / f"{key}-{digest}.npy"
        # Processos auxiliares só compensam com muitas imagens (cada um importa o PyTorch)
        workers = workers or (max(1, len(os.sched_getaffinity(0))) if len(wanted) >= 1000 else 1)
        print(f"cache: reduzindo {len(wanted)} imagens para {size} px com {workers} processos...", flush=True)
        tmp = data_path.with_suffix(".tmp.npy")
        np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(wanted), size, size, 3)).flush()
        rows = list(enumerate(wanted))
        chunks = [rows[i::workers * 4] for i in range(min(len(rows), workers * 4))]
        if workers == 1:
            done = [r for c in chunks for r in _fill((tmp, root, size, preprocess, crop, c))]
        else:
            import multiprocessing as mp      # fork: não reimporta o PyTorch (caro em disco de rede)
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
                done = [r for part in pool.map(_fill, [(tmp, root, size, preprocess, crop, c) for c in chunks]) for r in part]
        tmp.replace(data_path)
        lines = {wanted[r]: r for r in sorted(done)}
        index_path.write_text(json.dumps({"tamanho": size, "arquivo": data_path.name, "pedidas": wanted,
                                          "linhas": lines}))
        print(f"cache: pronto ({len(lines)} imagens, {data_path.stat().st_size / 1e9:.1f} GB)", flush=True)
        return data_path, lines
