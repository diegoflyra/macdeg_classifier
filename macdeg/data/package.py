"""Empacota o conjunto para o cluster: imagens reduzidas, manifest, divisões e drusas.

Uso: python -m macdeg.data.package [--dataset ~/macdeg_data/dataset_1024] [--out ~/macdeg_data/dataset_1024.zip]

As imagens já são JPEG: o zip só as guarda (sem recomprimir), o que é rápido. Não entram as assinaturas
de cópias nem o dropped.csv (que tem caminhos dos originais). Imprime o SHA-256 para conferir a cópia
no cluster (sha256sum dataset_1024.zip).
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

from macdeg.data.sources import resolve

FILES = ["manifest.csv", "splits_nested_seed0.csv", "extras_drusas.csv", "summary.json"]


def package(dataset: Path, out: Path) -> str:
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in FILES:
            archive.write(dataset / name, name)
        # Só as imagens usadas: as do manifest e as drusas sem diagnóstico (que o modelo pontua)
        import pandas as pd
        used = set(pd.read_csv(dataset / "manifest.csv", usecols=["image"])["image"])
        used |= set(pd.read_csv(dataset / "extras_drusas.csv", usecols=["image"])["image"])
        for image in sorted(used):
            archive.write(dataset / image, image)
    digest = hashlib.sha256()
    with open(out, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="~/macdeg_data/dataset_1024")
    ap.add_argument("--out", default="~/macdeg_data/dataset_1024.zip")
    args = ap.parse_args()
    out = resolve(args.out)
    print("sha256:", package(resolve(args.dataset), out), "|", f"{out.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
