"""Divisões da validação cruzada aninhada: 5 dobras externas e, dentro de cada uma, 4 internas.

Uso: python -m macdeg.data.splits [--dataset ~/macdeg_data/dataset_1024] [--seed 0]

- Dobra externa k: as imagens com outer == k são o teste da dobra k; nunca participam de nenhuma
  escolha (época, limiar, configuração) feita para ela.
- Dobra interna j da externa k (coluna inner_k): divide as outras 4 partes; treina em 3 e valida
  em 1. As previsões das 4 internas escolhem a época, o limiar e a configuração da dobra k.
- Tudo por grupo (paciente, ou fotos quase iguais quando a base não informa o paciente) e com a
  proporção de DMRI, normal e outra doença mantida dentro de cada base.
Grava splits_nested_seed<seed>.csv (uid, outer, inner_0..inner_4; -1 = imagem do teste dessa
dobra) e imprime só contagens.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from macdeg.data.sources import resolve

N_OUTER, N_INNER = 5, 4


def kind(manifest: pd.DataFrame) -> pd.Series:
    """dmri, normal ou outra (outra doença)."""
    return np.where(manifest["label"] == 1, "dmri",
                    np.where(manifest["label_detail"].eq("outra_doenca"), "outra", "normal"))


def assign_nested(manifest: pd.DataFrame, seed: int = 0, n_outer: int = N_OUTER, n_inner: int = N_INNER) -> pd.DataFrame:
    manifest = manifest.reset_index(drop=True)
    strata = pd.Series(manifest["source"] + "_" + kind(manifest), index=manifest.index)
    groups = manifest["group_id"]
    result = pd.DataFrame({"uid": manifest["uid"], "outer": -1})
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    for k, (rest, test) in enumerate(outer.split(manifest, strata, groups)):
        result.loc[test, "outer"] = k
        column = np.full(len(manifest), -1)
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1 + k)
        for j, (_, val) in enumerate(inner.split(rest, strata.iloc[rest], groups.iloc[rest])):
            column[rest[val]] = j
        result[f"inner_{k}"] = column
    return result


def describe(manifest: pd.DataFrame, splits: pd.DataFrame) -> dict:
    table = manifest.assign(kind=kind(manifest)).merge(splits, on="uid")
    counts = table.groupby(["outer", "source", "kind"]).size().unstack("kind", fill_value=0)
    return {f"externa{outer}/{source}": {k: int(v) for k, v in row.items()} for (outer, source), row in counts.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="~/macdeg_data/dataset_1024")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dataset = resolve(args.dataset)
    manifest = pd.read_csv(dataset / "manifest.csv", dtype={"patient_id": str})
    splits = assign_nested(manifest, args.seed)
    splits.to_csv(dataset / f"splits_nested_seed{args.seed}.csv", index=False)
    print(json.dumps(describe(manifest, splits), indent=1))


if __name__ == "__main__":
    main()
