"""Dobras das imagens com drusas sem diagnóstico de DMRI, para a saída auxiliar de drusas (`aux_drusas`).

Uso (no computador com as bases originais; depois copie o arquivo para o conjunto no cluster):
    python -m macdeg.data.drusas [--out ~/macdeg_data/dataset_1024]

Grava `drusas_dobras.csv` ao lado do `manifest.csv`: uid, sample_id, source, group_id, age, outer, inner_0..4.
Não muda o manifest nem as divisões: a assinatura dos dados continua a mesma.

Por que é preciso: o `extras_drusas.csv` põe cada imagem num grupo sozinha, sem o paciente. Isso bastava
enquanto as drusas só eram pontuadas; para treinar com elas, as imagens de um paciente não podem ficar no
treino de uma dobra e no teste da mesma dobra. Regras:
- paciente que também está no conjunto principal: herda o grupo e as dobras dele (ninguém atravessa dobras);
- paciente só com drusas: grupo novo (`dp_00001`), dobra externa e internas sorteadas (semente fixa);
- base sem ID de paciente (RFMiD): cada imagem é um grupo, como no conjunto principal.
A idade vem da planilha da base e só é usada na análise (notas por faixa etária), nunca no treino.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

DETAIL = "drusas_sem_diagnostico_de_dmri"
FOLDS_FILE = "drusas_dobras.csv"


def _family(source: pd.Series) -> pd.Series:
    return source.astype(str).str.replace("rfmid2", "rfmid", regex=False)


def assign_folds(extras: pd.DataFrame, links: pd.DataFrame, manifest: pd.DataFrame, splits: pd.DataFrame,
                 seed: int = 0) -> pd.DataFrame:
    """extras: uid, sample_id, source (extras_drusas.csv); links: uid, patient_id, age (das planilhas);
    manifest: uid, source, patient_id, group_id; splits: uid, outer, inner_0..inner_K.
    Devolve uma linha por drusa com group_id, age, outer e inner_*, no formato das divisões."""
    inner_cols = [c for c in splits.columns if c.startswith("inner_")]
    n_outer, n_inner = len(inner_cols), int(splits[inner_cols].max().max()) + 1
    as_text = lambda f: f.assign(uid=f["uid"].astype(str))     # uid é texto (hash), mesmo se só tiver dígitos
    extras, links, manifest, splits = map(as_text, (extras, links, manifest, splits))
    table = extras[["uid", "sample_id", "source"]].merge(links.drop_duplicates("uid")[["uid", "patient_id", "age"]],
                                                         on="uid", how="left", validate="one_to_one")
    table["patient_id"] = table["patient_id"].astype("string")
    main = manifest[["uid", "source", "patient_id", "group_id"]].merge(splits, on="uid", validate="one_to_one")
    main = main[main["patient_id"].notna()]
    main_key = _family(main["source"]) + "|" + main["patient_id"].astype(str)
    by_patient = main.assign(key=main_key).drop_duplicates("key").set_index("key")
    key = (_family(table["source"]) + "|" + table["patient_id"].astype(str)).where(table["patient_id"].notna())
    known = key.isin(by_patient.index)

    out = table[["uid", "sample_id", "source", "age"]].copy()
    out["group_id"] = None
    for col in ["outer", *inner_cols]:
        out[col] = -9
    hit = by_patient.loc[key[known]]
    out.loc[known, "group_id"] = hit["group_id"].to_numpy()
    for col in ["outer", *inner_cols]:
        out.loc[known, col] = hit[col].to_numpy()

    # Grupos novos: um por paciente (ou por imagem, sem paciente), em ordem determinística.
    new = ~known
    new_key = key[new].fillna("img|" + table.loc[new, "uid"].astype(str))
    groups = sorted(new_key.unique())
    rng = np.random.default_rng(seed)
    outer = rng.integers(0, n_outer, len(groups))
    inner = rng.integers(0, n_inner, (len(groups), n_outer))
    inner[np.arange(len(groups)), outer] = -1            # a dobra externa da imagem é o teste: sem dobra interna
    index = {g: i for i, g in enumerate(groups)}
    position = new_key.map(index).to_numpy()
    out.loc[new, "group_id"] = [f"dp_{i + 1:05d}" for i in position]
    out.loc[new, "outer"] = outer[position]
    for k, col in enumerate(inner_cols):
        out.loc[new, col] = inner[position, k]
    out[["outer", *inner_cols]] = out[["outer", *inner_cols]].astype(int)
    return out[["uid", "sample_id", "source", "group_id", "age", "outer", *inner_cols]]


def _links(sources: list[str]) -> pd.DataFrame:
    """uid (hash do arquivo original, como no build) -> paciente e idade, lendo as planilhas das bases."""
    from macdeg.data.sources import load_registry, read_source
    registry = load_registry()
    raw = pd.concat([read_source(n, registry["sources"][n]) for n in sources], ignore_index=True)
    raw = raw[raw["label_detail"].eq(DETAIL) & raw["path"].map(lambda p: Path(p).is_file())].copy()
    raw["uid"] = raw["path"].map(lambda p: hashlib.sha1(Path(p).read_bytes()).hexdigest()[:20])
    return raw[["uid", "patient_id", "age"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="~/macdeg_data/dataset_1024")
    ap.add_argument("--splits", default="splits_nested_seed0.csv")
    args = ap.parse_args()
    root = Path(args.out).expanduser()
    extras = pd.read_csv(root / "extras_drusas.csv", dtype={"uid": str})
    manifest = pd.read_csv(root / "manifest.csv", dtype={"uid": str, "patient_id": str})
    splits = pd.read_csv(root / args.splits, dtype={"uid": str})
    links = _links(sorted(extras["source"].unique()))
    folds = assign_folds(extras, links, manifest, splits)
    folds.to_csv(root / FOLDS_FILE, index=False)
    inherited = (~folds["group_id"].str.startswith("dp_")).sum()
    print(f"{len(folds)} drusas; {int(inherited)} herdaram o grupo de um paciente do conjunto principal; "
          f"com idade: {folds['age'].notna().mean():.0%}; por dobra externa: "
          f"{folds['outer'].value_counts().sort_index().to_dict()} -> {root / FOLDS_FILE}")


if __name__ == "__main__":
    main()
