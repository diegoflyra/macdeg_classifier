"""Tabela de metadados por imagem, ligada aos arquivos da HYAMD.

O `image_id` do labels.csv não coincide com o nome do arquivo em 1370 das 1560 linhas:
- os arquivos têm um `_` final no sufixo e o `patient_id` da coluna perdeu zeros à esquerda;
- num olho com n imagens, os arquivos usam {sem sufixo, 1..n-1} e o CSV usa {1..n}.

Não há como saber qual linha corresponde a qual arquivo dentro de um olho, mas isso não
importa: paciente, sexo, idade e rótulo são constantes por olho, e `load_metadata` confere
essa constância. A junção, portanto, é feita por olho (paciente + lado do nome do arquivo).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_NAME = re.compile(r"(\d+)_([A-Za-z]+)(?:_(\d+))?_?")


def _parse(stem: str) -> tuple[str, str]:
    match = _NAME.fullmatch(stem)
    if match is None:
        raise ValueError("nome fora do padrão esperado")
    return match.group(1), match.group(2)


def load_metadata(root: Path | str = "hyamd") -> pd.DataFrame:
    """Uma linha por arquivo: file, patient_id, name_side, side, eye_id, sex, age, AMD, side_anomaly."""
    root = Path(root)
    labels = pd.read_csv(root / "labels" / "labels.csv", dtype={"image_id": str, "patient_id": str})
    parsed = [_parse(s) for s in labels["image_id"]]
    labels["pid"] = [p for p, _ in parsed]
    labels["name_side"] = [s for _, s in parsed]

    # O patient_id da coluna é o do nome do arquivo sem zeros à esquerda.
    if not (labels["pid"].str.lstrip("0") == labels["patient_id"].str.lstrip("0")).all():
        raise ValueError("patient_id da coluna difere do nome do image_id")

    per_name_eye = labels.groupby(["pid", "name_side"])
    for column in ["sex", "age", "side", "AMD"]:
        if (per_name_eye[column].nunique() > 1).any():
            raise ValueError(f"{column} varia dentro de um olho: a junção por olho não é válida")
    eyes = per_name_eye[["sex", "age", "side", "AMD"]].first().reset_index()
    expected_counts = per_name_eye.size()

    files = sorted(p.name for p in (root / "Images").iterdir() if p.suffix.lower() == ".png")
    table = pd.DataFrame({"file": files})
    file_parts = [_parse(Path(f).stem) for f in files]
    table["pid"] = [p for p, _ in file_parts]
    table["name_side"] = [s for _, s in file_parts]
    if not table.groupby(["pid", "name_side"]).size().sort_index().equals(expected_counts.sort_index()):
        raise ValueError("número de imagens por olho difere entre arquivos e labels.csv")

    table = table.merge(eyes, on=["pid", "name_side"], how="left", validate="many_to_one")
    table["patient_id"] = table.pop("pid")
    table["side_anomaly"] = ~table["name_side"].isin(["L", "R"])
    table["eye_id"] = table["patient_id"] + "_" + table["name_side"]
    return table
