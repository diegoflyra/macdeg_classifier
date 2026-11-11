"""Leitores de cada base. Todos devolvem a mesma tabela, uma linha por imagem.

Colunas: source, path, patient_id, eye, label, label_detail, disease, age, sex, camera.
- label: 1 = DMRI, 0 = normal, NaN = fora do conjunto (outra doença, rótulo ambíguo).
- disease: nas imagens com outra doença (sem DMRI), as categorias de DISEASES separadas por "|".
  O build sorteia parte delas para entrar como "sem DMRI" (ver macdeg/data/select.py).
- patient_id: prefixado pela base, para que IDs iguais de bases diferentes não se misturem.
  Vazio quando a base não informa o paciente.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from macdeg.metadata import load_metadata

COLUMNS = ["source", "path", "patient_id", "eye", "label", "label_detail", "disease", "age", "sex", "camera"]
# Categorias comuns de "outra doença", iguais em todas as bases.
DISEASES = ["retinopatia_diabetica", "glaucoma", "edema_macular", "membrana_epirretiniana", "miopia",
            "retinopatia_hipertensiva", "catarata", "oclusao_vascular", "cicatriz", "nevo", "hemorragia",
            "descolamento", "nao_especificada", "outra"]


def _join(categories) -> str | None:
    present = set(categories)
    unique = [c for c in DISEASES if c in present]
    return "|".join(unique) if unique else None
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PROJECT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    """Caminhos do registro: `~` vira a pasta pessoal; relativos partem da raiz do projeto."""
    expanded = Path(os.path.expanduser(str(path)))
    return expanded if expanded.is_absolute() else PROJECT / expanded


def _finish(table: pd.DataFrame, name: str) -> pd.DataFrame:
    table = table.copy()
    table["source"] = name
    for column in COLUMNS:
        if column not in table:
            table[column] = np.nan
    has_patient = table["patient_id"].notna() & (table["patient_id"].astype(str) != "")
    table["patient_id"] = np.where(has_patient, name + ":" + table["patient_id"].astype(str), None)
    table["path"] = table["path"].astype(str)
    return table[COLUMNS]


def read_hyamd(name: str, spec: dict) -> pd.DataFrame:
    """HYAMD: 1 e 2 são DMRI. O controle (0) tem retinopatia diabética, não é normal: fica fora."""
    root = resolve(spec["path"])
    meta = load_metadata(root)
    detail = meta["AMD"].map({0: "retinopatia_diabetica", 1: "dmri_precoce", 2: "dmri_intermediaria_tardia"})
    label = meta["AMD"].map({0: np.nan, 1: 1, 2: 1})
    if spec.get("include_dr_as_negative"):
        label = meta["AMD"].map({0: 0, 1: 1, 2: 1})
    table = pd.DataFrame({
        "path": [str(root / "Images" / f) for f in meta["file"]], "patient_id": meta["patient_id"],
        "eye": meta["name_side"], "label": label, "label_detail": detail, "age": meta["age"], "sex": meta["sex"],
        "disease": np.where(meta["AMD"].eq(0), "retinopatia_diabetica", None)})
    return _finish(table, name)


def read_folders(name: str, spec: dict) -> pd.DataFrame:
    """Base organizada em pastas por classe. `classes` liga o nome da pasta (sem diferenciar
    maiúsculas) ao rótulo; pastas não listadas ficam fora. A classe é a pasta mais próxima da
    imagem que esteja em `classes`."""
    root = resolve(spec["path"])
    classes = {str(k).lower(): v for k, v in spec["classes"].items()}
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        folder = next((p.name for p in path.relative_to(root).parents if p.name.lower() in classes), None)
        if folder is None:
            continue
        label = classes[folder.lower()]
        rows.append({"path": str(path), "label": np.nan if label is None else label, "label_detail": folder})
    return _finish(pd.DataFrame(rows, columns=["path", "label", "label_detail"]), name)


def read_csv_labels(name: str, spec: dict) -> pd.DataFrame:
    """Base com uma planilha de rótulos. `values` liga cada valor da coluna de rótulo a 1, 0 ou
    null; valores não listados ficam fora. Colunas opcionais: patient, eye, age, sex."""
    root = resolve(spec["path"])
    sheet = pd.read_csv(root / spec["csv"], dtype=str)
    image_dir = root / spec.get("image_dir", ".")
    suffix = spec.get("image_suffix", "")
    values = {str(k): v for k, v in spec["values"].items()}
    raw = sheet[spec["label_column"]].astype(str)
    table = pd.DataFrame({
        "path": [str(image_dir / (str(v) + suffix)) for v in sheet[spec["image_column"]]],
        "label": raw.map(lambda v: values.get(v, None)).astype(float), "label_detail": raw})
    for key in ("patient", "eye", "age", "sex"):
        column = spec.get(f"{key}_column")
        if column:
            table["patient_id" if key == "patient" else key] = sheet[column].values
    return _finish(table, name)


ODIR_AMD = {"dry age-related macular degeneration", "wet age-related macular degeneration",
            "age-related macular degeneration"}
ODIR_UNUSABLE = {"low image quality", "image offset", "anterior segment image", "no fundus image",
                 "optic disk photographically invisible", "lens dust"}


def _keywords(text) -> set[str]:
    return {k.strip() for k in re.split(r"[,，]", str(text)) if k.strip()}


def odir_label(keywords: set[str], drusen_only) -> tuple[float, str]:
    """Rótulo de um olho da ODIR a partir das palavras-chave do diagnóstico desse olho."""
    if keywords & ODIR_UNUSABLE:
        return np.nan, "qualidade_insuficiente"
    if keywords & ODIR_AMD:
        return 1, "dmri"
    if "drusen" in keywords:
        return (np.nan if drusen_only is None else drusen_only), "drusas_sem_diagnostico_de_dmri"
    if keywords == {"normal fundus"}:
        return 0, "normal"
    return np.nan, "outra_condicao"


ODIR_DISEASES = [  # (trecho da palavra-chave, categoria); a primeira que casar vale para aquela palavra
    ("hypertensive retinopathy", "retinopatia_hipertensiva"), ("retinopathy", "retinopatia_diabetica"),
    ("glaucoma", "glaucoma"), ("macular edema", "edema_macular"), ("epiretinal membrane", "membrana_epirretiniana"),
    ("myopia", "miopia"), ("cataract", "catarata"), ("occlusion", "oclusao_vascular"),
    ("hemorrhage", "hemorragia"), ("retinal detachment", "descolamento"), ("nevus", "nevo"), ("scar", "cicatriz")]


def odir_disease(keywords: set[str]) -> str | None:
    """Categorias de outra doença de um olho da ODIR; palavras sem categoria viram "outra"."""
    found = []
    for word in keywords - {"normal fundus"}:
        found.append(next((cat for part, cat in ODIR_DISEASES if part in word), "outra"))
    return _join(found)


def read_odir(name: str, spec: dict) -> pd.DataFrame:
    """ODIR-5K (só o treino, que tem rótulos). O diagnóstico é por olho; o ID é do paciente.
    Normal = o único diagnóstico do olho é "normal fundus". Olho com DMRI e outra doença conta
    como DMRI. "drusen" sem diagnóstico de DMRI segue `drusen_only` (null = fora)."""
    root = resolve(spec["path"])
    sheet = pd.read_excel(root / "data.xlsx")
    rows = []
    for _, patient in sheet.iterrows():
        for side, eye in (("Left", "L"), ("Right", "R")):
            words = _keywords(patient[f"{side}-Diagnostic Keywords"])
            label, detail = odir_label(words, spec.get("drusen_only"))
            rows.append({"path": str(root / "Training Images" / patient[f"{side}-Fundus"]),
                         "patient_id": str(patient["ID"]), "eye": eye, "label": label, "label_detail": detail,
                         "disease": odir_disease(words) if detail == "outra_condicao" else None,
                         "age": patient["Patient Age"], "sex": patient["Patient Sex"]})
    return _finish(pd.DataFrame(rows), name)


BRSET_DISEASES = ["diabetic_retinopathy", "macular_edema", "scar", "nevus", "amd", "vascular_occlusion",
                  "hypertensive_retinopathy", "drusen", "hemorrhage", "retinal_detachment", "myopic_fundus",
                  "increased_cup_disc", "other"]


BRSET_TO_DISEASE = {"diabetic_retinopathy": "retinopatia_diabetica", "macular_edema": "edema_macular",
                    "scar": "cicatriz", "nevus": "nevo", "vascular_occlusion": "oclusao_vascular",
                    "hypertensive_retinopathy": "retinopatia_hipertensiva", "hemorrhage": "hemorragia",
                    "retinal_detachment": "descolamento", "myopic_fundus": "miopia",
                    "increased_cup_disc": "glaucoma", "other": "outra"}


def read_brset(name: str, spec: dict) -> pd.DataFrame:
    """BRSET. Só entram imagens de qualidade adequada.
    DMRI = coluna amd. Normal = nenhuma das 13 colunas de doença marcada e ICDR 0.
    Drusas sem DMRI seguem `drusen_only` (null = fora)."""
    root = resolve(spec["path"])
    sheet = pd.read_csv(root / "label_brset.csv", dtype={"image_id": str, "patient_id": str})
    adequate = sheet["quality"].eq("Adequate")
    normal = sheet[BRSET_DISEASES].sum(axis=1).eq(0) & sheet["DR_ICDR"].eq(0)
    amd = sheet["amd"].eq(1)
    drusen_only = sheet["drusen"].eq(1) & ~amd
    label = pd.Series(np.nan, index=sheet.index)
    detail = pd.Series("outra_condicao", index=sheet.index)
    label[amd], detail[amd] = 1, "dmri"
    label[normal], detail[normal] = 0, "normal"
    detail[drusen_only] = "drusas_sem_diagnostico_de_dmri"
    if spec.get("drusen_only") is not None:
        label[drusen_only] = spec["drusen_only"]
    label[~adequate], detail[~adequate] = np.nan, "qualidade_insuficiente"
    marked = sheet[list(BRSET_TO_DISEASE)].eq(1).assign(diabetic_retinopathy=lambda t: t["diabetic_retinopathy"]
                                                         | sheet["DR_ICDR"].gt(0))
    disease = [_join(BRSET_TO_DISEASE[c] for c in marked.columns[row]) for row in marked.to_numpy()]
    disease = pd.Series(disease, index=sheet.index).where(detail.eq("outra_condicao"), None)
    table = pd.DataFrame({
        "path": [str(root / "fundus_photos" / f"{i}.jpg") for i in sheet["image_id"]],
        "patient_id": sheet["patient_id"], "eye": sheet["exam_eye"].map({1: "R", 2: "L"}), "label": label,
        "label_detail": detail, "disease": disease, "age": sheet["patient_age"], "sex": sheet["patient_sex"].map({1: "M", 2: "F"}),
        "camera": sheet["camera"]})
    return _finish(table, name)


RFMID_TO_DISEASE = {"DR": "retinopatia_diabetica", "ODC": "glaucoma", "MYA": "miopia", "ERM": "membrana_epirretiniana",
                    "BRVO": "oclusao_vascular", "CRVO": "oclusao_vascular", "BRAO": "oclusao_vascular",
                    "CRAO": "oclusao_vascular", "CME": "edema_macular", "HR": "retinopatia_hipertensiva",
                    "VH": "hemorragia", "PRH": "hemorragia", "RD": "descolamento"}


def read_rfmid(name: str, spec: dict) -> pd.DataFrame:
    """RFMiD e RFMiD 2.0: uma coluna por doença (ARMD = DMRI, DN = drusas). Sem ID de paciente.
    `sets`: lista de {csv, images}. Normal = nenhuma doença marcada (e, na 2.0, WNL = 1)."""
    root = resolve(spec["path"])
    frames = []
    for part in spec["sets"]:
        sheet = pd.read_csv(root / part["csv"], encoding="latin-1")
        sheet.columns = [re.sub(r"[^A-Za-z0-9_]", "", str(c)) for c in sheet.columns]  # remove BOM e espaços
        sheet = sheet.loc[:, ~sheet.columns.str.startswith("Unnamed")]
        diseases = [c for c in sheet.columns if c not in ("ID", "Disease_Risk", "WNL")]
        marks = sheet[diseases].apply(pd.to_numeric, errors="coerce").fillna(0)
        normal = marks.sum(axis=1).eq(0)
        if "WNL" in sheet:
            normal &= sheet["WNL"].eq(1)
        if "Disease_Risk" in sheet:
            normal &= sheet["Disease_Risk"].eq(0)
        amd = marks["ARMD"].eq(1)
        drusen_only = marks["DN"].eq(1) & ~amd
        label = pd.Series(np.nan, index=sheet.index)
        detail = pd.Series("outra_condicao", index=sheet.index)
        label[normal], detail[normal] = 0, "normal"
        label[amd], detail[amd] = 1, "dmri"
        detail[drusen_only] = "drusas_sem_diagnostico_de_dmri"
        if spec.get("drusen_only") is not None:
            label[drusen_only] = spec["drusen_only"]
        other = detail.eq("outra_condicao")
        names = np.array(diseases)
        disease = [_join(RFMID_TO_DISEASE.get(c, "outra") for c in names[row]) if o else None
                   for row, o in zip(marks.eq(1).to_numpy(), other)]
        folder = root / part["images"]
        by_stem = {p.stem: p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        paths = [str(by_stem.get(str(i), folder / f"{i}.png")) for i in sheet["ID"]]
        frames.append(pd.DataFrame({"path": paths, "label": label, "label_detail": detail, "disease": disease}))
    return _finish(pd.concat(frames, ignore_index=True), name)


def read_adam(name: str, spec: dict) -> pd.DataFrame:
    """ADAM (iChallenge-AMD): 1.200 imagens, 400 por conjunto (treino, validação, teste), rótulo
    DMRI / não DMRI. "Não DMRI" mistura olhos normais e outras doenças (miopia, retinopatia...),
    sem dizer qual: por padrão fica fora; `non_amd_as_negative: true` conta como 0. Sem paciente."""
    root = resolve(spec["path"])
    rows = []
    for folder, label in (("AMD", 1), ("Non-AMD", 0)):
        for path in sorted((root / "Train" / "Training-image-400" / folder).glob("*.jpg")):
            rows.append((str(path), label))
    for csv, images in (("Validation/validation_classification_GT.txt", "Validation/image"),
                        ("Test/test_classification_GT.txt", "Test/Test-image-400")):
        for line in (root / csv).read_text().split("\n"):
            if line.split():
                file, label = line.split()
                rows.append((str(root / images / file), int(label)))
    table = pd.DataFrame(rows, columns=["path", "label"])
    table["label_detail"] = table["label"].map({1: "dmri", 0: "nao_dmri_normal_ou_outra_doenca"})
    table["disease"] = np.where(table["label"].eq(0), "nao_especificada", None)
    if not spec.get("non_amd_as_negative"):
        table.loc[table["label"].eq(0), "label"] = np.nan
    return _finish(table, name)


READERS = {"adam": read_adam, "hyamd": read_hyamd, "odir": read_odir, "brset": read_brset, "rfmid": read_rfmid,
           "folders": read_folders, "csv": read_csv_labels}


def load_registry(path: Path | str = PROJECT / "configs" / "sources.yaml") -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_source(name: str, spec: dict) -> pd.DataFrame:
    kind = spec["type"]
    if kind not in READERS:
        raise ValueError(f"{name}: tipo de leitor desconhecido: {kind}")
    return READERS[kind](name, spec)
