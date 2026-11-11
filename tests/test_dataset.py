"""Testes do conjunto multibase e do treino, com retinografias sintéticas."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from macdeg.data.build import build, dhash, fov_crop, fov_square
from macdeg.data.splits import assign_nested
from macdeg.metrics import evaluate, source_only_scores


def fake_fundus(seed: int, width: int = 400, height: int = 300) -> np.ndarray:
    """Círculo claro com 'vasos' aleatórios sobre fundo preto, num quadro retangular."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), np.uint8)
    center, radius = (width // 2, height // 2), int(height * 0.45)
    cv2.circle(image, center, radius, (40, 90, 170), -1)
    for _ in range(25):
        p1 = tuple(int(v) for v in rng.integers([width * 0.2, 0], [width * 0.8, height]))
        p2 = tuple(int(v) for v in rng.integers([width * 0.2, 0], [width * 0.8, height]))
        cv2.line(image, p1, p2, (20, 40, 110), int(rng.integers(1, 4)))
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, center, radius, 1, -1)
    return image * mask[..., None]


def write(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


@pytest.fixture
def two_sources(tmp_path):
    a, b = tmp_path / "base_a", tmp_path / "base_b"
    for k in range(12):
        write(a / "AMD" / f"a{k}.png", fake_fundus(k))
        write(a / "Normal" / f"n{k}.png", fake_fundus(100 + k))
    write(a / "Glaucoma" / "g0.png", fake_fundus(200))
    for k in range(12):
        write(b / "armd" / f"x{k}.jpg", fake_fundus(300 + k))
        write(b / "normal" / f"y{k}.jpg", fake_fundus(400 + k))
    # Cópias republicadas na base B:
    original = fake_fundus(0)
    write(b / "armd" / "copia_exata.png", original)
    cropped = cv2.resize(fov_crop(original), (180, 180), interpolation=cv2.INTER_AREA)
    write(b / "armd" / "copia_cortada_reduzida.jpg", cropped)
    write(b / "normal" / "copia_rotulo_trocado.png", fake_fundus(1))
    registry = {"sources": {
        "base_a": {"type": "folders", "path": str(a), "classes": {"AMD": 1, "Normal": 0, "Glaucoma": None}},
        "base_b": {"type": "folders", "path": str(b), "classes": {"armd": 1, "normal": 0}}}}
    out = tmp_path / "dataset"
    summary = build(registry, ["base_a", "base_b"], out, size=96, workers=2)
    return out, summary


def test_dhash_survives_border_crop_and_resize():
    original = fake_fundus(5)
    copy = cv2.resize(fov_crop(original), (150, 150), interpolation=cv2.INTER_AREA)
    other = fake_fundus(6)
    distance = lambda x, y: int(np.bitwise_count(dhash(x) ^ dhash(y)).sum())
    assert distance(original, copy) < distance(original, other)


def test_labels_are_mapped_and_other_diseases_left_out(two_sources):
    out, summary = two_sources
    manifest = pd.read_csv(out / "manifest.csv")
    assert set(manifest["label"]) == {0, 1}
    assert not manifest["label_detail"].str.lower().eq("glaucoma").any()
    assert {"source": "base_a", "label_detail": "Glaucoma", "n": 1} in summary["fora_do_conjunto_por_rotulo"]


def test_copies_across_sources_are_removed(two_sources):
    out, summary = two_sources
    dropped = pd.read_csv(out / "dropped.csv")
    reasons = dropped.groupby(["source", "reason"]).size().to_dict()
    assert reasons.get(("base_b", "duplicata_exata")) == 1
    assert reasons.get(("base_b", "quase_duplicata_entre_bases")) == 1
    # A cópia com rótulo trocado derruba as duas versões.
    assert reasons.get(("base_a", "rotulo_conflitante_entre_copias")) == 1
    assert reasons.get(("base_b", "rotulo_conflitante_entre_copias")) == 1
    manifest = pd.read_csv(out / "manifest.csv")
    assert manifest["uid"].is_unique
    assert summary["por_base"]["base_a"] == {"dmri": 11, "normal": 12, "outra_doenca": 0, "com_paciente": 0.0, "pacientes": 0, "grupos": 23}


def test_reduced_images_exist_with_the_requested_size(two_sources):
    out, _ = two_sources
    manifest = pd.read_csv(out / "manifest.csv")
    image = cv2.imread(str(out / manifest["image"].iloc[0]))
    assert image.shape == (96, 96, 3)


def test_fov_square_keeps_the_whole_retina():
    """Círculo mais largo que a altura da foto (como na BRSET): as bordas laterais não podem sumir."""
    image = np.zeros((400, 600, 3), np.uint8)
    cv2.circle(image, (300, 200), 250, (90, 60, 40), -1)
    image[190:210, 52:62] = 255   # marca na borda esquerda da retina
    image[190:210, 538:548] = 255  # e na direita
    square = fov_square(image, 256)
    assert square.shape == (256, 256, 3)
    assert square[:, :40].max() > 150 and square[:, -40:].max() > 150
    assert square[:10].max() < 20  # faixas pretas em cima e embaixo


def make_manifest(n_patients: int = 120, per_patient: int = 2) -> pd.DataFrame:
    rows = []
    for p in range(n_patients):
        source = "a" if p % 2 else "b"
        detail = "dmri" if (p // 2) % 4 == 0 else "outra_doenca" if (p // 2) % 4 == 1 else "normal"
        for k in range(per_patient):
            rows.append({"uid": f"u{p}_{k}", "source": source, "label": int(detail == "dmri"),
                         "label_detail": detail, "group_id": f"g{p}"})
    return pd.DataFrame(rows)


def test_nested_splits_keep_groups_together_and_never_mix_test():
    manifest = make_manifest()
    splits = assign_nested(manifest, seed=0).merge(manifest, on="uid")
    assert set(splits["outer"]) == set(range(5))
    for k in range(5):
        column = splits[f"inner_{k}"]
        assert (column[splits["outer"] == k] == -1).all()          # teste externo fora das internas
        assert set(column[splits["outer"] != k]) == set(range(4))
        sides = splits.assign(side=splits["outer"].astype(str) + "_" + column.astype(str))
        assert (sides.groupby("group_id")["side"].nunique() == 1).all()
    for (k, source), rows in splits.groupby(["outer", "source"]):
        assert rows["label"].nunique() == 2


def test_nested_splits_are_reproducible():
    manifest = make_manifest()
    assert assign_nested(manifest, seed=3).equals(assign_nested(manifest, seed=3))


def test_f2_and_threshold_from_other_folds():
    from macdeg.metrics import best_threshold, crossfit, f2_mean, fbeta
    assert fbeta([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(5 * 1 / (5 * 1 + 4 * 1 + 0))
    rng = np.random.default_rng(0)
    labels = (rng.random(400) < 0.2).astype(int)
    frame = pd.DataFrame({"source": np.where(np.arange(400) % 2, "a", "rfmid2"), "label": labels,
                          "prob": np.clip(labels * 0.5 + rng.random(400) * 0.6, 0, 1), "fold": np.arange(400) % 4})
    t = best_threshold(frame)
    assert 0 < t < 1
    fitted = crossfit(frame, "fold")
    assert fitted.groupby("fold")["threshold"].nunique().eq(1).all()
    assert 0 < f2_mean(fitted) <= 1


def test_report_and_paired_verdict():
    from macdeg.metrics import paired_verdict, report
    rng = np.random.default_rng(1)
    n = 600
    labels = (np.arange(n) % 5 == 0).astype(int)
    kinds = np.where(labels == 1, "dmri", np.where(np.arange(n) % 5 == 1, "outra", "normal"))
    base = pd.DataFrame({"sample_id": [f"img_{i:05d}" for i in range(n)], "source": np.where(np.arange(n) % 2, "a", "b"),
                         "group_id": [f"g{i // 2}" for i in range(n)], "label": labels, "kind": kinds,
                         "disease_group": np.where(kinds == "outra", "glaucoma", None)})
    weak = base.assign(prob=np.clip(labels * 0.2 + rng.random(n) * 0.8, 0, 1))
    strong = base.assign(prob=np.clip(labels * 0.7 + rng.random(n) * 0.3, 0, 1))
    for f in (weak, strong):
        f["pred"] = (f["prob"] >= 0.6).astype(int)
    r = report(strong, n_bootstrap=100)
    assert r["f2_media_bases_ic95"][0] <= r["f2_media_bases"] <= r["f2_media_bases_ic95"][1]
    assert set(r["por_base"]) == {"a", "b"} and "glaucoma" in r["especificidade_por_doenca"]
    assert paired_verdict(weak, strong, n=200)["veredito"] == "ganhou"
    assert paired_verdict(strong, weak, n=200)["veredito"] == "perdeu"
    assert paired_verdict(strong, strong, n=200)["veredito"] == "empatou"


def test_source_only_ruler_detects_a_source_shortcut():
    train = pd.DataFrame({"source": ["a"] * 10 + ["b"] * 10, "label": [1] * 10 + [0] * 10})
    val = pd.DataFrame({"source": ["a"] * 3 + ["b"] * 3, "group_id": [f"g{i}" for i in range(6)],
                        "label": [1] * 3 + [0] * 3, "prob": [0.5] * 6})
    assert evaluate(val, train, n_bootstrap=50)["regua_origem_auroc"] == 1.0
    assert np.allclose(source_only_scores(train.assign(label=[1, 0] * 10), val), 0.5)


def _splits(out):
    manifest = pd.read_csv(out / "manifest.csv")
    assign_nested(manifest, seed=0).to_csv(out / "splits_nested_seed0.csv", index=False)


SMOKE = {"name": "fumaça", "model": "resnet18", "pretrained": False, "image_size": 64, "batch_size": 4, "micro_batch": 2,
         "epochs": 2, "lr": 1e-3, "num_workers": 0, "balance": "pos_weight", "device": "cpu",
         "augment": {"rotate_deg": 10, "color": 0.1}}


def test_training_smoke_run_writes_predictions_with_our_ids_only(two_sources, tmp_path):
    from macdeg.train import run
    out, _ = two_sources
    _splits(out)
    config = {**SMOKE, "dataset": str(out)}
    target = tmp_path / "runs" / "outer0_inner1"
    metrics = run(config, outer=0, inner=1, seed=0, out=target)
    assert 0.0 <= metrics["auroc"] <= 1.0 and metrics["micro_batch"] == 2
    assert len(pd.read_csv(target / "history.csv")) == 2
    for name in ("val", "test"):
        preds = pd.read_csv(target / f"{name}_predictions.csv")
        assert preds["sample_id"].str.startswith("img_").all()
        assert not {"uid", "path", "image"} & set(preds.columns)
    manifest = pd.read_csv(out / "manifest.csv").merge(pd.read_csv(out / "splits_nested_seed0.csv"), on="uid")
    test_ids = set(manifest.loc[manifest["outer"] == 0, "sample_id"])
    assert set(pd.read_csv(target / "test_predictions.csv")["sample_id"]) == test_ids
    assert not test_ids & set(pd.read_csv(target / "val_predictions.csv")["sample_id"])
    assert run(config, outer=0, inner=1, seed=0, out=target) is None


def test_training_refuses_a_single_class(two_sources, tmp_path):
    from macdeg.train import run
    out, _ = two_sources
    manifest = pd.read_csv(out / "manifest.csv")
    manifest[manifest["label"] == 1].to_csv(out / "manifest.csv", index=False)
    _splits(out)
    with pytest.raises(SystemExit):
        run({**SMOKE, "dataset": str(out), "epochs": 1}, outer=0, inner=0, seed=0, out=tmp_path / "r")


def test_odir_labels_per_eye():
    from macdeg.data.sources import odir_label
    assert odir_label({"normal fundus"}, None) == (0, "normal")
    assert odir_label({"dry age-related macular degeneration", "cataract"}, None)[0] == 1
    assert np.isnan(odir_label({"normal fundus", "lens dust"}, None)[0])
    assert np.isnan(odir_label({"drusen"}, None)[0])
    assert odir_label({"drusen"}, 1)[0] == 1
    assert np.isnan(odir_label({"normal fundus", "moderate non proliferative retinopathy"}, None)[0])
    assert np.isnan(odir_label({"wet age-related macular degeneration", "low image quality"}, None)[0])


def test_brset_reader_maps_labels(tmp_path):
    from macdeg.data.sources import BRSET_DISEASES, read_brset
    rows = []
    def row(image, **marks):
        base = {"image_id": image, "patient_id": "7", "camera": "Canon CR", "patient_age": 70, "patient_sex": 2,
                "exam_eye": 1, "DR_ICDR": 0, "quality": "Adequate", **{d: 0 for d in BRSET_DISEASES}}
        base.update(marks)
        rows.append(base)
    row("img1")                                   # normal
    row("img2", amd=1, drusen=1)                  # DMRI
    row("img3", drusen=1)                         # drusas sem DMRI: fora
    row("img4", amd=1, quality="Inadequate")      # qualidade ruim: fora
    row("img5", DR_ICDR=1)                        # retinopatia leve: não é normal
    row("img6", increased_cup_disc=1)             # escavação aumentada: não é normal
    pd.DataFrame(rows).to_csv(tmp_path / "label_brset.csv", index=False)
    table = read_brset("brset", {"path": str(tmp_path)}).set_index(pd.Index([r["image_id"] for r in rows]))
    assert table.loc["img1", "label"] == 0 and table.loc["img2", "label"] == 1
    assert table[["label"]].loc[["img3", "img4", "img5", "img6"]].isna().all().all()
    assert table.loc["img4", "label_detail"] == "qualidade_insuficiente"
    assert table.loc["img1", "patient_id"] == "brset:7" and table.loc["img1", "camera"] == "Canon CR"
    with_drusen = read_brset("brset", {"path": str(tmp_path), "drusen_only": 1})
    assert with_drusen["label"].iloc[2] == 1


def test_rfmid_reader_handles_both_versions(tmp_path):
    from macdeg.data.sources import read_rfmid
    images = tmp_path / "imgs"
    images.mkdir()
    for i in (1, 2, 3, 4):
        (images / f"{i}.JPG").write_bytes(b"x")
    v1 = pd.DataFrame({"ID": [1, 2, 3, 4], "Disease_Risk": [0, 1, 1, 1], "ARMD": [0, 1, 0, 0], "DN": [0, 1, 1, 0],
                       "DR": [0, 0, 0, 1]})
    v1.to_csv(tmp_path / "v1.csv", index=False)
    # RFMiD 2.0: cabeçalho com BOM, coluna WNL e coluna vazia no fim.
    v2 = "﻿ID,WNL,ARMD,DN,DR,\n1,1,0,0,0,\n2,0,1,0,0,\n3,0,0,1,0,\n4,0,0,0,1,\n"
    (tmp_path / "v2.csv").write_bytes(v2.encode("utf-8"))
    for csv_name in ("v1.csv", "v2.csv"):
        table = read_rfmid("r", {"path": str(tmp_path), "sets": [{"csv": csv_name, "images": "imgs"}]})
        assert table["label"].tolist()[:2] == [0, 1]
        assert table["label"].iloc[2:].isna().all()
        assert table["label_detail"].iloc[2] == "drusas_sem_diagnostico_de_dmri"
        assert table["path"].iloc[0].endswith("1.JPG")
        assert table["patient_id"].isna().all()


def test_adam_reader_keeps_only_amd_by_default(tmp_path):
    from macdeg.data.sources import read_adam
    train = tmp_path / "Train" / "Training-image-400"
    for folder, names in (("AMD", ["A0001.jpg"]), ("Non-AMD", ["N0001.jpg", "N0002.jpg"])):
        (train / folder).mkdir(parents=True)
        for n in names:
            (train / folder / n).touch()
    (tmp_path / "Validation").mkdir()
    (tmp_path / "Validation" / "validation_classification_GT.txt").write_text("V0001.jpg 0\nV0002.jpg 1\n")
    (tmp_path / "Test").mkdir()
    (tmp_path / "Test" / "test_classification_GT.txt").write_text("T0002.jpg  1\nT0001.jpg  0\n")
    table = read_adam("adam", {"path": str(tmp_path)})
    assert len(table) == 7 and table["label"].eq(1).sum() == 3 and table["label"].isna().sum() == 4
    assert table["path"].str.endswith("Validation/image/V0002.jpg").any()
    both = read_adam("adam", {"path": str(tmp_path), "non_amd_as_negative": True})
    assert both["label"].eq(0).sum() == 4


def test_level_fill_spreads_evenly_and_respects_capacity():
    from macdeg.data.select import level_fill
    assert level_fill({}, {"a": 10, "b": 10, "c": 2}, 12) == {"a": 5, "b": 5, "c": 2}
    assert level_fill({"a": 4}, {"a": 10, "b": 10}, 6) == {"a": 1, "b": 5}
    assert level_fill({}, {"a": 1}, 5) == {"a": 1}


def test_select_other_diseases_rules():
    from macdeg.data.select import select_other_diseases
    rows = []
    rows += [dict(source="x", label=0, disease=None) for _ in range(40)]          # 40 normais
    rows += [dict(source="x", label=1, disease=None) for _ in range(5)]
    rows += [dict(source="y", label=1, disease=None) for _ in range(6)]           # base sem normais
    rows += [dict(source="y", label=np.nan, disease="retinopatia_diabetica") for _ in range(30)]
    rows += [dict(source="x", label=np.nan, disease="glaucoma") for _ in range(30)]
    rows += [dict(source="x", label=np.nan, disease="catarata") for _ in range(3)]
    rows += [dict(source="x", label=np.nan, disease="glaucoma|catarata") for _ in range(2)]
    table = pd.DataFrame(rows).assign(label_detail="x")
    out, stats = select_other_diseases(table, {"other_to_normal": 0.5, "floor_sources": ["y"], "seed": 0})
    picked = out[out["label_detail"] == "outra_doenca"]
    assert len(picked) == 20 == stats["alvo"]
    assert (picked["source"] == "y").sum() >= 6                     # piso: tantos "sem DMRI" quanto DMRI
    assert stats["por_categoria"]["catarata"] == 5                  # rara entra inteira (com as duplas)
    assert out.loc[out["disease"] == "glaucoma|catarata", "disease_group"].eq("catarata").all()
    again, _ = select_other_diseases(table, {"other_to_normal": 0.5, "floor_sources": ["y"], "seed": 0})
    assert again["label"].equals(out["label"])                      # mesma semente, mesmo sorteio
