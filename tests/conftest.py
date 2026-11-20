"""Fixtures compartilhadas entre os testes."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from macdeg.data.splits import assign_nested


@pytest.fixture()
def tiny_dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(150):
        source = ["a", "b", "rfmid2"][i % 3]
        detail = ["dmri", "normal", "normal", "outra_doenca"][(i // 3) % 4]
        uid = f"{i:020x}"
        image = np.full((64, 64, 3), 40, np.uint8)
        cv2.circle(image, (32, 32), 28, (60, 90, 160), -1)
        if detail == "dmri":  # "drusas" claras: o modelo tem algo a aprender
            for _ in range(6):
                cv2.circle(image, tuple(int(v) for v in rng.integers(20, 44, 2)), 2, (200, 220, 240), -1)
        cv2.imwrite(str(root / "images" / f"{uid}.jpg"), image)
        rows.append({"uid": uid, "source": source, "label": int(detail == "dmri"), "label_detail": detail,
                     "disease": "glaucoma" if detail == "outra_doenca" else None,
                     "disease_group": "glaucoma" if detail == "outra_doenca" else None,
                     "patient_id": None, "group_id": f"g{i}", "age": 50 + 20 * (detail == "dmri") + i % 7,
                     "image": f"images/{uid}.jpg"})
    manifest = pd.DataFrame(rows).sort_values("uid").reset_index(drop=True)
    manifest["sample_id"] = [f"img_{i:05d}" for i in range(len(manifest))]
    manifest.to_csv(root / "manifest.csv", index=False)
    assign_nested(manifest, seed=0).to_csv(root / "splits_nested_seed0.csv", index=False)
    drusas = manifest.head(10)[["uid", "source", "image"]].assign(sample_id=[f"dru_{i:05d}" for i in range(10)],
                                                                   group_id=[f"d{i}" for i in range(10)])
    drusas.to_csv(root / "extras_drusas.csv", index=False)
    return root
