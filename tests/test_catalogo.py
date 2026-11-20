"""Catálogo dos experimentos (macdeg.catalogo) com lotes sintéticos."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from macdeg.catalogo import load_batches, render


def _batch(folder, name, created, configs, teste=False, signature="abc"):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(len(name))
    lote = {"nome": name, "criado": created, "commit": "f" * 40, "teste": teste, "assinatura_dados": signature,
            "n_tarefas": 20 * len(configs), "experimento": {"nome": name.split("-")[0], "base_mudancas": {"aux_dr": 0.5}},
            "configuracoes": [{"nome": c, "mudancas": {}, "config": {"model": "resnet50", "lr": 3e-4, "aux_dr": 0.5}}
                              for c in configs]}
    report = {"referencia": configs[0], "tarefas_faltando": [], "configuracoes": {}}
    labels = np.array([0, 1] * 50)
    for k, c in enumerate(configs):
        report["configuracoes"][c] = {"mudancas": {} if k == 0 else {"model": "x"}, "f2_media_bases": 0.8,
                                      "f2_media_bases_ic95": [0.78, 0.82], "auroc": 0.97, "auprc": 0.8,
                                      "sensibilidade": 0.88, "epocas_melhores": [5, 7],
                                      "especificidade_por_doenca": {"retinopatia_diabetica": {"especificidade": 0.5}},
                                      **({"veredito": {"delta_f2_media_bases": 0.01, "ic95": [-0.01, 0.03],
                                                       "veredito": "empatou"}} if k else {})}
        pd.DataFrame({"sample_id": [f"img_{i:05d}" for i in range(100)], "source": ["a", "b"] * 50,
                      "group_id": [f"g{i}" for i in range(100)], "label": labels,
                      "pred": (rng.random(100) < 0.5).astype(int)}).to_csv(folder / f"previsoes_externas_{c}.csv", index=False)
    (folder / "lote.json").write_text(json.dumps(lote))
    (folder / "relatorio.json").write_text(json.dumps(report))
    pd.DataFrame({"configuracao": configs, "seconds": [3600.0] * len(configs)}).to_csv(folder / "historico.csv", index=False)


def test_catalog_lists_batches_in_order_with_verdicts_and_skips_test_batches(tmp_path):
    root = tmp_path / "resultados"
    _batch(root / "b", "rodadaB-2", "2026-11-02 10:00", ["rodadaB-00", "rodadaB-01"])
    _batch(root / "a", "rodadaA-1", "2026-11-01 10:00", ["rodadaA-00"])
    _batch(root / "t", "teste-1", "2026-11-03 10:00", ["teste-00"], teste=True)
    batches = load_batches(root)
    assert [b["lote"]["nome"] for b in batches] == ["rodadaA-1", "rodadaB-2"]
    text = render(batches)
    assert "## rodadaA-1" in text and "## rodadaB-2" in text and "teste-1" not in text
    assert "**empatou**" in text and "`model: \"x\"`" in text and "1,0 h" in text
    assert "contra a base do lote anterior (`rodadaA-00`)" in text      # mesma assinatura: compara as bases
    assert "img_" not in text                                            # só agregados no catálogo
