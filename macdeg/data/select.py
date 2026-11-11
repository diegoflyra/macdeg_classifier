"""Sorteia as imagens com outra doença que entram como "sem DMRI".

Regras (configs/sources.yaml, seção `selection`):
1. Total de outras doenças = `other_to_normal` × número de olhos normais (0,5 = um terço do "sem DMRI").
2. Bases em `floor_sources` (só DMRI e outras doenças, sem normais) recebem antes pelo menos tantos
   "sem DMRI" quanto têm de DMRI, para que reconhecer a base não diga o rótulo.
3. O restante é dividido por igual entre as categorias de doença (as que têm menos imagens entram
   inteiras e a sobra vai para as outras). Dentro de cada categoria, por igual entre as bases.
4. Imagem com mais de uma doença conta só para a categoria mais rara.
O sorteio usa a semente `seed`: o mesmo registro gera sempre o mesmo conjunto.
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

OTHER_DETAIL = "outra_doenca"


def level_fill(start: dict, capacity: dict, n: int) -> dict:
    """Distribui `n` unidades, uma a uma, sempre para quem tem a menor contagem atual (`start` + o já
    dado) e ainda tem capacidade. Empates pelo nome. Devolve quanto cada um recebeu."""
    given = {k: 0 for k in capacity}
    heap = [(start.get(k, 0), k) for k in sorted(capacity) if capacity[k] > 0]
    heapq.heapify(heap)
    while n > 0 and heap:
        count, key = heapq.heappop(heap)
        given[key] += 1
        n -= 1
        if given[key] < capacity[key]:
            heapq.heappush(heap, (count + 1, key))
    return given


def _sample(rows: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.Index:
    return rows.index[np.sort(rng.choice(len(rows), size=k, replace=False))] if k else rows.index[:0]


def select_other_diseases(table: pd.DataFrame, config: dict | None) -> tuple[pd.DataFrame, dict]:
    """Marca como 0 ("sem DMRI", label_detail = outra_doenca) as imagens sorteadas. Candidatas: rótulo
    NaN com `disease` preenchida. Acrescenta `disease_group` (a categoria usada no balanço)."""
    table = table.copy()
    config = config or {}
    ratio = float(config.get("other_to_normal", 0))
    candidates = table[table["label"].isna() & table["disease"].notna()]
    table["disease_group"] = None
    if ratio <= 0 or candidates.empty:
        return table, {"outras_doencas": 0}

    lists = candidates["disease"].str.split("|")
    frequency = lists.explode().value_counts()
    group = lists.map(lambda cats: min(cats, key=lambda c: (frequency[c], c)))
    table.loc[candidates.index, "disease_group"] = group
    candidates = table.loc[candidates.index]

    rng = np.random.default_rng(int(config.get("seed", 0)))
    total = int(round(ratio * int((table["label"] == 0).sum())))
    chosen = pd.Index([])

    for source in config.get("floor_sources", []):
        here = table["source"] == source
        need = int(((table["label"] == 1) & here).sum() - ((table["label"] == 0) & here).sum())
        pool = candidates[candidates["source"] == source]
        by_group = {g: len(rows) for g, rows in pool.groupby("disease_group")}
        for g, k in level_fill({}, by_group, max(0, min(need, total - len(chosen)))).items():
            chosen = chosen.append(_sample(pool[pool["disease_group"] == g], k, rng))

    remaining = candidates.drop(chosen)
    have = table.loc[chosen, "disease_group"].value_counts().to_dict()
    capacity = remaining["disease_group"].value_counts().to_dict()
    for g, k in level_fill(have, capacity, max(0, total - len(chosen))).items():
        pool = remaining[remaining["disease_group"] == g]
        per_source = level_fill({}, pool["source"].value_counts().to_dict(), k)
        for source, m in per_source.items():
            chosen = chosen.append(_sample(pool[pool["source"] == source], m, rng))

    table.loc[chosen, "label"] = 0
    table.loc[chosen, "label_detail"] = OTHER_DETAIL
    picked = table.loc[chosen]
    stats = {"outras_doencas": int(len(chosen)), "alvo": total,
             "por_categoria": picked["disease_group"].value_counts().sort_index().astype(int).to_dict(),
             "por_base": picked["source"].value_counts().sort_index().astype(int).to_dict()}
    return table, stats
