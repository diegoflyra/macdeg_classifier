"""macdeg.curvas: baixa as curvas do W&B (aqui, uma API falsa, sem rede) e gera os gráficos."""
from types import SimpleNamespace

from macdeg.curvas import CURVAS, fetch, plot


def _run(name, group, outer, epochs, state="finished"):
    rows = [{"epoca": e, "perda_treino": 1 / (e + 1), "auroc_val": 0.8 + e / 100, "lr": 1e-3,
             "segundos_epoca": 10.0, "segundos_esperando_dados": 1.0} for e in range(epochs)]
    return SimpleNamespace(name=name, group=group, state=state, config={"outer": outer, "inner": 0},
                           lastHistoryStep=epochs - 1, scan_history=lambda keys: [{k: r[k] for k in keys} for r in rows])


class FakeApi:
    def __init__(self, runs):
        self._runs = runs
        self.calls = []

    def runs(self, path, filters=None):
        self.calls.append((path, filters))
        return self._runs


def test_fetch_keeps_prefix_flags_empty_runs_and_plots(tmp_path):
    api = FakeApi([_run("a", "rodada6-x/rodada6-01", 0, 3), _run("b", "rodada6-y/rodada6-01", 1, 2),
                   _run("c", "rodada6-y/rodada6-01", 2, 0, state="crashed"), _run("d", "rodada5-z/rodada5-01", 0, 4),
                   _run("e", "rodada6-x", None, 0)])
    runs, curves = fetch(api, "eu/macdeg_classifier", "rodada6-")
    assert api.calls[0][1] == {"group": {"$regex": "^rodada6\\-"}}
    assert set(runs["run"]) == {"a", "b", "c", "e"}                       # a d é de outro grupo
    assert runs.set_index("run").loc["c", "epocas_baixadas"] == 0       # o "There's no data" do site
    assert len(curves) == 5 and set(CURVAS) <= set(curves.columns)
    written = plot(curves, tmp_path)
    assert [p.name for p in written] == ["rodada6-01.png"]          # os dois lotes juntos; o -analise fica fora
    assert all(p.stat().st_size > 0 for p in written)
