"""Testes do programa de marcação, só com imagens sintéticas."""
from __future__ import annotations

import csv
import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pandas as pd
import pytest

from macdeg.annotate import Session, build_queue, load_answers, make_server
from macdeg.annotation_report import build_report
from macdeg.localization import DD_PRIOR_R, DISC_TO_FOVEA


def make_metadata(n: int) -> pd.DataFrame:
    return pd.DataFrame({"file": [f"paciente{i:03d}_R.png" for i in range(n)],
                         "original_path": [f"imgs/paciente{i:03d}_R.png" for i in range(n)],
                         "fov_radius": 900.0})


@pytest.fixture
def served(tmp_path):
    (tmp_path / "imgs").mkdir()
    metadata = make_metadata(4)
    for path in metadata["original_path"]:
        cv2.imwrite(str(tmp_path / path), np.full((1000, 2000, 3), 90, np.uint8))
    session = Session(build_queue(metadata, repeat_frac=0.0), tmp_path / "out" / "clicks.csv", tmp_path)
    server = make_server(session, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield session, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(base: str, payload: dict) -> urllib.request.Request:
    return urllib.request.Request(f"{base}/api/answer", json.dumps(payload).encode(),
                                  {"Content-Type": "application/json"})


def test_queue_is_reproducible_and_covers_every_image():
    metadata = make_metadata(500)
    queue = build_queue(metadata, seed=0)
    assert queue == build_queue(metadata, seed=0)
    assert [i.file for i in queue] != sorted(i.file for i in queue)
    assert {i.file for i in queue if i.repeat == 0} == set(metadata["file"])
    assert sum(i.repeat == 0 for i in queue) == 500
    assert sum(i.repeat == 1 for i in queue) == 15


def test_repeats_stay_far_from_the_first_view():
    queue = build_queue(make_metadata(1000), seed=1, min_gap=100)
    first = {i.file: p for p, i in enumerate(queue) if i.repeat == 0}
    repeats = [(i.file, p) for p, i in enumerate(queue) if i.repeat == 1]
    assert repeats
    assert all(p - first[f] >= 100 for f, p in repeats)


def test_short_queue_keeps_repeats_at_the_end():
    queue = build_queue(make_metadata(50), seed=0, repeat_frac=0.1, min_gap=100)
    assert sum(i.repeat == 0 for i in queue) == 50
    assert all(p >= 50 for p, i in enumerate(queue) if i.repeat == 1)


def test_click_is_saved_in_native_pixels(served):
    session, base = served
    urllib.request.urlopen(post(base, {"position": 0, "status": "clicked", "fx": 0.25, "fy": 0.5, "seconds": 3}))
    answer = load_answers(session.out)[(session.queue[0].file, 0)]
    assert float(answer["x_native"]) == 500.0
    assert float(answer["y_native"]) == 500.0


def test_responses_never_reveal_the_file_name(served):
    session, base = served
    bodies = [urllib.request.urlopen(f"{base}{path}").read() for path in ("/", "/api/state", "/api/item/0")]
    assert not any(b"paciente" in body or b"AMD" in body for body in bodies)
    jpeg = urllib.request.urlopen(f"{base}/image/0").read()
    assert jpeg[:2] == b"\xff\xd8"


def test_page_loads_nothing_from_outside(served):
    _, base = served
    page = urllib.request.urlopen(base + "/").read().decode()
    assert "http://" not in page and "https://" not in page and "//cdn" not in page


def test_latest_answer_wins_and_state_resumes(served):
    session, base = served
    urllib.request.urlopen(post(base, {"position": 0, "status": "clicked", "fx": 0.1, "fy": 0.1, "seconds": 1}))
    urllib.request.urlopen(post(base, {"position": 0, "status": "not_visible", "seconds": 1}))
    state = json.load(urllib.request.urlopen(f"{base}/api/state"))
    assert state["done"] == 1 and state["next"] == 1

    reopened = Session(session.queue, session.out, session.root)
    assert reopened.state()["next"] == 1
    assert reopened.answers[(session.queue[0].file, 0)]["status"] == "not_visible"
    with session.out.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.reader(handle))) == 3  # cabeçalho + 2 linhas: nada é sobrescrito


def test_truncated_last_line_is_ignored_and_does_not_corrupt_the_next(served):
    session, base = served
    urllib.request.urlopen(post(base, {"position": 0, "status": "clicked", "fx": 0.5, "fy": 0.5, "seconds": 1}))
    with session.out.open("a", encoding="utf-8") as handle:
        handle.write(f"{session.queue[1].file},0,cli")
    reopened = Session(session.queue, session.out, session.root)
    assert reopened.state()["done"] == 1
    reopened.record(1, "clicked", 0.5, 0.5, 1.0)
    assert Session(session.queue, session.out, session.root).state()["done"] == 2


@pytest.mark.parametrize("payload", [
    {"position": 0, "status": "clicked", "fx": 1.5, "fy": 0.5},
    {"position": 0, "status": "clicked"},
    {"position": 0, "status": "invented"},
    {"status": "clicked", "fx": 0.5, "fy": 0.5},
])
def test_invalid_answers_are_rejected_without_writing(served, payload):
    session, base = served
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(post(base, payload))
    assert error.value.code == 400
    assert load_answers(session.out) == {}


def test_position_outside_the_queue_returns_404(served):
    _, base = served
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f"{base}/api/item/99")
    assert error.value.code == 404


def test_requests_with_a_foreign_host_are_refused(served):
    _, base = served
    request = urllib.request.Request(f"{base}/api/state", headers={"Host": "site-externo.example"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 403


def test_server_listens_only_on_loopback(served):
    _, base = served
    assert base.startswith("http://127.0.0.1:")


def report_fixture():
    radius = 1000.0
    dd = DD_PRIOR_R * radius
    metadata = pd.DataFrame({
        "file": ["a.png", "b.png", "c.png"], "AMD": [0, 0, 1], "disc_uncertain": [False, True, False],
        "fov_cx": 1000.0, "fov_cy": 1000.0, "fov_radius": radius,
        "disc_dx": [0.6, 0.6, 0.6], "disc_dy": 0.0,
        "fovea_dx": 0.6 + DISC_TO_FOVEA[0], "fovea_dy": DISC_TO_FOVEA[1]})
    disc = 1000.0 + 0.6 * radius
    clicks = pd.DataFrame([
        {"file": "a.png", "repeat": 0, "status": "clicked", "x_native": disc, "y_native": 1000.0, "seconds": 2.0},
        {"file": "b.png", "repeat": 0, "status": "clicked", "x_native": disc + 2 * dd, "y_native": 1000.0, "seconds": 4.0},
        {"file": "c.png", "repeat": 0, "status": "not_visible", "x_native": "", "y_native": "", "seconds": 6.0},
        {"file": "a.png", "repeat": 1, "status": "clicked", "x_native": disc + 0.1 * dd, "y_native": 1000.0, "seconds": 2.0},
    ])
    return clicks, metadata


def test_report_measures_detector_error_in_disc_diameters():
    clicks, metadata = report_fixture()
    clicked_only = clicks[clicks["status"] == "clicked"]
    report = build_report(clicked_only, metadata)
    assert report["by_AMD_and_flag"]["0_confiante"]["disc_error_median_dd"] == 0.0
    assert report["by_AMD_and_flag"]["0_incerto"]["disc_error_median_dd"] == 2.0
    assert report["by_AMD"]["0"]["disc_wrong_frac"] == 0.5


def test_report_click_at_the_detected_disc_means_no_crop_shift():
    clicks, metadata = report_fixture()
    report = build_report(clicks[clicks["status"] == "clicked"], metadata)
    assert report["by_AMD_and_flag"]["0_confiante"]["crop_shift_median_dd"] == pytest.approx(0.0, abs=1e-3)


def test_report_handles_images_not_visible_and_measures_repeatability():
    clicks, metadata = report_fixture()  # como no CSV real, "não consigo ver" tem coordenadas vazias
    report = build_report(clicks, metadata)
    assert report["by_AMD"]["1"]["not_visible"] == 1.0
    assert "disc_error_median_dd" not in report["by_AMD"]["1"]
    assert report["repeatability"]["n"] == 1
    assert report["repeatability"]["median_dd"] == pytest.approx(0.1, abs=1e-3)


def test_report_output_contains_no_file_names():
    clicks, metadata = report_fixture()
    assert ".png" not in json.dumps(build_report(clicks, metadata))
