import os
import pickle

import pandas as pd
import pytest

from behaviour.adapters.tailtrail import read_data_xlsx, read_settings, settings_to_toml, video_id_from_label
from behaviour.config import load_config

from .conftest import ROOT

DEFAULT_SETTINGS = ROOT / "tail-trail" / "default_settings"


@pytest.mark.skipif(not DEFAULT_SETTINGS.exists(), reason="TailTrail not in project")
def test_import_default_settings(tmp_path):
    s = read_settings(DEFAULT_SETTINGS)
    assert s["scale"] == pytest.approx(266.66)
    assert [r["name"] for r in s["rois"]] == ["3", "2", "1"]
    out = tmp_path / "tt.toml"
    out.write_text(settings_to_toml(s), encoding="utf-8")
    cfg = load_config(out)
    assert cfg.px_per_m == pytest.approx(266.66)
    assert [z.name for z in cfg.zones] == ["3", "2", "1"]
    assert cfg.zones[2].r == 175 and cfg.zones[2].cx == 320
    assert cfg.zones[0].hysteresis_px == pytest.approx(0.035 * 266.66)


def test_settings_loader_refuses_code(tmp_path):
    evil = tmp_path / "evil"
    evil.write_bytes(pickle.dumps(os.getcwd))
    with pytest.raises(pickle.UnpicklingError):
        read_settings(evil)


def test_video_id_from_label():
    assert video_id_from_label("Object 1") == "1"
    assert video_id_from_label("Object rat_07.avi") == "rat_07"
    assert video_id_from_label("ObjectR12") == "R12"


def write_xlsx(path):
    # layout from the TailTrail manual; rat_02 exported twice (last row should win)
    rows = [
        ["Object rat_01", 300.23, 56.34, 276.23, 49.64, 17, 22.60, 6.31, 18, 1.40, 0.39, 2, 0],
        ["Object rat_02", 300.0, 40.0, 250.0, 30.0, 10, 40.0, 8.0, 9, 5.0, 2.0, 3, 4],
        ["Object rat_02", 300.0, 41.0, 251.0, 31.0, 11, 39.0, 8.0, 9, 5.0, 2.0, 3, 1],
    ]
    cols = ["#", "total time (s)", "total distance (m)",
            "Outer time (s)", "Outer distance (m)", "Outer entrances",
            "Inner time (s)", "Inner distance (m)", "Inner entrances",
            "Center time (s)", "Center distance (m)", "Center entrances", "losses"]
    pd.DataFrame(rows, columns=cols).to_excel(path, index=False)


def get(long, vid, metric, zone=""):
    v = long[(long.video_id == vid) & (long.metric == metric) & (long.zone == zone)]["value"]
    assert len(v) == 1, (vid, metric, zone)
    return float(v.iloc[0])


def test_data_xlsx(tmp_path, cfg):
    path = tmp_path / "data.xlsx"
    write_xlsx(path)
    with pytest.warns(UserWarning, match="more than once"):
        long = read_data_xlsx(path, cfg)
    assert sorted(long.video_id.unique()) == ["rat_01", "rat_02"]
    assert get(long, "rat_01", "total_distance_m") == pytest.approx(56.34)
    assert get(long, "rat_01", "time_s", "center") == pytest.approx(1.40)
    # start_zone = center in the example config: the start "entrance" is removed there only
    assert get(long, "rat_01", "entries", "center") == 1
    assert get(long, "rat_01", "entries", "inner") == 18
    assert get(long, "rat_01", "unassigned_time_s") == pytest.approx(300.23 - 276.23 - 22.60 - 1.40)
    assert get(long, "rat_02", "total_distance_m") == 41.0
    assert get(long, "rat_02", "losses") == 1
    assert set(long.kind) == {"behaviour", "quality"}


def test_data_xlsx_without_start_zone(tmp_path, cfg):
    cfg.start_zone = None
    path = tmp_path / "data.xlsx"
    write_xlsx(path)
    with pytest.warns(UserWarning):
        long = read_data_xlsx(path, cfg)
    assert "entries" not in set(long.metric)
    raw = long[long.metric == "entries_incl_start"]
    assert set(raw.kind) == {"quality"} and len(raw) == 6
