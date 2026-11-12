import csv

import numpy as np
import pandas as pd
import pytest

from behaviour.adapters.boris import read_boris, read_events
from behaviour.cli import main
from behaviour.metrics import metrics_for_trajectory

from .conftest import ROOT

AGG_COLUMNS = ["Observation id", "Observation date", "Description", "Observation type", "Source",
               "Media duration (s)", "FPS", "Subject", "Behavior", "Behavioral category", "Behavior type",
               "Start (s)", "Stop (s)", "Duration (s)", "Media file name", "Comment start", "Comment stop"]


def aggregated(path, bouts, media="rat_01.avi", obs="obs 1", media_dur=10.0):
    """BORIS 8 aggregated events export; bouts = [(behaviour, start, stop), ...]."""
    rows = [[obs, "2026-10-01", "", "Media file", f"player #1:{media}", media_dur, 25.0, "No focal subject",
             b, "", "STATE", a, e, e - a, media, "", ""] for b, a, e in bouts]
    df = pd.DataFrame(rows, columns=AGG_COLUMNS)
    if str(path).endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def tabular_v7(path, events, media=r"C:\videos\rat_07.avi", total_length=120.0):
    """BORIS 7 tabular events export (metadata block above the table); events = [(time, behaviour, status)]."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows([["Observation id", "rat 07 FST"], [], ["Media file(s)"], [], ["Player #1", media], [],
                     ["Observation date", "2026-10-01 10:00:00"], ["Description", ""],
                     ["Time offset (s)", "0.000"], [], []])
        w.writerow(["Time", "Media file path", "Total length", "FPS", "Subject", "Behavior",
                    "Behavioral category", "Comment", "Status"])
        for t, b, s in events:
            w.writerow([f"{t:.3f}", media, total_length, 25, "", b, "", "", s])
    return path


def get(long, metric, zone="", vid=None):
    sel = (long.metric == metric) & (long.zone == zone)
    if vid is not None:
        sel &= long.video_id == vid
    v = long[sel]["value"]
    assert len(v) == 1, (metric, zone, vid)
    return float(v.iloc[0])


def test_zone_metrics_match_trajectory_definitions(tmp_path, cfg):
    """Same visit sequence coded in BORIS and as a trajectory gives identical zone metrics."""
    cfg.duration_s = None
    cfg.smooth_s = 0
    cfg.max_speed_m_s = 100
    for z in cfg.zones:
        z.hysteresis_px = 0
    # 3 s center, 2 s inner, 2 s outer, 3 s center
    p = aggregated(tmp_path / "agg.csv", [("Center", 0, 3), ("Inner", 3, 5), ("Outer", 5, 7), ("Center", 7, 10)])
    manual = read_boris([p], cfg)
    assert set(manual.video_id) == {"rat_01"} and set(manual.tool) == {"manual"}

    x = np.r_[np.full(30, 320.0), np.full(20, 400.0), np.full(20, 480.0), np.full(30, 320.0)]
    traj = pd.DataFrame({"time_s": np.arange(100) / 10, "x_px": x, "y_px": 240.0})
    auto = metrics_for_trajectory(traj, cfg, "rat_01", "opencv")

    key = ["metric", "zone"]
    both = manual.merge(auto, on=key, suffixes=("_manual", "_auto"))
    assert set(both.metric) == {"duration_s", "unassigned_time_s", "time_s", "time_pct", "entries", "latency_s"}
    np.testing.assert_allclose(both.value_manual, both.value_auto, atol=1e-9)
    assert get(manual, "entries", "center") == 1  # start zone not counted
    assert get(manual, "latency_s", "outer") == 5
    assert "distance_m" not in set(manual.metric)


def test_zone_map_window_and_gaps(tmp_path, cfg):
    cfg.start_s, cfg.duration_s = 2.0, 10.0
    p = aggregated(tmp_path / "agg.csv", [("Centre zone", 0, 5), ("Centre zone", 6, 8), ("inner", 8, 12),
                                          ("outer", 12, 20)], media_dur=20.0)
    long = read_boris([p], cfg, zone_map={"Centre zone": "center"})
    # window 2-12 s: center 2-5 and 6-8 (uncoded 5-6), inner 8-12
    assert get(long, "duration_s") == 10
    assert get(long, "time_s", "center") == pytest.approx(5.0)
    assert get(long, "time_pct", "center") == pytest.approx(50.0)
    assert get(long, "time_s", "inner") == pytest.approx(4.0)
    assert get(long, "time_s", "outer") == 0
    assert get(long, "unassigned_time_s") == pytest.approx(1.0)
    assert get(long, "entries", "center") == 0  # start zone, and the return after the gap is not an entry
    assert get(long, "entries", "inner") == 1
    assert get(long, "latency_s", "center") == 0
    assert get(long, "latency_s", "inner") == pytest.approx(6.0)
    assert np.isnan(get(long, "latency_s", "outer"))


def test_fst_tabular_v7(tmp_path, cfg):
    events = [(0, "Swimming", "START"), (30, "Swimming", "STOP"), (30, "Climbing", "START"),
              (45, "Head shake", "POINT"), (60, "Climbing", "STOP"), (60, "Float", "START"),
              (100, "Float", "STOP"), (100, "Swimming", "START"), (103, "Swimming", "STOP"),
              (103, "Float", "START"), (120, "Float", "STOP")]
    p = tabular_v7(tmp_path / "tab.csv", events)
    ev = read_events(p)
    assert set(ev.video_id) == {"rat_07"}  # media path stem, Windows separators
    assert len(ev) == 5
    assert set(read_events(p, video_id="observation").video_id) == {"rat 07 FST"}

    long = read_boris([p], cfg, fst_map={"Float": "immobile"})
    assert get(long, "duration_s") == 120  # media is shorter than the 300 s window
    assert get(long, "fst_immobile_s") == pytest.approx(57.0)
    assert get(long, "fst_swim_s") == pytest.approx(33.0)
    assert get(long, "fst_climb_s") == pytest.approx(30.0)
    assert get(long, "fst_immobility_latency_s") == pytest.approx(60.0)
    assert get(long, "fst_uncoded_s") == pytest.approx(0.0)
    # 24 bins: 0-30 swim, 30-60 climb, 60-100 immobile, 100-105 swim (3 s vs 2 s), 105-120 immobile
    assert get(long, "fst_immobile_bins") == 11
    assert get(long, "fst_swim_bins") == 7
    assert get(long, "fst_climb_bins") == 6
    assert "time_s" not in set(long.metric)  # no zone coding in this file
    assert set(long.kind) == {"behaviour", "quality"}


def test_only_immobility_coded_xlsx(tmp_path, cfg):
    cfg.duration_s = 20.0
    a = aggregated(tmp_path / "a.xlsx", [("Immobility", 3, 5), ("Immobility", 12, 20)],
                   media=r"D:\fst\rat_01.mp4", media_dur=30.0)
    b = aggregated(tmp_path / "b.xlsx", [], media="rat_02.mp4", media_dur=30.0)
    b2 = aggregated(tmp_path / "b2.csv", [("Grooming", 1, 2)], media="rat_02.mp4", media_dur=30.0)
    with pytest.warns(UserWarning, match="Grooming"):
        long = read_boris([a, b, b2], cfg)
    assert get(long, "fst_immobile_s", vid="rat_01") == pytest.approx(10.0)
    assert get(long, "fst_immobility_latency_s", vid="rat_01") == pytest.approx(3.0)
    assert get(long, "fst_uncoded_s", vid="rat_01") == pytest.approx(10.0)
    # bins: 0-5 has 2 s immobile vs 3 s uncoded -> not immobile; 10-15 3 vs 2 -> immobile; 15-20 immobile
    assert get(long, "fst_immobile_bins", vid="rat_01") == 2
    assert not {"fst_swim_s", "fst_climb_s", "fst_swim_bins"} & set(long.metric)
    # rat_02 has only an unmapped behaviour: duration but no FST metrics
    assert set(long[long.video_id == "rat_02"].metric) == {"duration_s"}


def test_mapping_errors_and_unpaired(tmp_path, cfg):
    p = aggregated(tmp_path / "agg.csv", [("Center", 0, 3)])
    with pytest.raises(ValueError, match="not zones"):
        read_boris([p], cfg, zone_map={"Center": "middle"})
    with pytest.raises(ValueError, match="not FST states"):
        read_boris([p], cfg, fst_map={"Float": "floating"})
    t = tabular_v7(tmp_path / "tab.csv", [(10, "Immobility", "START")], total_length=60.0)
    with pytest.warns(UserWarning, match="never stopped"):
        long = read_boris([t], cfg)
    assert get(long, "fst_immobile_s") == pytest.approx(50.0)


def test_cli_boris_then_compare(tmp_path):
    cfg_path = ROOT / "configs" / "example_open_field.toml"
    exports = tmp_path / "boris"
    exports.mkdir()
    for i, imm in enumerate([60.0, 120.0, 200.0], 1):
        aggregated(exports / f"rat_{i:02d}.csv", [("Swimming", 0, 300 - imm), ("Immobility", 300 - imm, 300)],
                   media=f"rat_{i:02d}.avi", media_dur=300.0)
    manual = tmp_path / "manual.csv"
    main(["boris", str(exports), "-c", str(cfg_path), "-o", str(manual)])
    long = pd.read_csv(manual, dtype={"video_id": str}, keep_default_na=False, na_values=[""])
    long["zone"] = long["zone"].fillna("")
    assert get(long, "fst_immobile_s", vid="rat_02") == pytest.approx(120.0)

    auto = long[long.metric == "fst_immobile_s"].assign(tool="opencv")
    auto["value"] = auto["value"] + 5.0
    auto_path = tmp_path / "opencv_fst.csv"
    auto.to_csv(auto_path, index=False)
    out = tmp_path / "agreement.csv"
    main(["compare", str(manual), str(auto_path), "--reference", "manual", "-o", str(out)])
    res = pd.read_csv(out)
    row = res[res.metric == "fst_immobile_s"].iloc[0]
    assert row["n"] == 3 and row["bias"] == pytest.approx(5.0)
