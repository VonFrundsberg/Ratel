import cv2
import numpy as np
import pandas as pd
import pytest

from behaviour.cli import main
from behaviour.config import FstParams, load_config
from behaviour.fst import fst_metrics, predominant_bins, union
from behaviour.trackers.opencv_tracker import TrackerParams, track

from .conftest import ROOT

FPS = 10.0
AREA = 1000.0
STILL, MOVING = 10.0, 300.0  # motion_px; ratio 0.01 vs 0.3 against the fixture's 0.05 threshold


def frames(motion, fps=FPS, area=AREA):
    n = len(motion)
    return pd.DataFrame({"frame": np.arange(n), "time_s": np.arange(n) / fps, "x_px": 320.0, "y_px": 240.0,
                         "area_px": area, "motion_px": motion})


def get(long, metric, zone=""):
    v = long[(long.metric == metric) & (long.zone == zone)]["value"]
    assert len(v) == 1, (metric, zone)
    return float(v.iloc[0])


@pytest.fixture
def fcfg(cfg):
    cfg.duration_s = None
    cfg.fst = FstParams(motion_threshold=0.05, smooth_s=0.0, min_immobile_s=1.0, bin_s=5.0)
    return cfg


def test_immobility_bouts_latency_bins(fcfg):
    # 7 s moving, 0.5 s still (too short), 2.5 s moving, 10 s still, 5 s moving  (25 s)
    m = np.r_[np.full(70, MOVING), np.full(5, STILL), np.full(25, MOVING), np.full(100, STILL), np.full(50, MOVING)]
    long = fst_metrics(frames(m), fcfg, "rat", "opencv")
    assert get(long, "duration_s") == pytest.approx(25.0)
    assert get(long, "fst_immobile_s") == pytest.approx(10.0)
    assert get(long, "fst_immobility_latency_s") == pytest.approx(10.0)
    # bins 0-5, 5-10 mobile; 10-15, 15-20 immobile; 20-25 mobile
    assert get(long, "fst_immobile_bins") == 2
    assert set(long.tool) == {"opencv"}
    assert set(long.kind) == {"behaviour", "quality"}


def test_threshold_is_relative_to_area(fcfg):
    m = np.full(100, 40.0)  # 40 changed px: still for a 1000 px animal, moving for a 400 px one
    assert get(fst_metrics(frames(m, area=1000.0), fcfg, "a"), "fst_immobile_s") == pytest.approx(10.0)
    assert get(fst_metrics(frames(m, area=400.0), fcfg, "a"), "fst_immobile_s") == 0


def test_missing_motion_is_not_still(fcfg):
    m = np.full(100, STILL)
    m[0] = np.nan          # first frame never has motion
    m[50:55] = np.nan      # lost detection splits the bout
    long = fst_metrics(frames(m), fcfg, "rat")
    assert get(long, "fst_immobile_s") == pytest.approx(9.4)
    assert get(long, "fst_immobility_latency_s") == pytest.approx(0.1)
    assert get(long, "tracked_fraction") == pytest.approx(0.94)


def test_smoothing_bridges_single_noisy_frames(fcfg):
    m = np.full(100, STILL)
    m[5::10] = 200.0  # one noisy frame per second breaks every bout below 1 s
    assert get(fst_metrics(frames(m), fcfg, "rat"), "fst_immobile_s") == 0
    fcfg.fst.smooth_s = 0.5  # 5-frame mean: (200 + 4 * 10) / 5 / 1000 = 0.048 < 0.05
    assert get(fst_metrics(frames(m), fcfg, "rat"), "fst_immobile_s") == pytest.approx(10.0)


def test_window_from_config(fcfg):
    # FST protocols often score only the last 4 of 6 min: here 6 s still, 4 s moving, 10 s still
    fcfg.start_s, fcfg.duration_s = 5.0, 10.0
    m = np.r_[np.full(60, STILL), np.full(40, MOVING), np.full(100, STILL)]
    long = fst_metrics(frames(m), fcfg, "rat")
    assert get(long, "duration_s") == pytest.approx(10.0)
    # window 5-15 s: 1 s still (counts: the bout is cut by the window but lasts >= 1 s), 4 s moving, 5 s still
    assert get(long, "fst_immobile_s") == pytest.approx(6.0)
    assert get(long, "fst_immobility_latency_s") == 0
    assert get(long, "fst_immobile_bins") == 1


def test_bins_disabled(fcfg):
    fcfg.fst.bin_s = 0
    assert "fst_immobile_bins" not in set(fst_metrics(frames(np.full(100, STILL)), fcfg, "rat").metric)


def test_predominant_bins():
    states = {"immobile": [(0.0, 3.0), (5.0, 7.0)], "swim": [(3.0, 5.0), (7.0, 9.0)]}
    # bin 1: 3 s vs 2 s; bin 2: 2 s vs 2 s (tie -> first listed) vs 1 s other; 10-12 s is not a full bin
    assert predominant_bins(states, 12.0, 5.0) == {"immobile": 2, "swim": 0}
    # 1 s of immobility in 5 s otherwise uncoded / mobile is not an immobile bin
    assert predominant_bins({"immobile": [(0.0, 1.0)]}, 5.0, 5.0) == {"immobile": 0}
    assert union([(0, 1), (1, 2), (3, 4), (3.5, 3.6)]) == [(0, 2), (3, 4)]


def test_fst_config_section(tmp_path):
    text = (ROOT / "configs" / "example_open_field.toml").read_text(encoding="utf-8")
    p = tmp_path / "fst.toml"
    p.write_text(text + "\n[fst]\nmotion_threshold = 0.1\nmin_immobile_s = 2\n", encoding="utf-8")
    c = load_config(p)
    assert c.fst.motion_threshold == 0.1 and c.fst.min_immobile_s == 2.0 and c.fst.bin_s == 5.0
    p.write_text(text + "\n[fst]\nthreshold = 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown \\[fst\\]"):
        load_config(p)


def test_cli_fst(tmp_path):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    frames(np.r_[np.full(30, MOVING), np.full(70, STILL)]).to_csv(tracks / "rat_01.csv", index=False)
    frames(np.full(100, MOVING)).to_csv(tracks / "rat_02.csv", index=False)
    cfg_path = tmp_path / "fst.toml"
    cfg_path.write_text((ROOT / "configs" / "example_open_field.toml").read_text(encoding="utf-8")
                        + "\n[fst]\nmotion_threshold = 0.05\nsmooth_s = 0\n", encoding="utf-8")
    out = tmp_path / "fst.csv"
    main(["fst", str(tracks), "-c", str(cfg_path), "-o", str(out)])
    long = pd.read_csv(out, keep_default_na=False, na_values=[""])
    long["zone"] = long["zone"].fillna("")
    assert get(long[long.video_id == "rat_01"], "fst_immobile_s") == pytest.approx(7.0)
    assert get(long[long.video_id == "rat_02"], "fst_immobile_s") == 0
    assert np.isnan(get(long[long.video_id == "rat_02"], "fst_immobility_latency_s"))


def test_tracker_motion_end_to_end(tmp_path, fcfg):
    """Synthetic swim video: the animal paddles round the arena for 5 s, then floats still for 3 s.

    It must not sit still for most of the video, or the median background absorbs it
    (use an empty-arena --background for real FST videos)."""
    path = tmp_path / "fst_sim.avi"
    fps, n_move, n = 25, 125, 200
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
    rng = np.random.default_rng(1)
    base = np.full((480, 640, 3), 90, np.uint8)
    cv2.circle(base, (320, 240), 200, (110, 110, 110), -1)
    for i in range(n):
        f = base.copy()
        k = min(i, n_move - 1)
        ang = 2 * np.pi * k / n_move
        wiggle = 25 * np.sin(i * 0.9) if i < n_move else 0
        centre = (int(round(320 + 100 * np.cos(ang))), int(round(240 + 100 * np.sin(ang))))
        cv2.ellipse(f, centre, (18, 9), np.degrees(ang) + 90 + wiggle, 0, 360, (235, 235, 235), -1)
        w.write(cv2.add(f, rng.integers(0, 6, f.shape, dtype=np.uint8)))
    w.release()
    df = track(path, fcfg.zones, TrackerParams(), progress=False)
    assert df["motion_px"].notna().mean() > 0.95
    long = fst_metrics(df, fcfg, "fst_sim")
    assert get(long, "fst_immobile_s") == pytest.approx(3.0, abs=0.2)
    assert get(long, "fst_immobility_latency_s") == pytest.approx(5.0, abs=0.2)
