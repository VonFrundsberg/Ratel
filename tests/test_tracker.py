"""End-to-end: synthetic video with a known path -> tracker -> metrics."""

import cv2
import numpy as np
import pytest

from behaviour.cli import main
from behaviour.metrics import metrics_for_trajectory
from behaviour.trackers.opencv_tracker import TrackerParams, track

FPS = 25
N = 250  # 10 s


def true_path():
    t = np.arange(N) / FPS
    # circle of radius 100 px around the arena centre, one lap in 10 s,
    # then a straight run to the centre would complicate things; keep it simple
    ang = 2 * np.pi * t / 10
    return 320 + 100 * np.cos(ang), 240 + 100 * np.sin(ang)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "rat_sim.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (640, 480))
    assert w.isOpened()
    rng = np.random.default_rng(0)
    base = np.full((480, 640, 3), 90, np.uint8)
    cv2.circle(base, (320, 240), 200, (110, 110, 110), -1)  # arena floor
    cv2.circle(base, (320, 240), 120, (60, 60, 60), 2)      # painted lines
    xs, ys = true_path()
    for x, y in zip(xs, ys):
        f = base.copy()
        cv2.ellipse(f, (int(round(x)), int(round(y))), (18, 9), 0, 0, 360, (235, 235, 235), -1)
        f = cv2.add(f, rng.integers(0, 6, f.shape, dtype=np.uint8))  # sensor noise
        w.write(f)
    w.release()
    return path


def test_tracker_follows_known_path(video, cfg):
    df = track(video, cfg.zones, TrackerParams(), progress=False)
    assert len(df) == N
    assert df["x_px"].notna().mean() > 0.99
    xs, ys = true_path()
    err = np.hypot(df["x_px"] - xs, df["y_px"] - ys)
    assert np.nanmedian(err) < 1.5
    assert np.nanmax(err) < 4
    assert (df["area_px"].dropna() > 300).all()


def test_tracker_metrics_end_to_end(video, cfg):
    cfg.duration_s = None
    df = track(video, cfg.zones, TrackerParams(), progress=False)
    long = metrics_for_trajectory(df, cfg, "rat_sim", "opencv")
    dist = long.query("metric == 'total_distance_m'")["value"].item()
    true_len = 2 * np.pi * 100 * (N - 1) / N / cfg.px_per_m
    assert dist == pytest.approx(true_len, rel=0.03)
    # path radius 100 px lies in the inner ring (47..120) the whole time
    assert long.query("metric == 'time_pct' and zone == 'inner'")["value"].item() == pytest.approx(100)


def test_scene_change_frames_are_lost(tmp_path, cfg):
    """Lights flicker / lid closes: the whole arena changes -> frame lost, not a fake position."""
    path = tmp_path / "flash.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (640, 480))
    base = np.full((480, 640, 3), 110, np.uint8)
    for i in range(40):
        f = base.copy()
        cv2.ellipse(f, (250 + 5 * i, 240), (18, 9), 0, 0, 360, (235, 235, 235), -1)
        if 20 <= i < 23:
            f = cv2.add(f, np.full_like(f, 80))
        w.write(f)
    w.release()
    df = track(path, cfg.zones, TrackerParams(), progress=False)
    assert df.loc[20:22, "x_px"].isna().all()
    assert df.drop(index=[20, 21, 22])["x_px"].notna().all()


def test_cli_pipeline(video, tmp_path):
    from .conftest import ROOT

    cfg_path = str(ROOT / "configs" / "example_open_field.toml")
    tracks = tmp_path / "tracks"
    main(["track", str(video), "-c", cfg_path, "-o", str(tracks), "--qc"])
    assert (tracks / "rat_sim.csv").exists() and (tracks / "rat_sim_qc.mp4").exists()
    main(["metrics", str(tracks), "-c", cfg_path, "--tool", "opencv", "-o", str(tmp_path / "opencv.csv")])
    main(["metrics", str(tracks), "-c", cfg_path, "--tool", "copy", "-o", str(tmp_path / "copy.csv")])
    main(["compare", str(tmp_path / "opencv.csv"), str(tmp_path / "copy.csv"), "-o", str(tmp_path / "agr.csv")])
    main(["preview", str(video), "-c", cfg_path, "-o", str(tmp_path / "zones.png")])
    assert (tmp_path / "zones.png").exists()


def test_dlc_adapter(tmp_path):
    from behaviour.adapters import deeplabcut as dlc

    header = ["scorer,DLC,DLC,DLC,DLC,DLC,DLC", "bodyparts,bodycentre,bodycentre,bodycentre,tl,tl,tl",
              "coords,x,y,likelihood,x,y,likelihood"]
    rows = [f"{i},{10 + i},{20 + i},{0.99 if i != 2 else 0.5},5,6,0.99" for i in range(4)]
    f = tmp_path / "rat01DLC_resnet50_testshuffle1_1000.csv"
    f.write_text("\n".join(header + rows) + "\n")
    df = dlc.trajectory(f, fps=25)
    assert list(df.columns) == ["frame", "time_s", "x_px", "y_px"]
    assert np.isnan(df.x_px[2]) and df.x_px[3] == 13 and df.time_s[1] == 0.04
    np.testing.assert_allclose(dlc.landmarks(f, ["tl"]), [[5, 6]])
    assert dlc.video_id_from_filename(f.stem) == "rat01"
    assert dlc.video_id_from_filename("FST_1_cropDeepCut_resnet50_FST1Aug15shuffle1_1030000") == "FST_1_crop"
    main(["dlc", str(f), "--fps", "25", "-o", str(tmp_path / "out")])
    assert (tmp_path / "out" / "rat01.csv").exists()
