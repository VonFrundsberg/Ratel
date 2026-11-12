"""Additional single-rat tracking backends. All positions use source-frame pixels."""
from pathlib import Path
import importlib.util
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd


def eligible(mask, x, y):
    ix, iy = round(x), round(y)
    return 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix] != 0


def tracktor_frames(video, mask, assets, lab_profile=False):
    source = assets / "tracktor_engine.py"
    spec = importlib.util.spec_from_file_location("ratel_tracktor_engine", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cap = cv2.VideoCapture(str(video))
    rows = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Upstream looks for dark animals. Invert luminance for the light coat.
            target = frame
            if lab_profile:
                # Warm, light coat against the cool grey floor and specular glare.
                red = frame[:, :, 2].astype(np.int16)
                warmth = red - frame[:, :, 0]
                channel = np.clip(100 + 4 * warmth, 0, 255).astype(np.uint8)
                target = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
            binary = module.colour_to_thresh(cv2.bitwise_not(target), block_size=71, offset=28)
            if lab_profile:
                binary[(red < 130) | (warmth < 2)] = 0
            binary = cv2.bitwise_and(binary, mask)
            _, contours, _, points = module.detect_and_draw_contours(frame, binary, [], [], 250, 2400)
            if points:
                index = max(range(len(points)), key=lambda i: cv2.contourArea(contours[i]))
                x, y = points[index]
            else:
                x = y = np.nan
            rows.append((len(rows), x, y, len(points)))
    finally:
        cap.release()
    return pd.DataFrame(rows, columns=["frame", "x_px", "y_px", "candidates"])


def fasttrack_frames(video, mask, assets, scratch, frames, lab_profile=False):
    exe = assets / "fasttrack" / "FastTrack.exe"
    with tempfile.TemporaryDirectory(dir=scratch) as folder:
        # FastTrack always writes a database beside its input. Keep it off the raw volume.
        local = Path(folder) / ("input" + video.suffix)
        shutil.copyfile(video, local)
        # Some lab MP4 headers overstate frame count. FastTrack's default
        # background sampler then seeks beyond EOF and aborts before tracking.
        cap = cv2.VideoCapture(str(video))
        samples = []
        try:
            for i in np.linspace(0, max(0, frames - 2), min(51, frames), dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Cannot read FastTrack background frame {i}")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                samples.append(gray)
        finally:
            cap.release()
        if lab_profile:
            background = np.median(samples, axis=0).astype(np.uint8)
        else:
            background = np.min(samples, axis=0)
        bg_path = Path(folder) / "sampled.pgm"
        if not cv2.imwrite(str(bg_path), background):
            raise OSError("Could not save FastTrack background")
        args = ["--backPath", str(bg_path), "--maxArea", "2400", "--minArea", "250", "--lightBack", "1", "--thresh", "30",
                "--reg", "0", "--spot", "2", "--normDist", "40", "--normAngle", "45",
                "--normArea", "0", "--normPerim", "0", "--maxDist", "120", "--maxTime", "10",
                "--nBack", "51", "--methBack", "0", "--regBack", "0", "--xTop", "0", "--yTop", "0",
                "--xBottom", "0", "--yBottom", "0", "--morph", "3", "--morphSize", "1", "--morphType", "2"]
        result = subprocess.run([str(exe), "--cli", "--path", str(local), *args],
                                cwd=folder, env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
                                capture_output=True, text=True, errors="replace", check=True)
        exports = list(Path(folder).rglob("tracking.txt"))
        message = (result.stdout + result.stderr).strip()
        # FastTrack 6.7 warns about the one non-existent trailing header frame.
        trailing_only = f"Images {frames}, where skipped because unreadable."
        last_line = " ".join(message.splitlines()[-1].split()) if message else ""
        if len(exports) != 1 or ("Tracking ended normally" not in message and last_line != trailing_only):
            raise RuntimeError("Incomplete FastTrack export: " + (result.stdout + result.stderr)[-500:])
        raw = pd.read_csv(exports[0], sep="\t")
    if not raw.imageNumber.between(0, frames - 1).all():
        raise ValueError("FastTrack frame indices exceed decoded video")
    raw = raw[[eligible(mask, x, y) for x, y in zip(raw.xBody, raw.yBody)]] if len(raw) else raw
    counts = raw.groupby("imageNumber").size()
    largest = raw.sort_values("areaBody", ascending=False, kind="stable").drop_duplicates("imageNumber")
    df = largest.set_index("imageNumber")[["xBody", "yBody", "id"]].reindex(range(frames))
    df.columns = ["x_px", "y_px", "object_id"]
    df["candidates"] = counts.reindex(df.index, fill_value=0)
    return df.rename_axis("frame").reset_index()


def superanimal_frames(video, mask, assets, stride):
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.transforms.functional import normalize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None,
                                             num_classes=2, box_score_thresh=0.6)
    state = torch.load(assets / "superanimal.pt", weights_only=True, map_location="cpu")["model"]
    net.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)
    net.to(device).eval()
    left, top, width, height = cv2.boundingRect(mask)
    cap = cv2.VideoCapture(str(video))
    rows = []
    try:
        with torch.inference_mode():
            frame = 0
            while cap.grab():
                if frame % stride == 0:
                    ok, im = cap.retrieve()
                    if not ok:
                        raise ValueError(f"Cannot decode frame {frame}")
                    crop = im[top:top+height, left:left+width]
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device).float() / 255
                    tensor = normalize(tensor, [.485, .456, .406], [.229, .224, .225])
                    pred = net([tensor])[0]
                    hits = []
                    for box, score in zip(pred["boxes"].cpu().tolist(), pred["scores"].cpu().tolist()):
                        x = left + (box[0] + box[2]) / 2
                        y = top + (box[1] + box[3]) / 2
                        if eligible(mask, x, y):
                            hits.append((x, y, score))
                    x, y, score = hits[0] if hits else (np.nan, np.nan, np.nan)
                    rows.append((frame, x, y, score, len(hits)))
                frame += 1
    finally:
        cap.release()
    return pd.DataFrame(rows, columns=["frame", "x_px", "y_px", "score", "candidates"])
