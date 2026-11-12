"""Per-recording square or circular arena calibration for the native TailTrail batch runner.

The geometry follows the four OFT pilots: median tl/tr/br/bl landmarks, mean
side length for scale, axis-aligned centre/periphery rectangles and a circular
mask enclosing the periphery. Without landmarks, fit the floor on temporal
median images and require agreement between the early and late samples.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .adapters.deeplabcut import read_dlc
from .adapters.tailtrail import dump_settings, load_raw_settings, rectangle_contours


class CalibrationError(ValueError):
    """The arena cannot be fitted reliably enough to start unattended tracking."""


def _ordered(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, float).reshape(4, 2)
    angles = np.arctan2(points[:, 1] - points[:, 1].mean(), points[:, 0] - points[:, 0].mean())
    points = points[np.argsort(angles)]
    return np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)


def _validate_corners(corners: np.ndarray, width: int, height: int):
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise CalibrationError("Four finite arena corners are required")
    contour = corners.astype(np.float32)
    if not cv2.isContourConvex(contour):
        raise CalibrationError("Arena corners do not form a convex quadrilateral in tl,tr,br,bl order")
    if not .05 <= cv2.contourArea(contour) / (width * height) <= .85:
        raise CalibrationError("Arena occupies less than 5% or more than 85% of the frame")
    if np.any(corners < 1) or np.any(corners[:, 0] >= width - 1) or np.any(corners[:, 1] >= height - 1):
        raise CalibrationError("Arena corners touch or extend beyond the recording frame")
    edges = np.roll(corners, -1, axis=0) - corners
    sides = np.linalg.norm(edges, axis=1)
    if sides.min() < .35 * sides.max():
        raise CalibrationError("Arena perspective or aspect ratio is too extreme for this square-arena profile")
    cosines = np.sum(edges * np.roll(edges, 1, axis=0), axis=1) / (sides * np.roll(sides, 1))
    if np.max(np.abs(cosines)) > .5:
        raise CalibrationError("Arena corner angles are outside 60–120 degrees")
    if corners[0, 0] >= corners[1, 0] or corners[0, 1] >= corners[3, 1]:
        raise CalibrationError("Landmarks must be ordered top-left, top-right, bottom-right, bottom-left")
    horizontal = np.abs(edges[[0, 2], 1]) / sides[[0, 2]]
    vertical = np.abs(edges[[1, 3], 0]) / sides[[1, 3]]
    if max(horizontal.max(), vertical.max()) > np.sin(np.deg2rad(15)):
        raise CalibrationError("Camera rotation exceeds 15 degrees; TailTrail's axis-aligned ROI approximation needs review")


def _samples(video: Path):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open recording for calibration: {video}")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if min(width, height, count) <= 0:
            raise ValueError("Invalid recording dimensions or frame count")
        factor = min(1.0, 960 / max(width, height))
        shape = (round(width * factor), round(height * factor))
        indices = np.unique(np.linspace(.15 * (count - 1), .85 * (count - 1), 9).astype(int))
        if len(indices) < 5:
            raise CalibrationError("At least five distinct frames are needed for automatic calibration")
        frames = []
        for index in indices:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                raise ValueError(f"Cannot seek to calibration frame {index}")
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode calibration frame {index}")
            frames.append(cv2.resize(frame, shape, interpolation=cv2.INTER_AREA))
        return np.asarray(frames), width, height, indices.tolist()
    finally:
        cap.release()


def _detect_floor(frame: np.ndarray):
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    height, width = gray.shape
    diagonal = np.hypot(width, height)
    candidates = []
    for low, high in ((10, 30), (20, 60), (30, 90)):
        edges = cv2.Canny(gray, low, high)
        distances = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        for size in (3, 5, 7):
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
            contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if not .05 * width * height < area < .85 * width * height:
                    continue
                polygon = cv2.approxPolyDP(contour, .018 * cv2.arcLength(contour, True), True)
                if len(polygon) != 4:
                    continue
                corners = _ordered(polygon)
                try:
                    _validate_corners(corners, width, height)
                except CalibrationError:
                    continue
                quad_area = cv2.contourArea(corners.astype(np.float32))
                fit = min(area, quad_area) / max(area, quad_area)
                if fit < .93:
                    continue
                mask = np.zeros(gray.shape, np.uint8)
                cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
                interior = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
                spread = float(np.diff(np.percentile(gray[interior], [10, 90]))[0])
                border = np.concatenate([np.linspace(a, b, 80) for a, b in zip(corners, np.roll(corners, -1, axis=0))])
                border = np.rint(border).astype(int)
                # Wide-angle cameras curve the floor edges between corners.
                support = float(np.mean(distances[border[:, 1], border[:, 0]] <= max(3, .01 * diagonal)))
                score = .45 * support + .35 * np.exp(-spread / 40) + .20 * fit
                candidates.append(dict(corners=corners, score=float(score), edge_support=support,
                                       intensity_spread=spread, contour_fit=float(fit)))
    if not candidates:
        raise CalibrationError("No closed rectangular arena floor found in the sampled frames")
    candidates.sort(key=lambda c: -c["score"])
    best = candidates[0]
    if best["score"] < .80 or best["edge_support"] < .75:
        raise CalibrationError(f"Arena floor fit is weak (score {best['score']:.3f}; edge support {best['edge_support']:.3f})")
    for other in candidates[1:]:
        difference = np.linalg.norm(best["corners"] - other["corners"], axis=1).max()
        if difference > .02 * diagonal and best["score"] - other["score"] < .035:
            raise CalibrationError("Multiple different arena floors have similar fit scores")
    return best["corners"], {key: value for key, value in best.items() if key != "corners"}


def _image_corners(frames: np.ndarray):
    middle = np.median(frames, axis=0).astype(np.uint8)
    corners, quality = _detect_floor(middle)
    split = len(frames) // 2
    early, _ = _detect_floor(np.median(frames[:split + 1], axis=0).astype(np.uint8))
    late, _ = _detect_floor(np.median(frames[split:], axis=0).astype(np.uint8))
    drift = float(max(np.linalg.norm(early - corners, axis=1).max(), np.linalg.norm(late - corners, axis=1).max()))
    mean_side = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1).mean()
    quality["temporal_corner_shift_px"] = drift
    if drift > max(4, .02 * mean_side):
        raise CalibrationError(f"Arena fit changes across the recording ({drift:.1f} px); camera movement or occlusion needs review")
    return corners, quality


def _landmark_corners(path: Path):
    df = read_dlc(path)
    corners, quality = [], {}
    for name in ("tl", "tr", "br", "bl"):
        if not {(name, "x"), (name, "y"), (name, "likelihood")} <= set(df.columns):
            raise CalibrationError(f"DLC file has no x/y/likelihood columns for arena landmark {name}")
        points = df.loc[:, [(name, "x"), (name, "y")]].to_numpy(float)
        good = (df[(name, "likelihood")].to_numpy(float) > .95) & np.isfinite(points).all(axis=1)
        if good.sum() < max(3, .5 * len(df)):
            raise CalibrationError(f"Landmark {name} has too few detections above likelihood 0.95")
        point = np.median(points[good], axis=0)
        quality[name] = dict(valid_fraction=float(good.mean()),
                             p95_spread_px=float(np.percentile(np.linalg.norm(points[good] - point, axis=1), 95)))
        corners.append(point)
    corners = np.asarray(corners)
    side = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1).mean()
    if max(q["p95_spread_px"] for q in quality.values()) > max(4, .02 * side):
        raise CalibrationError("DLC arena corners are unstable across the recording")
    return corners, quality


def _refine_circle(gray: np.ndarray, center, radius: float):
    """Refine a proposal on the signed floor-to-wall edge, excluding markings.

    Hough circles only seed the search. Measure an edge point on each ray,
    then robustly fit those image points rather than scoring a rounded proposal.
    """
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    radii = radius + np.arange(-10, 10.01, .5)
    xy = np.asarray(center) + unit[:, None, :] * radii[None, :, None]
    def sample(offset):
        points = xy + offset * unit[:, None, :]
        return cv2.remap(gray.astype(np.float32), points[:, :, 0].astype(np.float32),
                         points[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
    gradient = sample(-2) - sample(2)
    peaks = np.argmax(gradient, axis=1)
    strength = gradient[np.arange(len(unit)), peaks]
    good = (strength >= 8) & (peaks > 0) & (peaks < len(radii)-1)
    points = xy[np.arange(len(unit)), peaks]
    for _ in range(5):
        if good.mean() < .80:
            return None
        selected = points[good]
        # Centre the linear system to avoid loss of precision at large pixels.
        origin = selected.mean(axis=0)
        local = selected - origin
        solution = np.linalg.lstsq(np.column_stack((2*local, np.ones(len(local)))),
                                   np.sum(local**2, axis=1), rcond=None)[0]
        fitted_center = origin + solution[:2]
        fitted_radius = float(np.sqrt(solution[2] + np.sum(solution[:2]**2)))
        residual = np.abs(np.linalg.norm(points - fitted_center, axis=1) - fitted_radius)
        good = (strength >= 8) & (peaks > 0) & (peaks < len(radii)-1) & (residual <= 3)
    if good.mean() < .80:
        return None
    return fitted_center, fitted_radius, float(good.mean()), float(np.percentile(residual[good], 90))


def _detect_circle(frame: np.ndarray):
    """Fit a near-circular, bright floor bounded by a darker wall.

    This profile deliberately rejects strong perspective and ambiguous boundaries;
    it does not infer a physical diameter from the image.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    height, width = gray.shape
    edges = cv2.Canny(gray, 20, 60)
    distances = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Hough proposals cover floor edges interrupted by grid markings or glare.
    # Proposals still pass the same image support and wall contrast checks below.
    contours = list(contours)
    # Hough's centre suppression also suppresses concentric circles. Search
    # overlapping radius bands so a painted inner ring cannot hide the floor.
    for lower, upper in ((.20, .31), (.29, .41), (.39, .49)):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                   param1=60, param2=35,
                                   minRadius=round(lower * min(width, height)),
                                   maxRadius=round(upper * min(width, height)))
        if circles is not None:
            for cx, cy, radius in circles[0]:
                contours.append(cv2.ellipse2Poly((round(float(cx)), round(float(cy))),
                                (round(float(radius)), round(float(radius))), 0, 0, 360, 1).reshape(-1, 1, 2))
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates = []
    for contour in contours:
        if len(contour) < 80:
            continue
        (cx, cy), axes, _ = cv2.fitEllipse(contour)
        radius = float(np.mean(axes) / 2)
        if min(axes) / max(axes) < .92 or not .20 * min(width, height) < radius < .49 * min(width, height):
            continue
        if cx-1.06*radius < 2 or cy-1.06*radius < 2 or cx+1.06*radius >= width-2 or cy+1.06*radius >= height-2:
            continue
        points = contour.reshape(-1, 2)
        residual = float(np.percentile(np.abs(np.linalg.norm(points - (cx, cy), axis=1) - radius), 90))
        if residual > max(3, .02 * radius):
            continue
        refined = _refine_circle(gray, (cx, cy), radius)
        if refined is None:
            continue
        (cx, cy), radius, radial_support, residual = refined
        if cx-1.06*radius < 2 or cy-1.06*radius < 2 or cx+1.06*radius >= width-2 or cy+1.06*radius >= height-2:
            continue
        def sample(factor):
            xy = np.rint((cx, cy) + unit * radius * factor).astype(int)
            if np.any(xy < 0) or np.any(xy[:, 0] >= width) or np.any(xy[:, 1] >= height):
                raise CalibrationError("Circular boundary sampling extends outside the image")
            return xy
        rim = sample(1)
        support = float(np.mean(distances[rim[:, 1], rim[:, 0]] <= 3))
        inner, outer = sample(.94), sample(1.06)
        contrast = float(np.median(gray[inner[:, 1], inner[:, 0]].astype(float) - gray[outer[:, 1], outer[:, 0]].astype(float)))
        if support < .80 or contrast < 8:
            continue
        score = support + min(contrast, 100) / 100
        candidates.append(dict(center=[float(cx), float(cy)], radius=radius,
                               score=score, edge_support=support, inside_minus_outside=contrast,
                               radial_edge_support=radial_support,
                               p90_radial_residual_px=residual, axis_ratio=float(min(axes)/max(axes))))
    if not candidates:
        raise CalibrationError("No supported circular bright floor with a darker outer wall found")
    candidates.sort(key=lambda c: -c['score'])
    best = candidates[0]
    for other in candidates[1:]:
        difference = np.linalg.norm(np.array(best['center']) - other['center']) + abs(best['radius'] - other['radius'])
        if difference > 5 and best['score'] - other['score'] < .08:
            raise CalibrationError("Multiple circular boundaries have similar support; review floor versus rim")
    return best


def _image_circle(frames: np.ndarray):
    best = _detect_circle(np.median(frames, axis=0).astype(np.uint8))
    split = len(frames) // 2
    fits = [_detect_circle(np.median(part, axis=0).astype(np.uint8))
            for part in (frames[:split+1], frames[split:])]
    drift = max(float(np.linalg.norm(np.array(fit['center']) - best['center']) +
                      abs(fit['radius'] - best['radius'])) for fit in fits)
    best['temporal_boundary_shift_px'] = drift
    if drift > max(4, .02 * best['radius']):
        raise CalibrationError(f"Circular floor changes across the recording ({drift:.1f} px)")
    return best


def circle_geometry(circle: dict, width: int, height: int, params: dict) -> dict:
    cx, cy = circle['center']
    radius = circle['radius']
    zones = []
    for name, factor in (("Center", params['center_scale']), ("Periphery", params['periphery_scale'])):
        r = round(radius * factor)
        if r <= 0 or cx-r < 0 or cy-r < 0 or cx+r >= width or cy+r >= height:
            raise CalibrationError(f"Calibrated {name} circle extends beyond the recording frame")
        zones.append(dict(name=name, x=round(cx), y=round(cy), radius=r,
                          hysteresis_m=params['hysteresis_m']))
    # TailTrail's native mask is frame-centred, while ROI circles can be offset.
    offset = float(np.hypot(round(cx)-width//2, round(cy)-height//2))
    reach = offset + max(z['radius'] for z in zones)
    return dict(scale_px_per_m=round(2*radius / params['arena_diameter_m'], 2),
                mask_radius=int(np.ceil(reach + params['mask_margin_px'])),
                mask_center=[width//2, height//2], floor_center_offset_px=offset,
                zones=zones)


def _circle_contours(x, y, radius, hysteresis_px):
    def contour(r):
        points = cv2.ellipse2Poly((x, y), (r, r), 0, 0, 360, 1)
        return (points.reshape(-1, 1, 2).astype(np.int32),)
    return contour(radius), contour(radius + hysteresis_px)


def geometry(corners: np.ndarray, width: int, height: int, params: dict) -> dict:
    _validate_corners(corners, width, height)
    side = np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1))
    scale = round(float(side / params["arena_side_m"]), 2)
    if scale <= 0:
        raise CalibrationError("Calculated distance scale is not positive")
    centroid, zones, outer_corners = corners.mean(axis=0), [], []
    for name, factor in (("Center", params["center_scale"]), ("Periphery", params["periphery_scale"])):
        # Match the pilot's TOML rounding and PolygonZone float32 storage
        # before doing its rectangle arithmetic in float64.
        points = np.round(centroid + factor * (corners - centroid), 1).astype(np.float32).astype(float)
        tl, tr, br, bl = points
        left, right = (tl[0] + bl[0]) / 2, (tr[0] + br[0]) / 2
        top, bottom = (tl[1] + tr[1]) / 2, (bl[1] + br[1]) / 2
        x, y, w, h = round(left), round(top), round(right - left), round(bottom - top)
        if x < 0 or y < 0 or x + w >= width or y + h >= height:
            raise CalibrationError(f"Calibrated {name} ROI would extend beyond the recording frame")
        zones.append(dict(name=name, x=x, y=y, width=w, height=h, hysteresis_m=params["hysteresis_m"]))
        if name == "Periphery":
            outer_corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    reach = max(np.hypot(x - width / 2, y - height / 2) for x, y in outer_corners)
    return dict(scale_px_per_m=scale, mask_radius=int(np.ceil(reach + params["mask_margin_px"])), zones=zones)


def _preview(frame: np.ndarray, report: dict, output: Path):
    image = frame.copy()
    factor = np.array([image.shape[1] / report["width"], image.shape[0] / report["height"]])
    if "corners" in report:
        points = np.rint(np.asarray(report["corners"]) * factor).astype(np.int32)
        cv2.polylines(image, [points], True, (255, 255, 0), 2)
    if "circle" in report:
        circle = report["circle"]
        cv2.circle(image, tuple(np.rint(np.array(circle["center"]) * factor).astype(int)),
                   round(circle["radius"] * factor.mean()), (255, 255, 0), 2)
    if "geometry" in report:
        for zone, color in zip(report["geometry"]["zones"], ((0, 0, 255), (0, 255, 128))):
            a = tuple(np.rint(np.array([zone["x"], zone["y"]]) * factor).astype(int))
            if "radius" in zone:
                cv2.circle(image, a, round(zone["radius"] * factor.mean()), color, 2)
                cv2.putText(image, zone["name"], a, cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                continue
            b = tuple(np.rint(np.array([zone["x"] + zone["width"], zone["y"] + zone["height"]]) * factor).astype(int))
            cv2.rectangle(image, a, b, color, 2)
            cv2.putText(image, zone["name"], a, cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        radius = round(report["geometry"]["mask_radius"] * factor.mean())
        cv2.circle(image, (image.shape[1] // 2, image.shape[0] // 2), radius, (0, 220, 255), 1)
    label = "READY" if report["accepted"] else "NEEDS REVIEW"
    cv2.putText(image, f"{label}: {report['source']}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65,
                (0, 230, 0) if report["accepted"] else (0, 0, 255), 2)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("Could not encode calibration preview")
    encoded.tofile(str(output / "calibration_preview.png"))


def calibrate(video: Path, template: Path, output: Path, params: dict, job_id: str) -> dict:
    raw = load_raw_settings(template)
    if not isinstance(raw, dict) or not raw.get("rois"):
        raise ValueError("Calibration needs a valid saved TailTrail settings template with an ROI")
    frames, width, height, indices = _samples(video)
    background = np.median(frames, axis=0).astype(np.uint8)
    report = dict(job_id=job_id, accepted=False, source="dlc" if params["landmarks_file"] else "image",
                  video=str(video), width=width, height=height, sample_frames=indices, parameters=params)
    try:
        if params.get("arena_shape", "square") == "circle":
            circle = _image_circle(frames)
            factor = width / frames.shape[2]
            circle["center"] = (np.array(circle["center"]) * factor).tolist()
            circle["radius"] *= factor
            report.update(circle=circle, quality=circle.copy())
            report["geometry"] = circle_geometry(circle, width, height, params)
        elif params["landmarks_file"]:
            corners, quality = _landmark_corners(Path(params["landmarks_file"]))
        else:
            corners, quality = _image_corners(frames)
            factor = np.array([width / frames.shape[2], height / frames.shape[1]])
            corners *= factor
        if params.get("arena_shape", "square") != "circle":
            report.update(corners=corners.tolist(), quality=quality)
            report["geometry"] = geometry(corners, width, height, params)
        rois = []
        for zone, color in zip(report["geometry"]["zones"], ((255, 0, 0), (0, 255, 128))):
            roi = copy.deepcopy(raw["rois"][0])
            roi.name, roi.type, roi.visible = zone["name"], "Rectangle", True
            roi.color, roi.color_code = color, "#%02x%02x%02x" % color
            if "radius" in zone:
                roi.type = "Circle"
                roi.x, roi.y, roi.radius = (zone[key] for key in ("x", "y", "radius"))
                roi.hysteresis = params["hysteresis_m"]
                roi.contour, roi.hysteresis_contour = _circle_contours(
                    roi.x, roi.y, roi.radius, int(roi.hysteresis * report["geometry"]["scale_px_per_m"]))
                rois.append(roi)
                continue
            roi.x, roi.y, roi.width, roi.height = (zone[key] for key in ("x", "y", "width", "height"))
            roi.hysteresis = params["hysteresis_m"]
            roi.contour, roi.hysteresis_contour = rectangle_contours(
                roi.x, roi.y, roi.width, roi.height, int(roi.hysteresis * report["geometry"]["scale_px_per_m"]))
            rois.append(roi)
        raw.update(rois=rois, scale=report["geometry"]["scale_px_per_m"],
                   mask_radius=report["geometry"]["mask_radius"], results_path=output.as_posix(),
                   translate_frame_x=0, translate_frame_y=0, rotate_frame=0)
        dump_settings(raw, output / "settings.pickle")
        report["settings_sha256"] = hashlib.sha256((output / "settings.pickle").read_bytes()).hexdigest()
        report["accepted"] = True
    except CalibrationError as exc:
        report["reason"] = str(exc)
    _preview(background, report, output)
    temporary = output / "calibration.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "calibration.json")
    return report

