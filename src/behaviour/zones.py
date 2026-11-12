"""Zone geometry and zone assignment with hysteresis.

Zones follow TailTrail's convention: they are kept in a priority list and a
point belongs to the *first* zone that contains it.  Nested zones therefore
become exclusive rings when the smaller zone is listed first
(center -> inner -> outer).

Hysteresis (band width ``h`` px) suppresses flicker at a boundary.  Where the
band sits is set by ``mode``:

* ``outer``     enter at the border, leave only once ``h`` outside it
                (our reading of TailTrail's manual: "both lines must be crossed")
* ``inner``     enter only once ``h`` inside, leave at the border
* ``symmetric`` enter at ``h/2`` inside, leave at ``h/2`` outside
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Zone:
    name: str
    hysteresis_px: float = 0.0

    def signed_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Distance to the zone border in px, positive inside, negative outside."""
        raise NotImplementedError

    def draw(self, img: np.ndarray, color, thickness: int = 1, margin_px: float = 0.0) -> None:
        raise NotImplementedError

    def contains(self, x, y) -> np.ndarray:
        return self.signed_distance(np.asarray(x, float), np.asarray(y, float)) >= 0


@dataclass
class CircleZone(Zone):
    cx: float = 0.0
    cy: float = 0.0
    r: float = 0.0

    def signed_distance(self, x, y):
        return self.r - np.hypot(np.asarray(x, float) - self.cx, np.asarray(y, float) - self.cy)

    def draw(self, img, color, thickness=1, margin_px=0.0):
        cv2.circle(img, (round(self.cx), round(self.cy)), max(round(self.r + margin_px), 0), color, thickness)


@dataclass
class RectangleZone(Zone):
    """Axis-aligned rectangle given by its top-left corner."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def signed_distance(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        dx = np.maximum(self.x - x, x - (self.x + self.width))
        dy = np.maximum(self.y - y, y - (self.y + self.height))
        outside = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
        inside = np.minimum(np.maximum(dx, dy), 0)
        return -(outside + inside)

    def draw(self, img, color, thickness=1, margin_px=0.0):
        p0 = (round(self.x - margin_px), round(self.y - margin_px))
        p1 = (round(self.x + self.width + margin_px), round(self.y + self.height + margin_px))
        cv2.rectangle(img, p0, p1, color, thickness)


@dataclass
class PolygonZone(Zone):
    """Arbitrary polygon, e.g. a TailTrail quadrant exported as its contour."""

    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))

    def __post_init__(self):
        self.points = np.asarray(self.points, np.float32).reshape(-1, 2)

    def signed_distance(self, x, y):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        contour = self.points.reshape(-1, 1, 2)
        out = np.full(x.shape, np.nan)
        ok = np.isfinite(x) & np.isfinite(y)
        for i in np.flatnonzero(ok):
            out[i] = cv2.pointPolygonTest(contour, (float(x[i]), float(y[i])), True)
        return out

    def draw(self, img, color, thickness=1, margin_px=0.0):
        pts = self.points.astype(np.int32).reshape(-1, 1, 2)
        if thickness < 0:
            cv2.fillPoly(img, [pts], color)
            if margin_px > 0:
                k = 2 * round(margin_px) + 1
                img[:] = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        else:
            cv2.polylines(img, [pts], True, color, thickness)


HYSTERESIS_MODES = {"outer": (0.0, -1.0), "inner": (1.0, 0.0), "symmetric": (0.5, -0.5)}


def assign_zones(zones: list[Zone], x: np.ndarray, y: np.ndarray, mode: str = "outer") -> np.ndarray:
    """Return the zone index for each frame (-1 = outside every zone).

    Frames with NaN positions keep the previous zone (the animal is assumed to
    still be where it was last seen); leading NaNs take the first known zone.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    out = np.full(n, -1, dtype=int)
    if n == 0 or not zones:
        return out
    enter_f, stay_f = HYSTERESIS_MODES[mode]
    sd = np.vstack([z.signed_distance(x, y) for z in zones])  # (n_zones, n_frames)
    h = np.array([z.hysteresis_px for z in zones])[:, None]
    deep_in = sd >= enter_f * h
    loosely_in = sd >= stay_f * h
    plain_in = sd >= 0
    valid = np.isfinite(x) & np.isfinite(y)

    current = None
    for i in range(n):
        if not valid[i]:
            out[i] = -1 if current is None else current
            continue
        new = None
        for k in range(len(zones)):
            if k == current:
                if loosely_in[k, i]:
                    new = k
                    break
            elif deep_in[k, i]:
                new = k
                break
        if new is None:
            hits = np.flatnonzero(plain_in[:, i])
            new = int(hits[0]) if hits.size else -1
        current = None if new == -1 else new
        out[i] = new

    first = np.flatnonzero(valid)
    if first.size:
        out[: first[0]] = out[first[0]]
    return out


def arena_mask(zones: list[Zone], shape: tuple[int, int], margin_px: float = 20.0) -> np.ndarray:
    """Binary mask (uint8 0/255) of everything within ``margin_px`` of any zone."""
    mask = np.zeros(shape[:2], np.uint8)
    for z in zones:
        layer = np.zeros_like(mask)
        z.draw(layer, 255, thickness=-1, margin_px=margin_px)
        mask |= layer
    return mask
