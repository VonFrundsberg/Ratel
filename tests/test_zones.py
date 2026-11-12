import numpy as np
import pytest

from behaviour.zones import CircleZone, PolygonZone, RectangleZone, assign_zones


def test_circle_signed_distance():
    z = CircleZone("c", cx=0, cy=0, r=10)
    np.testing.assert_allclose(z.signed_distance([0, 10, 13], [0, 0, 0]), [10, 0, -3])


def test_rectangle_signed_distance():
    z = RectangleZone("r", x=0, y=0, width=10, height=4)
    np.testing.assert_allclose(z.signed_distance([5, 5, 13, 13], [2, 1, 2, 8]), [2, 1, -3, -5])


def test_polygon_matches_rectangle():
    poly = PolygonZone("p", points=[[0, 0], [10, 0], [10, 4], [0, 4]])
    rect = RectangleZone("r", x=0, y=0, width=10, height=4)
    x, y = [5, 1, 12], [2, 3, 2]
    np.testing.assert_allclose(poly.signed_distance(x, y), rect.signed_distance(x, y), atol=1e-4)


def nested(h=0.0):
    return [CircleZone("center", h, 0, 0, 10), CircleZone("inner", h, 0, 0, 30), CircleZone("outer", h, 0, 0, 50)]


def test_priority_makes_rings():
    x = np.array([0, 20, 40, 60.0])
    np.testing.assert_array_equal(assign_zones(nested(), x, np.zeros(4)), [0, 1, 2, -1])


def test_outer_hysteresis_suppresses_border_flicker():
    # wobbling around the center border (r=10) inside a 4 px band
    x = np.array([0, 9, 11, 9, 12, 9, 13, 20.0])
    zones = assign_zones(nested(h=4), x, np.zeros_like(x), "outer")
    # never leaves center until it is >4 px outside (r > 14)
    np.testing.assert_array_equal(zones, [0, 0, 0, 0, 0, 0, 0, 1])
    no_h = assign_zones(nested(h=0), x, np.zeros_like(x), "outer")
    assert (np.diff(no_h) != 0).sum() == 5


@pytest.mark.parametrize("mode,expected", [
    ("outer", [1, 0, 0, 0, 1]),   # enter at r=10, leave beyond 14
    ("inner", [1, 1, 0, 1, 1]),   # enter only inside 6, leave at 10
    ("symmetric", [1, 1, 0, 0, 1]),  # enter inside 8, leave beyond 12
])
def test_hysteresis_modes(mode, expected):
    x = np.array([20, 9, 5, 11, 15.0])
    np.testing.assert_array_equal(assign_zones(nested(h=4), x, np.zeros_like(x), mode), expected)


def test_nan_holds_previous_zone():
    x = np.array([np.nan, 0, np.nan, 20.0])
    np.testing.assert_array_equal(assign_zones(nested(), x, np.zeros_like(x)), [0, 0, 0, 1])
