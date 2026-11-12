from pathlib import Path

import pytest

from behaviour.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    """Example config: center r=47, inner r=120, outer r=192 around (320, 240), 320 px/m."""
    return load_config(ROOT / "configs" / "example_open_field.toml")
