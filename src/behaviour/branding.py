"""RATEL's public names, separate from stable benchmark and export identifiers."""

SOFTWARE_NAME = "RATEL"
SOFTWARE_FULL_NAME = "Rodent Activity Tracking and Ethogram Lab"
REPORT_TITLE = "RATEL: an integrated framework for rodent behavioural analysis"

DISPLAY_NAMES = {
    "native_default": "RATEL (filtered)",
    "native_raw": "RATEL (raw)",
    "native": "RATEL",
    "native_manual": "RATEL manual",
    "opencv": "RATEL (OpenCV)",
}


def display_name(identifier: str) -> str:
    if identifier.startswith("native_manual_"):
        return "RATEL manual: " + identifier.removeprefix("native_manual_")
    return DISPLAY_NAMES.get(identifier, identifier)
