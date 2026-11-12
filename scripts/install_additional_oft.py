"""Install pinned Tracktor, portable Windows FastTrack and DLC detector weights."""
from hashlib import sha256
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO


def fetch(url, digest):
    with urlopen(url, timeout=120) as response:
        data = response.read()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"Checksum failed for {url}")
    return data


def main():
    target = Path(sys.prefix) / "share" / "ratel-oft"
    target.mkdir(parents=True, exist_ok=True)
    code = fetch("https://raw.githubusercontent.com/vivekhsridhar/tracktor/3a2537aada178fc4dc190f04e45830a368ab3b5f/tracktor/tracktor.py",
                 "4dbba7b375d5ee7866469a7462bec3f23a94b8aff0ad631a41f9aeb164aea1ba")
    # OpenCV 4 no longer returns the unused image that this display conversion reads.
    code = code.replace(b"    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)", b"")
    (target / "tracktor_engine.py").write_bytes(code)
    if sys.platform == "win32":
        archive = fetch("https://github.com/FastTrackOrg/FastTrack/releases/download/v6.7.0/FastTrack-windows-x86-64.zip",
                        "569cf7a4f6ee518a8c99345f870436e71d6da317b7144c993998d7bc6a333845")
        with ZipFile(BytesIO(archive)) as z:
            z.extractall(target / "fasttrack")
    model = fetch("https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse/resolve/d443257e4b1145f7439e0953c2015d60fcdc1185/superanimal_topviewmouse_fasterrcnn_mobilenet_v3_large_fpn.pt",
                  "762c92c1ac10182f16dae3259cce631810930e05e19200b21b7d34a251c3f631")
    (target / "superanimal.pt").write_bytes(model)
    print(f"Installed verified OFT assets in {target}")


if __name__ == "__main__":
    main()
