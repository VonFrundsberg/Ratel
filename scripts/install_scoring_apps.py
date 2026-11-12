"""Download pinned desktop scoring tools into the ignored project output folder."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).absolute().parents[1]
ASSETS = {
    "boris": ("https://github.com/olivierfriard/BORIS/releases/download/v10.0.2/boris-10.0.2-win64.zip",
              "c22c5a56c1d3b0d25cd6cd3ae875f614919c05f00d8d2aee5db5a7fd866636df", "boris.zip"),
    "dbscorer": ("https://raw.githubusercontent.com/swanandlab/DBscorer/21083f641c7b1551ac8b9b174517183ef7b610ab/DBscorerV2.exe",
                "8477babcdd2ab1fcf378464b84a35078fd9c7484dbfaa4b5dba797b348295a0c", "DBscorerV2.exe"),
    "behaviordepot": ("https://raw.githubusercontent.com/DeNardoLab/BehaviorDEPOT/626c2296ac2793747f97fcf6094bbfa90c3988d9/BehaviorDEPOT_v1_7.mltbx",
                     "9fd3a4875f68a02a8660d69f3250457167f47a57bb6bd348c4e5d5aa4f4be704", "BehaviorDEPOT_v1_7.mltbx"),
}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tools", nargs="+", choices=ASSETS)
    a = ap.parse_args()
    cache = ROOT / "output/package_cache/scorers"
    cache.mkdir(parents=True, exist_ok=True)
    for tool in a.tools:
        url, digest, name = ASSETS[tool]
        path = cache / name
        if not path.exists() or sha256(path.read_bytes()).hexdigest() != digest:
            partial = path.with_suffix(".download")
            with urlopen(url, timeout=120) as response, partial.open("wb") as f:
                while chunk := response.read(1024*1024):
                    f.write(chunk)
            if sha256(partial.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Unexpected source hash for {tool}")
            partial.replace(path)
        dest = ROOT / "output/scoring_tools" / tool
        dest.mkdir(parents=True, exist_ok=True)
        if tool == "boris":
            with ZipFile(path) as archive:
                archive.extractall(dest)
        else:
            import shutil
            shutil.copyfile(path, dest / name)
        print(f"{tool}: downloaded and verified -> {dest}")


if __name__ == "__main__":
    main()
