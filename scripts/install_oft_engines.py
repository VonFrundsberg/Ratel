"""Install verified upstream tracking modules into the active Python environment."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sysconfig
import tarfile
from urllib.request import urlopen


def download(url, digest):
    with urlopen(url, timeout=120) as response:
        data = response.read()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256 for {url}")
    return data


def main():
    target = Path(sysconfig.get_path("purelib"))
    ez = download(
        "https://raw.githubusercontent.com/denisecailab/ezTrack/69af797af4268b9c36faa1ca314df539aea7389f/LocationTracking/LocationTracking_Functions.py",
        "74940c5a6a87e681d2142f7ddacf22d7c737c58387b200babdbf5d31cef40360")
    package = download(
        "https://conda.anaconda.org/sraorao/win-64/animapp-0.1.5.6-py38_0.tar.bz2",
        "4f65b48d29d8e5c03396979916e2d8624ce0df9c644b7e56525a4962a0afbf21")
    # Only the upstream display initialization assumes an interactive notebook.
    ez = ez.replace(b"hv.notebook_extension('bokeh')", b"hv.extension('bokeh')")
    (target / "LocationTracking_Functions.py").write_bytes(ez)
    dest = target / "animapp"
    dest.mkdir(exist_ok=True)
    with tarfile.open(fileobj=BytesIO(package), mode="r:bz2") as archive:
        for name in ("__init__.py", "animapp.py", "process_csv.py"):
            (dest / name).write_bytes(archive.extractfile("Lib/site-packages/animapp/" + name).read())
    print(f"Installed ezTrack 69af797 and AnimApp 0.1.5.6 tracking modules in {target}")


if __name__ == "__main__":
    main()
