"""Prepare a local Python runtime from the lab's PyInstaller TailTrail bundle.

The original executable and settings are read-only. Extracted vendor code keeps
TailTrail's licence alongside it; none is copied into this repository's source.
"""

from __future__ import annotations

import hashlib
import json
import marshal
import struct
import urllib.request
import zipfile
import zlib
from contextlib import contextmanager
from pathlib import Path

PYTHON_URL = "https://www.python.org/ftp/python/3.13.5/python-3.13.5-embed-amd64.zip"
PYTHON_SHA256 = "7d2650fd9d1b9d002d4a315d5f354247fd6a44f30517c7ef577b08f57a0fb6d9"
RUNTIME_VERSION = 1


def file_hash(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@contextmanager
def folder_lock(folder: Path):
    """An OS lock is released even if a batch is killed or the PC restarts."""
    import msvcrt

    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".batch.lock").open("a+b") as lock:
        lock.write(b"\0")
        lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(f"Another batch is using {folder}") from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def archive_members(exe: Path):
    """Read PyInstaller CArchive records, including the embedded PYZ archive."""
    data = exe.read_bytes()
    cookie = data.rfind(b"MEI\014\013\012\013\016")
    if cookie < 0 or cookie + 88 > len(data):
        raise ValueError(f"Not a supported PyInstaller executable: {exe}")
    _, length, toc, toc_size, version, _ = struct.unpack("!8sIIII64s", data[cookie:cookie + 88])
    if version != 313:
        raise ValueError(f"TailTrail uses Python {version}; this runner supports its Python 3.13 build")
    start = cookie + 88 - length
    cursor, end = start + toc, start + toc + toc_size
    if not 0 <= start <= cursor <= end == cookie:
        raise ValueError("Invalid TailTrail archive bounds")
    while cursor < end:
        size, offset, packed, unpacked, compressed, kind = struct.unpack(
            "!IIIIBc", data[cursor:cursor + 18])
        if size < 18 or cursor + size > end or offset + packed > toc:
            raise ValueError("Invalid TailTrail archive record")
        name = data[cursor + 18:cursor + size].split(b"\0", 1)[0].decode("utf-8")
        content = data[start + offset:start + offset + packed]
        if compressed:
            content = zlib.decompress(content)
        if len(content) != unpacked:
            raise ValueError(f"Invalid size for TailTrail archive member {name}")
        yield name, kind, content
        cursor += size


def _destination(root: Path, name: str) -> Path:
    dest = (root / name).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise ValueError(f"Archive path escapes runtime folder: {name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def prepare_runtime(exe: Path, cache: Path, digest: str) -> Path:
    root = cache / digest[:16]
    stamp = root / "runtime.json"
    expected = dict(exe_sha256=digest, runtime_version=RUNTIME_VERSION)
    with folder_lock(cache):
        if stamp.exists() and json.loads(stamp.read_text(encoding="utf-8")) == expected:
            if (root / "python.exe").exists() and (root / "src/tracker.pyc").exists():
                return root / "python.exe"
        # Check the installed bundle before downloading anything.
        members = list(archive_members(exe))
        pyz = next(content for name, _, content in members if name == "PYZ.pyz")
        if pyz[:8] != b"PYZ\0\xf3\r\r\n":
            raise ValueError("Unexpected TailTrail Python bytecode format")
        licence = exe.parent / "license.txt"
        if not licence.is_file():
            raise FileNotFoundError(f"Keep TailTrail's license.txt next to {exe}")
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / "python-3.13.5-embed-amd64.zip"
        if not archive.exists():
            print("Downloading the matching portable Python launcher (10 MB)...", flush=True)
            with urllib.request.urlopen(PYTHON_URL, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != PYTHON_SHA256:
                raise ValueError("Portable Python download checksum mismatch")
            archive.write_bytes(content)
        if file_hash(archive) != PYTHON_SHA256:
            raise ValueError(f"Checksum mismatch; remove {archive} and run again")
        root.mkdir(parents=True, exist_ok=True)
        print(f"Preparing TailTrail runtime in {root}", flush=True)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not name.endswith("/"):
                    _destination(root, name).write_bytes(z.read(name))
        header = pyz[4:8] + bytes(12)
        for name, kind, content in members:
            if kind in (b"b", b"x"):
                _destination(root, name).write_bytes(content)
            elif kind in (b"m", b"M"):
                _destination(root, name + ".pyc").write_bytes(header + content)
        entries = marshal.loads(pyz[struct.unpack("!i", pyz[8:12])[0]:])
        for name, (kind, offset, size) in entries:
            module = name.replace(".", "/")
            if kind == 3:  # namespace package
                path, content = module + "/__init__.py", b""
            elif kind in (0, 1):
                path = module + ("/__init__.pyc" if kind == 1 else ".pyc")
                content = header + zlib.decompress(pyz[offset:offset + size])
            else:
                raise ValueError(f"Unsupported PYZ record: {name}, type {kind}")
            _destination(root, path).write_bytes(content)
        (root / "python313._pth").write_text(".\nbase_library.zip\npython313.zip\n", encoding="utf-8")
        (root / "TAILTRAIL_LICENSE.txt").write_bytes(licence.read_bytes())
        stamp.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return root / "python.exe"
