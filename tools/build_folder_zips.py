"""Build the ready-to-upload zip of every folder template.

Most templates are one ``.py`` file, and a user uploads that file as it is.
A FOLDER template (today the YOLOs: ``object_detection/pytorch/yolo_v8/`` holds
``model.py`` and ``loss.py``) cannot be uploaded that way. The SDK's
``upload_model`` takes one path, a ``.py`` or a ``.zip``, and a YOLO trains
with its own ``loss.py``, so uploading the folder's ``model.py`` alone fails its
checks with "loss.py file missing in the zip". The upload needs a zip with
``model.py`` and ``loss.py`` at its root.

So the zoo ships that zip next to the folder: ``yolo_v8/`` and ``yolo_v8.zip``
side by side, and a user who cloned the repo passes the ``.zip``. The SDK's
``get_paths`` resolves ``.../yolo_v8.zip`` (or ``.../yolo_v8``, no extension)
to it, and the model is named after the zip's stem, ``yolo_v8``.

What a folder template is: a template (``build_index.discover``'s rule, a
``.py`` declaring ``framework = "..."``) that sits one directory below its
framework directory, ``<category>/<framework>/<name>/<file>.py``. Its zip holds
every ``.py`` directly in ``<name>/`` -- flat, no folder prefix, no
subdirectories, no ``__pycache__`` -- and is written to
``<category>/<framework>/<name>.zip``.

The zip is committed, so it is a second copy of its folder.
``tests/test_folder_template_zips.py`` rebuilds every zip and fails on any
byte of difference, and on a ``.zip`` under model_zoo/ that no folder builds,
so the copy cannot drift from its source. That only works if the build is
reproducible: entries are written in name order, with a fixed timestamp and
fixed permissions, and STORED rather than deflated (deflate output depends on
the zlib build; these files are a few KB).

Usage::

    python tools/build_folder_zips.py            # (re)write every folder zip
    python tools/build_folder_zips.py --check    # exit 1 if one is missing,
                                                 # stale, or has no folder
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model_zoo"

#: 1980-01-01 00:00:00, the earliest time a zip entry can carry. A fixed value
#: so the bytes depend on the sources only, never on a checkout's mtimes.
FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
#: -rw-r--r-- on a regular file, in the high 16 bits as Unix zip tools write it.
FIXED_EXTERNAL_ATTR = (0o100644 & 0xFFFF) << 16
#: "Unix". zipfile defaults it from the host OS, which would make a zip built
#: on Windows differ from one built on Linux.
CREATE_SYSTEM_UNIX = 3


def _build_index():
    spec = importlib.util.spec_from_file_location(
        "build_index", ROOT / "tools" / "build_index.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def folders(model_root: pathlib.Path = MODEL_ROOT) -> list[pathlib.Path]:
    """Every folder template's directory, sorted. A template directly in its
    framework directory (``<category>/<framework>/<name>.py``) is not one."""
    found = set()
    for path, _consts in _build_index().discover(model_root):
        rel = path.relative_to(model_root)
        if len(rel.parts) == 4:
            found.add(path.parent)
        elif len(rel.parts) != 3:
            raise ValueError(
                f"{rel}: a template is <category>/<framework>/<name>.py or "
                "<category>/<framework>/<name>/<file>.py; this one is neither"
            )
    return sorted(found)


def zip_path(folder: pathlib.Path) -> pathlib.Path:
    """``<category>/<framework>/<name>/`` -> ``<category>/<framework>/<name>.zip``."""
    return folder.with_name(folder.name + ".zip")


def members(folder: pathlib.Path) -> list[pathlib.Path]:
    """The files that go in the zip: every ``.py`` directly in the folder."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix == ".py")


def build(folder: pathlib.Path) -> bytes:
    """The zip's bytes, the same on every machine for the same sources."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in members(folder):
            info = zipfile.ZipInfo(path.name, date_time=FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = FIXED_EXTERNAL_ATTR
            info.create_system = CREATE_SYSTEM_UNIX
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def problems(model_root: pathlib.Path = MODEL_ROOT) -> list[str]:
    """Every way the committed zips disagree with their folders, one line each.
    Empty means every folder has its zip, byte for byte, and no zip is
    orphaned. Finding no folder template at all is a problem too: this check
    would otherwise pass by looking at nothing."""
    found = folders(model_root)
    if not found:
        return [f"no folder template found under {model_root}"]
    lines = []
    expected = set()
    for folder in found:
        target = zip_path(folder)
        expected.add(target)
        rel = target.relative_to(model_root)
        if not target.is_file():
            lines.append(f"{rel}: missing")
        elif target.read_bytes() != build(folder):
            lines.append(f"{rel}: stale (does not match {folder.name}/)")
    scratch = _build_index().SCRATCH_DIR_PREFIX
    zips = {
        p
        for p in model_root.rglob("*.zip")
        if not any(part.startswith(scratch) for part in p.relative_to(model_root).parts)
    }
    for stray in sorted(zips - expected):
        lines.append(f"{stray.relative_to(model_root)}: no folder template builds it")
    return lines


def write_all(model_root: pathlib.Path = MODEL_ROOT) -> list[pathlib.Path]:
    written = []
    for folder in folders(model_root):
        target = zip_path(folder)
        data = build(folder)
        if not target.is_file() or target.read_bytes() != data:
            target.write_bytes(data)
            written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if a folder zip is missing or stale"
    )
    args = parser.parse_args(argv)
    if args.check:
        found = problems()
        if not found:
            print("every folder template zip is up to date.")
            return 0
        print("\n".join(found), file=sys.stderr)
        print("Run: python tools/build_folder_zips.py", file=sys.stderr)
        return 1
    for target in write_all():
        print(f"wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
