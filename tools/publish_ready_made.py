#!/usr/bin/env python3
"""Publish every template in model_zoo/index.v1.json as a tracebloc ready-made model.

Each template is sent on its own to the backend's ``ready-made-models/publish/``:
its index row's facts, the file its ``upload`` names, and the tokenizer it
declares (``tokenizer_file = "<name>"``, a file beside it). The backend runs the
checks a user upload passes; the same files again run none.

    TRACEBLOC_PUBLISH_URL=https://... TRACEBLOC_PUBLISH_TOKEN=... \\
        python3 tools/publish_ready_made.py [--template ID ...] [--dry-run]

One line per template. Exits 1 when any template was not published, after
trying the rest, and 2 when the index or the environment cannot be read, or
the index lists a template under ``archive/``, before anything is sent. The
token is never printed. Standard library only.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
INDEX = "model_zoo/index.v1.json"
#: Templates kept out of the zoo, at the path they had under ``model_zoo/``.
ARCHIVE = "archive"
INDEX_VERSION = 1
ENDPOINT = "ready-made-models/publish/"
#: A busy check answers 503; it is asked again at most this many times.
RETRIES = 3
TIMEOUT_S = 600

#: The template's tokenizer declaration: a bare file name beside it.
TOKENIZER_FILE = re.compile(
    r"""^tokenizer_file\s*=\s*["']([A-Za-z0-9_.-]+\.json)["']\s*(?:#.*)?$""",
    re.MULTILINE,
)
#: The facts sent from each index row, besides its files.
FIELDS = ("id", "category", "framework", "label", "model_type", "batch")


class Unreadable(Exception):
    """The index, a template or the environment cannot be read."""


@dataclass(frozen=True)
class Template:
    row: dict
    path: Path
    tokenizer: Optional[Path]

    @property
    def id(self) -> str:
        return str(self.row["id"])


def _declared_tokenizer(path: Path) -> Optional[str]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            if "model.py" not in archive.namelist():
                return None
            text = archive.read("model.py").decode("utf-8", errors="replace")
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    match = TOKENIZER_FILE.search(text)
    return match.group(1) if match else None


def template(root: Path, row: dict) -> Template:
    """The files of index row ``row``. Raises :class:`Unreadable`."""
    missing = [key for key in (*FIELDS, "upload") if key not in row]
    if missing:
        raise Unreadable(f"{row.get('id')!r}: missing {', '.join(missing)}")
    upload = row["upload"]
    zoo = (root / "model_zoo").resolve()
    path = (zoo / str(upload)).resolve()
    if zoo not in path.parents or not path.is_file():
        raise Unreadable(f"{row['id']}: {upload} is not in the zoo")
    declared = _declared_tokenizer(path)
    tokenizer = None
    if declared is not None:
        tokenizer = path.parent / declared
        if not tokenizer.is_file():
            raise Unreadable(f"{row['id']}: declares a missing {declared}")
    return Template(row, path, tokenizer)


def index_rows(root: Path) -> list:
    """The index's template rows. Raises :class:`Unreadable` for an index that
    cannot be read or lists nothing: "cannot tell" is not "nothing to publish"."""
    try:
        document = json.loads((root / INDEX).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Unreadable(f"{INDEX} could not be read: {error}") from error
    if not isinstance(document, dict) or document.get("version") != INDEX_VERSION:
        raise Unreadable(f"{INDEX} is not version {INDEX_VERSION}")
    rows = document.get("templates")
    if not isinstance(rows, list) or not rows:
        raise Unreadable(f"{INDEX} lists no templates")
    if not all(isinstance(row, dict) and "id" in row for row in rows):
        raise Unreadable(f"{INDEX} has a template row without an id")
    return rows


def archived(root: Path) -> set:
    """The ids of the templates under ``archive/``, as the index would spell them."""
    archive = root / ARCHIVE
    return {p.relative_to(archive).with_suffix("").as_posix() for p in archive.rglob("*.py")}


def refuse_archived(root: Path, rows: list) -> None:
    """Raises :class:`Unreadable` when the index lists an archived template."""
    listed = sorted({str(row["id"]) for row in rows} & archived(root))
    if listed:
        raise Unreadable(f"{INDEX} lists archived templates: {', '.join(listed)}")


def form(item: Template) -> tuple[bytes, str]:
    """The multipart body for ``item`` and its content type. A ``model_type``
    of null is left out: the template declares none."""
    boundary = uuid.uuid4().hex
    out = io.BytesIO()

    def part(name: str, value: bytes, filename: Optional[str] = None) -> None:
        disposition = f'form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        out.write(f"--{boundary}\r\nContent-Disposition: {disposition}\r\n".encode())
        if filename is not None:
            out.write(b"Content-Type: application/octet-stream\r\n")
        out.write(b"\r\n" + value + b"\r\n")

    for key in FIELDS:
        value = item.row[key]
        if value is None:
            continue
        part("template_id" if key == "id" else key, str(value).encode("utf-8"))
    part("upload_file", item.path.read_bytes(), item.path.name)
    if item.tokenizer is not None:
        part("upload_tokenizer", item.tokenizer.read_bytes(), item.tokenizer.name)
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


Send = Callable[[urllib.request.Request], tuple[int, dict, dict]]


def send(request: urllib.request.Request) -> tuple[int, dict, dict]:
    """``(status, headers, json body)`` for ``request``; an answer that is not
    JSON is ``{}``."""
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            status, headers, raw = response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        status, headers, raw = error.code, dict(error.headers or {}), error.read()
    try:
        body = json.loads(raw.decode("utf-8"))
    except ValueError:
        body = {}
    return status, headers, body if isinstance(body, dict) else {}


def publish(
    url: str,
    token: str,
    item: Template,
    transport: Send = send,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, str]:
    """Whether ``item`` was published, and its line."""
    data, content_type = form(item)
    for attempt in range(RETRIES + 1):
        request = urllib.request.Request(
            url + ENDPOINT,
            data=data,
            method="POST",
            headers={"Authorization": f"Token {token}", "Content-Type": content_type},
        )
        try:
            status, headers, body = transport(request)
        except (urllib.error.URLError, OSError) as error:
            return False, f"error      {item.id}: {error}"
        if status == 503 and attempt < RETRIES:
            sleep(float(headers.get("Retry-After") or 5))
            continue
        break
    if status == 200 and body.get("outcome") in ("created", "unchanged"):
        mark = " (over the Run tab's parameter limit)" if body.get("over_run_tab_limit") else ""
        return True, f"{body['outcome']:<10} {item.id} v{body.get('version')}{mark}"
    if status == 422:
        return False, f"refused    {item.id}: {body.get('message')}"
    return False, f"error      {item.id}: HTTP {status} {json.dumps(body)[:500]}"


def target() -> tuple[str, str]:
    """The backend's base URL and the publisher's token, from the environment.
    Raises :class:`Unreadable` when either is missing or the URL is not https."""
    url = os.environ.get("TRACEBLOC_PUBLISH_URL", "").strip()
    token = os.environ.get("TRACEBLOC_PUBLISH_TOKEN", "").strip()
    if not url or not token:
        raise Unreadable("TRACEBLOC_PUBLISH_URL and TRACEBLOC_PUBLISH_TOKEN must be set")
    if urlsplit(url).scheme != "https":
        raise Unreadable("TRACEBLOC_PUBLISH_URL must be an https URL")
    return url.rstrip("/") + "/", token


def main(
    argv: Optional[Sequence[str]] = None, root: Path = ROOT, transport: Send = send
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--template", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="list, send nothing")
    args = parser.parse_args(argv)
    try:
        rows = index_rows(root)
        refuse_archived(root, rows)
        wanted = set(args.template)
        unknown = wanted - {row["id"] for row in rows}
        if unknown:
            raise Unreadable(f"not in the index: {', '.join(sorted(unknown))}")
        url, token = ("", "") if args.dry_run else target()
    except Unreadable as error:
        sys.stderr.write(f"{error}\n")
        return 2
    failed = 0
    for row in rows:
        if wanted and row["id"] not in wanted:
            continue
        try:
            item = template(root, row)
        except (Unreadable, zipfile.BadZipFile) as error:
            ok, line = False, f"invalid    {row['id']}: {error}"
        else:
            if args.dry_run:
                tokenizer = f" + {item.tokenizer.name}" if item.tokenizer else ""
                ok, line = True, f"would send {item.id}: {item.path.name}{tokenizer}"
            else:
                ok, line = publish(url, token, item, transport)
        failed += not ok
        sys.stdout.write(line + "\n")
    if failed:
        sys.stderr.write(f"{failed} template(s) not published\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
