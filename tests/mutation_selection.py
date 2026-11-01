"""Which tests are MUTATION ROWS, and the pytest hooks that keep them off pull requests.

The rows run once a week, through ``make mutation-nightly`` (called on a schedule by
``.github/workflows/mutation-nightly.yml``), and never on a pull request. A pull
request runs each guard's behaviour tests, and ``mutation-coverage / check`` proves
every guard has a row; a guard that a pull request breaks is caught by the next
weekly run.

THE RULE. A test is a mutation row when its file is listed under ``rows:`` in
``.github/mutation-coverage.yml`` AND either the file's name ends in ``_rows.py`` (a
file of rows only) or the test function's name starts with ``test_row_`` or
``test_MUTATION_`` (a file that also holds its guard's behaviour tests, which stay on
the pull request). The list is read from the declaration, never restated here.

THE HOOKS. Without ``--mutation-rows`` every row is deselected; with it every other
test is. ``tests/conftest.py`` composes them with its other hooks.

FAIL CLOSED. Each of these is a ``pytest.UsageError``, never a narrower run: a
declaration that cannot be read, or whose ``rows:`` has a shape the reader does not
know; a ``rows:`` entry that names no file; and, with ``--mutation-rows``, a run that
selects no row, or a rows file that gives the run no test.

Run as a script, it prints the rows files one per line, for the Makefile. Stdlib
only (pytest when present): PyYAML is not installed in the test environments, and
the Makefile runs this before its venv exists.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # run as a script on a bare interpreter
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ".github/mutation-coverage.yml"
OPTION = "--mutation-rows"
ROWS_ONLY_SUFFIX = "_rows.py"
ROW_PREFIXES = ("test_row_", "test_MUTATION_")

_KEY = re.compile(r"rows[ \t]*:(.*)")
_ITEM = re.compile(r"( *)-(?: +(.*))?")
_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+")
_REPORT = pytest.StashKey[str]() if pytest else None


class Refusal(Exception):
    """The rows cannot be told from the other tests: a usage error, never a narrower run."""


def _content(line: str) -> str:
    """``line`` without a trailing YAML comment, right-stripped."""
    return re.sub(r"(?:^|[ \t])#.*", "", line).rstrip()


def _path(value: str, where: str) -> str:
    segments = value.split("/")
    if not all(_SEGMENT.fullmatch(s) and s not in (".", "..") for s in segments):
        raise Refusal(
            f"{where}: rows entry {value!r} is not a plain repo-relative path; this "
            "reader takes literal paths (no quotes, globs, mappings, `./`, `..` or `/`)"
        )
    return value


def read_rows(text: str, where: str = DECLARATION) -> list[str]:
    """The ``rows:`` block list of ``text``, in order, or ``[]`` for ``rows: []``.

    Raises Refusal on every other shape: no top-level ``rows:``, two of them, any
    other inline value, a line under it that is not a ``- <path>`` item, items at
    mixed indents, a tab, an entry that is not a plain path, a duplicate, or no value.
    """
    lines = text.splitlines()
    keys = [n for n, line in enumerate(lines) if _KEY.fullmatch(_content(line))]
    if not keys:
        raise Refusal(f"{where}: no top-level `rows:` list")
    if len(keys) > 1:
        raise Refusal(f"{where}: `rows:` appears {len(keys)} times at the top level")
    start = keys[0]
    inline = _KEY.fullmatch(_content(lines[start])).group(1).strip()
    if inline not in ("", "[]"):
        raise Refusal(
            f"{where}:{start + 1}: `rows:` has the inline value {inline!r}; this reader "
            "takes a block list of `- <path>` lines, or `[]`"
        )
    entries: list[str] = []
    indent = None
    for n in range(start + 1, len(lines)):
        line = _content(lines[n])
        if not line:
            continue
        if not line.startswith((" ", "\t", "-")):
            break  # the next top-level key ends the list
        at = f"{where}:{n + 1}"
        if "\t" in line:
            raise Refusal(f"{at}: a tab under `rows:`")
        item = _ITEM.fullmatch(line)
        if not item or not item.group(2):
            raise Refusal(f"{at}: {line.strip()!r} under `rows:` is not a `- <path>` item")
        if indent is None:
            indent = len(item.group(1))
        elif len(item.group(1)) != indent:
            raise Refusal(f"{at}: `rows:` items at mixed indents")
        path = _path(item.group(2), at)
        if path in entries:
            raise Refusal(f"{at}: rows entry {path} is listed twice")
        entries.append(path)
    if inline == "[]" and entries:
        raise Refusal(f"{where}:{start + 1}: `rows: []` is followed by items")
    if inline == "" and not entries:
        raise Refusal(f"{where}:{start + 1}: `rows:` has no value; list the row files, or `[]`")
    return entries


def row_files(root: Path = ROOT) -> list[str]:
    """The declaration's rows files, repo-relative, each an existing file. Raises Refusal."""
    try:
        text = (root / DECLARATION).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Refusal(f"{DECLARATION} cannot be read: {exc}") from exc
    entries = read_rows(text)
    missing = [entry for entry in entries if not (root / entry).is_file()]
    if missing:
        raise Refusal(f"`rows:` names {missing}, which is not a file in this checkout")
    return entries


def is_row(path: str, name: str, rows) -> bool:
    """Whether test function ``name`` in repo-relative ``path`` is a mutation row."""
    if path not in rows:
        return False
    return path.endswith(ROWS_ONLY_SUFFIX) or name.startswith(ROW_PREFIXES)


def _location(item) -> str | None:
    try:
        return Path(item.path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def pytest_addoption(parser):
    parser.addoption(
        OPTION,
        action="store_true",
        default=False,
        help="run only the mutation rows (the weekly run, `make mutation-nightly`); "
        "without it they are deselected",
    )


def select(config, items) -> None:
    """Deselect the rows, or with --mutation-rows every other test. Fails closed."""
    try:
        rows = row_files()
    except Refusal as exc:
        raise pytest.UsageError(f"mutation rows: {exc}") from None
    weekly = config.getoption(OPTION)
    given = dict.fromkeys(rows, 0)
    seen = set()
    keep, drop = [], []
    for item in items:
        path = _location(item)
        name = getattr(item, "originalname", None) or item.name
        row = path is not None and is_row(path, name, given)
        if path in given:
            seen.add(path)
        if row:
            given[path] += 1
        (keep if row == weekly else drop).append(item)
    if weekly:
        if not any(given.values()):
            raise pytest.UsageError(
                f"mutation rows: {OPTION} selected no mutation row (rows files: {rows})"
            )
        absent = [path for path in rows if path not in seen]
        if absent:
            raise pytest.UsageError(
                f"mutation rows: rows file(s) {absent} were not collected (or failed to); "
                "the weekly run takes every rows file, so narrow it with -k, not by path"
            )
        silent = [path for path, count in given.items() if not count]
        if silent:
            raise pytest.UsageError(
                f"mutation rows: rows file(s) {silent} gave the weekly run no test; name "
                f"the rows `test_row_*` or `test_MUTATION_*`, or the file "
                f"`*{ROWS_ONLY_SUFFIX}`, or drop it from `rows:`"
            )
    if weekly:
        config.stash[_REPORT] = (
            f"mutation rows: {sum(given.values())} selected from {len(rows)} rows files; "
            f"{len(drop)} other tests deselected"
        )
    elif drop:
        config.stash[_REPORT] = (
            f"mutation rows: {len(drop)} deselected; they run weekly (make mutation-nightly)"
        )
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_report_collectionfinish(config):
    return config.stash.get(_REPORT, None)


def main() -> int:
    try:
        rows = row_files()
    except Refusal as exc:
        print(f"mutation rows: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"mutation rows: {DECLARATION} lists no rows file; the weekly run would run nothing",
              file=sys.stderr)
        return 2
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
