"""``tools/publish_ready_made.py``: what it sends for each template, and how it
reads the backend's answers. The backend is faked; nothing leaves the runner."""

from __future__ import annotations

import email.parser
import email.policy
import importlib.util
import json
import pathlib
import sys
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "publish_ready_made", ROOT / "tools/publish_ready_made.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Registered first: its dataclass resolves its annotations through it.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tool = _load()
URL = "https://backend.example/"


def row(template_id, upload, **extra):
    category, framework = template_id.split("/")[:2]
    return {
        "id": template_id,
        "upload": upload,
        "category": category,
        "framework": framework,
        "model_type": "",
        "label": "Net",
        "batch": 8,
        "params": None,
        "averageable": True,
        **extra,
    }


@pytest.fixture
def zoo(tmp_path):
    root = tmp_path / "zoo"
    files = root / "model_zoo"
    (files / "text_classification/pytorch").mkdir(parents=True)
    (files / "object_detection/pytorch").mkdir(parents=True)
    (files / "text_classification/pytorch/net.py").write_text(
        'tokenizer_file = "net_tokenizer.json"\n'
    )
    (files / "text_classification/pytorch/net_tokenizer.json").write_bytes(b'{"t": 1}')
    with zipfile.ZipFile(files / "object_detection/pytorch/det.zip", "w") as z:
        z.writestr("model.py", "x = 1\n")
        z.writestr("loss.py", "y = 2\n")
    rows = [
        row("text_classification/pytorch/net", "text_classification/pytorch/net.py"),
        row(
            "object_detection/pytorch/det/model",
            "object_detection/pytorch/det.zip",
            model_type=None,
        ),
    ]
    (files / "index.v1.json").write_text(json.dumps({"version": 1, "templates": rows}))
    return root


def parts(request):
    """The multipart form ``request`` carries, as ``{name: (filename, bytes)}``."""
    head = f"Content-Type: {request.get_header('Content-type')}\r\n\r\n".encode()
    message = email.parser.BytesParser(policy=email.policy.default).parsebytes(
        head + request.data
    )
    return {
        p.get_param("name", header="content-disposition"): (
            p.get_filename(),
            p.get_payload(decode=True),
        )
        for p in message.iter_parts()
    }


class Backend:
    """Answers each request in turn and keeps what it was sent."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.answers.pop(0)


def ok(outcome="created", **extra):
    return 200, {}, {"outcome": outcome, "version": 1, "over_run_tab_limit": False, **extra}


def test_each_template_is_sent_with_its_facts_its_file_and_its_tokenizer(zoo, monkeypatch, capsys):
    monkeypatch.setenv("TRACEBLOC_PUBLISH_URL", URL)
    monkeypatch.setenv("TRACEBLOC_PUBLISH_TOKEN", "secret-token")
    backend = Backend(ok(), ok("unchanged"))
    assert tool.main([], root=zoo, transport=backend) == 0
    text, det = (parts(r) for r in backend.requests)
    assert backend.requests[0].full_url == URL + "ready-made-models/publish/"
    assert backend.requests[0].get_header("Authorization") == "Token secret-token"
    assert text["template_id"] == (None, b"text_classification/pytorch/net")
    assert text["upload_file"] == ("net.py", b'tokenizer_file = "net_tokenizer.json"\n')
    assert text["upload_tokenizer"] == ("net_tokenizer.json", b'{"t": 1}')
    assert text["model_type"] == (None, b"")
    assert det["upload_file"][0] == "det.zip"
    assert "upload_tokenizer" not in det
    assert "model_type" not in det, "null means the template declares none"
    out = capsys.readouterr()
    assert "created    text_classification/pytorch/net v1" in out.out
    assert "secret-token" not in out.out + out.err


def test_a_refusal_fails_the_run_after_the_rest(zoo, monkeypatch, capsys):
    monkeypatch.setenv("TRACEBLOC_PUBLISH_URL", URL)
    monkeypatch.setenv("TRACEBLOC_PUBLISH_TOKEN", "t")
    backend = Backend((422, {}, {"outcome": "refused", "message": "no"}), ok())
    assert tool.main([], root=zoo, transport=backend) == 1
    assert len(backend.requests) == 2
    assert "refused    text_classification/pytorch/net: no" in capsys.readouterr().out


def test_a_busy_backend_is_asked_again_after_retry_after(zoo):
    slept = []
    item = tool.template(zoo, tool.index_rows(zoo)[0])
    backend = Backend((503, {"Retry-After": "7"}, {}), ok())
    assert tool.publish(URL, "t", item, backend, slept.append)[0]
    assert slept == [7.0]
    always_busy = Backend(*[(503, {}, {})] * (tool.RETRIES + 1))
    published, line = tool.publish(URL, "t", item, always_busy, slept.append)
    assert not published and "HTTP 503" in line


@pytest.mark.parametrize("status", [400, 401, 403, 500])
def test_any_other_answer_is_an_error(zoo, status):
    item = tool.template(zoo, tool.index_rows(zoo)[0])
    published, line = tool.publish(URL, "t", item, Backend((status, {}, {})))
    assert not published and f"HTTP {status}" in line


def test_only_the_named_templates_are_sent(zoo, monkeypatch):
    monkeypatch.setenv("TRACEBLOC_PUBLISH_URL", URL)
    monkeypatch.setenv("TRACEBLOC_PUBLISH_TOKEN", "t")
    backend = Backend(ok())
    assert tool.main(["--template", "object_detection/pytorch/det/model"], root=zoo, transport=backend) == 0
    assert len(backend.requests) == 1
    assert tool.main(["--template", "nope/x/y"], root=zoo, transport=backend) == 2


@pytest.mark.parametrize(
    "env",
    [{}, {"TRACEBLOC_PUBLISH_URL": URL}, {"TRACEBLOC_PUBLISH_URL": "http://plain/", "TRACEBLOC_PUBLISH_TOKEN": "t"}],
)
def test_no_https_target_and_token_is_an_error_before_anything_is_sent(zoo, monkeypatch, env):
    monkeypatch.delenv("TRACEBLOC_PUBLISH_URL", raising=False)
    monkeypatch.delenv("TRACEBLOC_PUBLISH_TOKEN", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    backend = Backend()
    assert tool.main([], root=zoo, transport=backend) == 2
    assert backend.requests == []


def test_a_dry_run_sends_nothing_and_needs_no_target(zoo, monkeypatch, capsys):
    monkeypatch.delenv("TRACEBLOC_PUBLISH_URL", raising=False)
    assert tool.main(["--dry-run"], root=zoo, transport=Backend()) == 0
    assert "would send text_classification/pytorch/net: net.py + net_tokenizer.json" in capsys.readouterr().out


def test_the_real_index_reads_in_full():
    rows = tool.index_rows(ROOT)
    items = [tool.template(ROOT, r) for r in rows]
    assert len(items) == len(rows) > 0
    text = [i for i in items if i.row["category"] == "text_classification"]
    assert text and all(i.tokenizer is not None for i in text)
