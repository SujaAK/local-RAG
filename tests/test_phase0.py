"""Harness fixes: cache key tied to the loaded model, parse errors kept, question checker."""
import json

from scripts.check_questions import check, summary
from src import generate as G
from src.utils import OutputCache, load_config

CH = {"p:0:0": {"chunk_id": "p:0:0", "paper_id": "p",
                "text": "Citation recall is the share of claims fully supported by their cited passages in the output."}}
GOODQ = {"id": "q1", "split": "dev", "type": "single_hop", "answerable": True, "question": "What is citation recall?",
         "gold_evidence": [{"paper_id": "p", "text": "share of claims fully supported by their cited passages"}]}


def _fake_urlopen(payloads):
    class R:
        def __init__(s, d): s.d = d
        def read(s): return json.dumps(s.d).encode()
        def __enter__(s): return s
        def __exit__(s, *a): return False
    class Op:
        def open(self, url, timeout=0):
            for k, v in payloads.items():
                if url.endswith(k):
                    return R(v)
            raise OSError("down")
    return Op()


def _identity(monkeypatch, model_path):
    import urllib.request
    G._IDENT.clear()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: _fake_urlopen(
        {"/props": {"model_path": model_path, "default_generation_settings": {"n_ctx": 4096}}}))
    return G.server_identity(load_config(overrides={"backend": "llamacpp"}))


def test_server_identity_changes_with_model_file(monkeypatch):
    a = _identity(monkeypatch, "qwen-1.5b.gguf")
    b = _identity(monkeypatch, "qwen-3b.gguf")
    assert a != b and "n_ctx=4096" in a
    assert OutputCache.key("llamacpp", "label", a, [], {}, None) != OutputCache.key("llamacpp", "label", b, [], {}, None)


def test_server_identity_unknown_when_server_down(monkeypatch):
    import urllib.request
    G._IDENT.clear()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: _fake_urlopen({}))
    assert G.server_identity(load_config()) == "unknown"


def test_pipeline_result_keeps_parse_error(tmp_path, monkeypatch):
    from tests.test_pipeline import GOOD, make
    from src import pipeline as P
    pipe = make(tmp_path, monkeypatch, [dict(GOOD)])
    monkeypatch.setattr(P, "generate_claims", lambda *a, **k: {"claims": [], "insufficient_evidence": True,
                                                              "parse_error": True})
    r = pipe.answer("q", "S3")
    assert r["parse_error"] is True and r["abstained"]


def test_checker_accepts_good_question():
    errs, _ = check([dict(GOODQ)], CH)
    assert errs == []


def test_checker_flags_problems():
    bad_span = dict(GOODQ, id="q2", gold_evidence=[{"paper_id": "p", "text": "text that is not in the paper at all"}])
    no_gold = dict(GOODQ, id="q3", gold_evidence=[])
    unans = dict(GOODQ, id="q4", type="unanswerable", answerable=False)       # has gold -> error
    multi = dict(GOODQ, id="q5", type="multi_hop")                            # one span only
    conflict = dict(GOODQ, id="q6", type="conflict", conflict_sides=["p", "other"])
    ghost = dict(GOODQ, id="q7", gold_evidence=[{"paper_id": "nope", "text": "x y z"}])
    dup = dict(GOODQ)
    errs, _ = check([bad_span, no_gold, unans, multi, conflict, ghost, GOODQ, dup], CH)
    joined = "\n".join(errs)
    for needle in ("q2", "q3", "q4", "q5", "q6", "q7", "duplicate id"):
        assert needle in joined


def test_checker_summary_counts():
    assert "single_hop" in summary([dict(GOODQ)])
