"""Pipeline logic with fake retriever, generator and verifier (no models needed)."""
import pytest

from src import pipeline as P
from src.utils import load_config

CHUNKS = {f"p:0:{i}": {"chunk_id": f"p:0:{i}", "paper_id": "p", "title": "Paper P", "year": 2025,
                       "section": "Results", "page": 3, "text": t}
          for i, t in enumerate(["Accuracy was 66.0 on the benchmark.", "Other unrelated passage."])}


class FakeRetriever:
    embedder = None

    def retrieve(self, q, mode="hybrid", rerank=None, final_k=None, filters=None):
        return [dict(c, rank=i + 1) for i, c in enumerate(CHUNKS.values())]


class FakeVerifier:
    def verify_claims(self, claims, evidence, tau_e=None, tau_c=None):
        return [{"label": "Supported" if "supported" in c["text"] else "Unsupported", "score": 0.9,
                 "p_entail": 0.9, "p_contra": 0.0, "p_neutral": 0.1, "p_contra_other": 0.0} for c in claims]


def make(tmp_path, monkeypatch, claims, insufficient=False, **cfg_over):
    cfg = load_config(overrides={"paths": {"runs_dir": str(tmp_path)}, **cfg_over})
    monkeypatch.setattr(P, "generate_claims", lambda *a, **k: {"claims": claims, "insufficient_evidence": insufficient})
    monkeypatch.setattr(P, "generate_baseline", lambda *a, **k: {"claims": claims, "insufficient_evidence": False})
    return P.Pipeline(cfg, retriever=FakeRetriever(), verifier=FakeVerifier(), chunks=CHUNKS)


GOOD = {"text": "supported: accuracy was 66.0.", "evidence_ids": ["S1"]}
FAB = {"text": "supported: made up.", "evidence_ids": ["S9"]}
WEAK = {"text": "weak claim.", "evidence_ids": ["S2"]}


def test_s3_drops_fabricated_hides_unsupported_and_cites_sources(tmp_path, monkeypatch):
    r = make(tmp_path, monkeypatch, [dict(GOOD), dict(FAB), dict(WEAK)]).answer("q", "S3")
    assert [c["text"] for c in r["claims"]] == [GOOD["text"], WEAK["text"]]
    assert len(r["dropped_claims"]) == 1 and r["n_generated"] == 3
    assert [r["claims"][i]["label"] for i in r["shown"]] == ["Supported"]
    assert not r["abstained"] and "Sources" in r["markdown"] and "Paper P (2025)" in r["markdown"]


def test_s3_abstains_when_supported_fraction_low(tmp_path, monkeypatch):
    r = make(tmp_path, monkeypatch, [dict(WEAK), dict(WEAK)]).answer("q", "S3")
    assert r["abstained"] and r["shown"] == [] and "abstains" in r["markdown"] and "Sources" in r["markdown"]


def test_s3_abstains_on_model_insufficient_flag(tmp_path, monkeypatch):
    assert make(tmp_path, monkeypatch, [dict(GOOD)], insufficient=True).answer("q", "S3")["abstained"]


def test_s2_validates_but_has_no_nli_labels(tmp_path, monkeypatch):
    r = make(tmp_path, monkeypatch, [dict(GOOD), dict(FAB)]).answer("q", "S2")
    assert len(r["claims"]) == 1 and r["claims"][0].get("label") is None and not r["abstained"]


def test_s1_baseline_keeps_unvalidated_claims_but_records_fabrication(tmp_path, monkeypatch):
    r = make(tmp_path, monkeypatch, [dict(GOOD), dict(FAB)]).answer("q", "S1")
    assert len(r["claims"]) == 2 and r["claims"][1]["validators"]["V1"] is False


def test_private_mode_disables_output_cache(tmp_path, monkeypatch):
    p = make(tmp_path, monkeypatch, [dict(GOOD)], privacy={"encrypt_at_rest": True})
    assert p.cache.enabled is False
    assert make(tmp_path, monkeypatch, [dict(GOOD)]).cache.enabled is True
    assert make(tmp_path, monkeypatch, [dict(GOOD)], privacy={"cache_outputs": False}).cache.enabled is False


def test_unknown_system_raises(tmp_path, monkeypatch):
    with pytest.raises(KeyError):
        make(tmp_path, monkeypatch, []).answer("q", "S9")
