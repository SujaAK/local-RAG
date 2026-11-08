"""Citation repair (S3R): a claim that fails against its cited passage is re-pointed to a passage that entails it."""
import numpy as np

from src import pipeline as P
from src.utils import load_config
from src.verify import Verifier


def fake_verifier(cfg):
    v = object.__new__(Verifier)            # no model download: stub the NLI scorer
    v.cfg, v.embedder = cfg, None

    def probs(pairs):                         # entailment iff the premise mentions 'citation recall'
        return np.array([[0.01, 0.98, 0.01] if "citation recall" in prem else [0.05, 0.05, 0.90] for prem, _ in pairs])
    v.probs = probs
    return v


EVID = [{"id": "S1", "text": "A table of human agreement scores.", "paper_id": "p", "title": "T", "year": 2023, "section": "x", "page": 1},
        {"id": "S2", "text": "We define citation recall as support by cited passages.", "paper_id": "p", "title": "T", "year": 2023, "section": "x", "page": 2},
        {"id": "S3", "text": "Unrelated passage about data.", "paper_id": "p", "title": "T", "year": 2023, "section": "x", "page": 3}]


def test_repair_repoints_miscited_claim():
    cfg = load_config()
    claims = [{"text": "ALCE defines citation recall via support.", "evidence_ids": ["S1"]}]
    res = fake_verifier(cfg).verify_claims(claims, EVID, repair=True)[0]
    assert res["label"] == "Supported" and res["repaired_ids"] == ["S2"]


def test_without_repair_flag_nothing_changes():
    cfg = load_config()
    claims = [{"text": "ALCE defines citation recall via support.", "evidence_ids": ["S1"]}]
    res = fake_verifier(cfg).verify_claims(claims, EVID)[0]
    assert res["label"] != "Supported" and "repaired_ids" not in res


def test_no_repair_when_no_passage_supports_claim():
    cfg = load_config()
    claims = [{"text": "Something else entirely.", "evidence_ids": ["S1"]}]
    class V(Verifier):
        pass
    v = fake_verifier(cfg)
    v.probs = lambda pairs: np.array([[0.05, 0.05, 0.90] for _ in pairs])
    res = v.verify_claims(claims, EVID, repair=True)[0]
    assert res["label"] != "Supported" and "repaired_ids" not in res


def test_s3r_pipeline_updates_citation_and_counts_repairs(tmp_path, monkeypatch):
    from tests.test_pipeline import CHUNKS, FakeRetriever

    class RepairingVerifier:
        def verify_claims(self, claims, evidence, tau_e=None, tau_c=None, repair=False):
            assert repair is True
            return [{"label": "Supported", "score": 0.9, "p_entail": 0.9, "p_contra": 0.0, "p_neutral": 0.1,
                     "p_contra_other": 0.0, "repaired_ids": ["S2"]} for _ in claims]

    cfg = load_config(overrides={"paths": {"runs_dir": str(tmp_path)}})
    claim = {"text": "ok claim.", "evidence_ids": ["S1"]}
    monkeypatch.setattr(P, "generate_claims", lambda *a, **k: {"claims": [dict(claim)], "insufficient_evidence": False})
    r = P.Pipeline(cfg, retriever=FakeRetriever(), verifier=RepairingVerifier(), chunks=CHUNKS).answer("q", "S3R")
    c = r["claims"][0]
    assert c["evidence_ids"] == ["S2"] and c["original_evidence_ids"] == ["S1"] and c["citation_repaired"]
    assert r["n_repaired"] == 1 and not r["abstained"]


EVID_X = [  # the only passage that entails the claim comes from a DIFFERENT paper
    {"id": "S1", "text": "A table of human agreement scores.", "paper_id": "p", "title": "T", "year": 2023, "section": "x", "page": 1},
    {"id": "S3", "text": "Unrelated passage about data.", "paper_id": "p", "title": "T", "year": 2023, "section": "x", "page": 3},
    {"id": "S4", "text": "In our benchmark we define citation recall as support.", "paper_id": "q", "title": "U", "year": 2024, "section": "y", "page": 2}]


def test_repair_stays_within_the_cited_paper_by_default():
    cfg = load_config()
    claims = [{"text": "ALCE defines citation recall via support.", "evidence_ids": ["S1"]}]
    res = fake_verifier(cfg).verify_claims(claims, EVID_X, repair=True)[0]
    assert "repaired_ids" not in res and res["label"] != "Supported"
    assert [t["id"] for t in res["repair_trace"]] == ["S3"] or {t["id"] for t in res["repair_trace"]} <= {"S3"}


def test_repair_scope_any_can_cross_papers():
    cfg = load_config(overrides={"verify": {"repair_scope": "any"}})
    claims = [{"text": "ALCE defines citation recall via support.", "evidence_ids": ["S1"]}]
    res = fake_verifier(cfg).verify_claims(claims, EVID_X, repair=True)[0]
    assert res["repaired_ids"] == ["S4"] and any(t["id"] == "S4" for t in res["repair_trace"])