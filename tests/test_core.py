import json

import numpy as np
import pytest

from eval import metrics as M
from eval import stats as S
from src import crypto
from src.chunk import chunk_text, page_for_offset
from src.ingest import clean_heading, is_heading, safe_id
from src.render import decide_abstain, shown_indices
from src.retrieve import rrf
from src.utils import config_hash, load_config, split_sentences
from src.validate import validate_claim
from src.verify import label_from_probs, softmax

EV = {"S1": {"id": "S1", "text": "The model reached 85.2% precision and 66.0 accuracy, which the authors call \"superhuman synthesis\"."},
      "S2": {"id": "S2", "text": "Other text."}}


def test_config_loads_and_hash_stable():
    cfg = load_config()
    assert cfg.chunk.tokens > cfg.chunk.overlap and cfg.seed == 42
    assert config_hash(cfg) == config_hash(load_config())


def test_rrf_deterministic_and_ties():
    r = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert r[:2] == ["a", "b"]           # tie -> ordered by id
    assert set(r) == {"a", "b", "c", "d"}


def test_chunking_offsets_and_overlap():
    text = " ".join(f"w{i}" for i in range(1000))
    ch = chunk_text(text, 100, 10)
    assert all(text[s:e] == t for s, e, t in ch)
    assert ch[1][2].split()[0] == "w90"
    assert ch[-1][2].split()[-1] == "w999"
    assert chunk_text("", 100, 10) == []
    with pytest.raises(ValueError):
        chunk_text("a b", 10, 10)


def test_page_for_offset():
    assert page_for_offset([[0, 3], [100, 4]], 150) == 4
    assert page_for_offset([[0, 3], [100, 4]], 5) == 3


def test_validators():
    ok = {"text": "It reached 85.2% precision.", "evidence_ids": ["S1"]}
    assert validate_claim(ok, EV)["ok"]
    assert not validate_claim({"text": "x", "evidence_ids": []}, EV)["V2"]
    assert not validate_claim({"text": "x", "evidence_ids": ["S9"]}, EV)["V1"]
    assert not validate_claim({"text": "Accuracy was 70.1.", "evidence_ids": ["S1"]}, EV)["V3"]
    assert validate_claim({"text": 'They claim "superhuman synthesis".', "evidence_ids": ["S1"]}, EV)["ok"]
    assert not validate_claim({"text": 'They said "totally made up".', "evidence_ids": ["S1"]}, EV)["V4"]


def test_label_logic_and_abstention():
    assert label_from_probs(0.9, 0.05, 0.05, 0.0, 0.7, 0.5) == "Supported"
    assert label_from_probs(0.1, 0.8, 0.1, 0.0, 0.7, 0.5) == "Unsupported"
    assert label_from_probs(0.9, 0.05, 0.05, 0.6, 0.7, 0.5) == "Contested"
    assert label_from_probs(0.2, 0.1, 0.7, 0.0, 0.7, 0.5) == "Insufficient"
    assert label_from_probs(0, 0, 1, 0, 0.7, 0.5, has_evidence=False) == "Insufficient"
    claims = [{"label": "Supported"}, {"label": "Unsupported"}, {"label": "Contested"}]
    assert shown_indices(claims) == [0, 2]
    assert not decide_abstain(claims, False, 0.3)
    assert decide_abstain(claims, False, 0.5)
    assert decide_abstain(claims, True, 0.0)
    assert decide_abstain([], False, 0.0)
    assert np.isclose(softmax(np.array([[1.0, 2.0, 3.0]])).sum(), 1.0)


def test_metrics():
    chunks = [{"paper_id": "p", "text": "alpha beta gamma"}, {"paper_id": "p", "text": "delta"}]
    gold = [{"paper_id": "p", "text": "alpha beta"}, {"paper_id": "p", "text": "delta"}]
    m = M.retrieval_metrics(chunks, gold, (1, 2))
    assert m["recall@1"] == 1.0 and m["allgold@1"] == 0.0 and m["allgold@2"] == 1.0 and m["mrr"] == 1.0
    cov, risk, area = M.risk_coverage([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1])
    assert risk[0] <= risk[-1] and cov[-1] == 1.0
    assert M.ece([1.0, 0.0], [1, 0]) == 0.0
    ab = M.abstention_prf([{"answerable": False, "abstained": True}, {"answerable": True, "abstained": True},
                           {"answerable": True, "abstained": False}])
    assert ab["abstention_recall"] == 1.0 and ab["false_abstention_rate"] == 0.5
    assert M.determinism_rate([["a", "b"], ["a", "c"]]) == 0.5
    ev = {"S1": {"text": "x supports claim"}}
    cs = M.citation_scores([{"text": "claim", "evidence_ids": ["S1", "S7"]}], ev,
                           lambda texts, h: any("supports" in t for t in texts))
    assert cs["fabrication_rate"] == 1.0 and cs["citation_recall_auto"] == 1.0


def test_stats():
    a, b = np.zeros(50), np.ones(50) + np.random.default_rng(0).normal(0, 0.1, 50)
    r = S.paired_bootstrap(a, b, n=2000)
    assert r["diff"] > 0.8 and r["lo"] > 0 and r["p"] < 0.01
    adj = S.holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert S.cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert S.cohen_kappa(list("aabb"), list("bbaa")) == -1.0
    assert S.wilson_halfwidth(0.5, 100) == pytest.approx(0.098, abs=0.002)


def test_crypto_roundtrip():
    blob = crypto.encrypt_bytes(b"secret", "pw")
    assert b"secret" not in blob and crypto.decrypt_bytes(blob, "pw") == b"secret"
    with pytest.raises(Exception):
        crypto.decrypt_bytes(blob, "wrong")


def test_ingest_helpers():
    assert is_heading("1 Introduction", 12, True, 10)
    assert is_heading("RESULTS", 10, False, 10)
    assert not is_heading("This is an ordinary sentence in the body of the paper that goes on.", 10, False, 10)
    assert clean_heading("3.2 Method.") == "Method"
    assert ":" not in safe_id("a:b c")


def test_sentences():
    assert split_sentences("One thing. Another thing! Third?") == ["One thing.", "Another thing!", "Third?"]
