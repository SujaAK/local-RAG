"""Regression tests for the section-A fixes (Unicode, V3 numbers, citation precision, retrieval depth,
parse retry/salvage, offline guard, contested abstention)."""
import json
import socket
from types import SimpleNamespace

from eval import metrics as M
from eval.run_eval import score_one
from src import audit
from src import generate as G
from src.render import decide_abstain
from src.sparse import tokenize
from src.utils import load_config, norm_text, normalize_unicode, split_sentences
from src.validate import _numbers, validate_claim


def test_unicode_ligatures_are_folded():
    assert normalize_unicode("ef\ufb01cient \ufb02ow") == "efficient flow"
    assert norm_text("Ef\ufb01cient") == "efficient"
    assert tokenize("ef\ufb01cient retrieval") == tokenize("efficient retrieval") == ["efficient", "retrieval"]
    assert split_sentences("An ef\ufb01cient method. Another.")[0] == "An efficient method."


def test_v3_numbers_with_units_and_suffixes():
    assert _numbers("a 3.5x speedup") == ["3.5"] and _numbers("the 1.5B model") == ["1.5"]
    assert _numbers("85.2% F1 in 2020.") == ["85.2", "2020"] and _numbers("the 3rd run") == []
    ev = {"S1": {"text": "It reached a 3.5x speedup with the 1.5B model."}}
    ok = validate_claim({"text": "It gave a 3.5x speedup with the 1.5B model.", "evidence_ids": ["S1"]}, ev)
    assert ok["failures"] == []
    bad = validate_claim({"text": "It gave a 3.6x speedup.", "evidence_ids": ["S1"]}, ev)
    assert any("V3" in f for f in bad["failures"])


def test_citation_precision_is_strict():
    ev = {"S1": {"text": "alpha"}, "S2": {"text": "beta"}}
    never = lambda texts, h: False                                           # nothing supports anything
    cs = M.citation_scores([{"text": "c", "evidence_ids": ["S1"]}], ev, never)
    assert cs["citation_precision_auto"] == 0.0                              # was 1.0 before the fix
    cs = M.citation_scores([{"text": "c", "evidence_ids": ["S1", "S2"]}], ev, lambda t, h: "alpha" in t)
    assert cs["citation_recall_auto"] == 1.0 and cs["citation_precision_auto"] == 0.5   # S2 is redundant
    cs = M.citation_scores([{"text": "c", "evidence_ids": ["S1", "S9"]}], ev, lambda t, h: "alpha" in t)
    assert cs["citation_precision_auto"] == 0.5 and cs["fabrication_rate"] == 1.0       # fabricated id is irrelevant


class _Retriever:
    def retrieve(self, q, mode="hybrid", rerank=None, final_k=None, filters=None):
        return [{"chunk_id": f"p:0:{i}", "paper_id": "p", "text": "gold span words here" if i == 7 else f"filler {i}"}
                for i in range(final_k or 6)]


def test_retrieval_metrics_look_past_final_k():
    pipe = SimpleNamespace(retriever=_Retriever(), chunks={})
    res = {"abstained": False, "n_generated": 0, "shown": [], "latency": {"total": 1.0}, "claims": [],
           "dropped_claims": [], "evidence": {}, "retrieved": [f"p:0:{i}" for i in range(6)], "parse_error": False}
    pipe.chunks = {c: {"chunk_id": c, "paper_id": "p", "text": "filler"} for c in res["retrieved"]}
    q = {"id": "q", "question": "x", "answerable": True,
         "gold_evidence": [{"paper_id": "p", "text": "gold span words here"}]}
    row = score_one(pipe, q, res, (5, 10), None, "S1")
    assert row["recall@5"] == 0.0 and row["recall@10"] == 1.0                # rank 8 is only visible at depth 10


def _fake_chat(outputs, seen):
    it = iter(outputs)

    def fake(cfg, model, messages, fmt, cache, num_predict=None):
        seen.append(num_predict)
        return next(it)
    return fake


TRUNC = '{"claims": [{"text": "A.", "evidence_ids": ["S1"]}, {"text": "B", "evid'
FULL = '{"claims": [{"text": "A.", "evidence_ids": ["S1"]}, {"text": "B.", "evidence_ids": ["S2"]}], "insufficient_evidence": false}'


def _run(monkeypatch, outputs):
    seen = []
    monkeypatch.setattr(G, "_chat", _fake_chat(outputs, seen))
    return G.generate_claims(load_config(), "m", "q", "[S1] t", None), seen


def test_truncated_json_is_retried_with_larger_budget(monkeypatch):
    r, seen = _run(monkeypatch, [TRUNC, FULL])
    assert len(r["claims"]) == 2 and not r.get("parse_error") and r["retries"] == 1
    assert seen[0] is None and seen[1] > load_config().generation.num_predict


def test_truncated_json_is_salvaged_when_retry_fails(monkeypatch):
    r, _ = _run(monkeypatch, [TRUNC, TRUNC])
    assert r["truncated"] and [c["text"] for c in r["claims"]] == ["A."] and not r.get("parse_error")


def test_unusable_output_is_a_parse_error(monkeypatch):
    r, _ = _run(monkeypatch, ["not json at all", "still not json"])
    assert r["parse_error"] and r["claims"] == [] and r["insufficient_evidence"]


def test_guard_is_idempotent_and_covers_connect_ex():
    audit.block_network()
    first = socket.socket.connect
    audit.block_network()
    audit.block_network()
    assert socket.socket.connect is first and getattr(socket.socket.connect_ex, "_rag_guard", False)
    s = socket.socket(socket.AF_INET)
    try:
        s.connect_ex(("93.184.216.34", 80))
        raise AssertionError("guard did not block connect_ex")
    except ConnectionError:
        pass
    s.close()


def test_host_parsing_and_loopback_rules():
    for v in ("127.0.0.1", "127.0.0.1:11434", "http://127.0.0.1:11434", "localhost:11434", "[::1]:11434", ""):
        assert audit.is_local_host(audit.host_of(v)), v
    for v in ("0.0.0.0:11434", "10.0.0.5", "http://example.com:80"):
        assert not audit.is_local_host(audit.host_of(v)), v
    assert audit.is_local_host("::ffff:127.0.0.1") and not audit.is_local_host("::ffff:8.8.8.8")


def test_contested_only_answer_is_not_hidden_when_enabled():
    claims = [{"label": "Contested"}]
    assert decide_abstain(claims, False, 0.5)                                  # old behaviour kept as default
    assert not decide_abstain(claims, False, 0.5, count_contested=True)
    assert decide_abstain([{"label": "Unsupported"}], False, 0.5, count_contested=True)


def test_repeated_claims_are_dropped():
    items = [{"text": "A uses B.", "evidence_ids": ["S1"]}] * 5 + [{"text": "a  uses b.", "evidence_ids": ["S1"]},
                                                                     {"text": "A uses B.", "evidence_ids": ["S2"]}]
    assert len(G._claims_from(items)) == 2          # same text+citations collapse; a different citation is kept


def test_reverify_script_helpers():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("reverify", pathlib.Path(__file__).parents[1] / "scripts" / "reverify.py")
    rv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)
    rows = [{"qid": "q", "evidence": [{"id": "S1", "text": "We propose ALCE, a benchmark."}],
             "claims": [{"text": "We propose ALCE, a benchmark.", "evidence_ids": ["S1"], "label": "Insufficient"},
                        {"text": "We propose ALCE, a benchmark.", "evidence_ids": ["S1"], "label": "Insufficient"}]}]
    assert len(list(rv.unique_claims(rows))) == 1
    assert rv.is_verbatim(rows[0]["claims"][0], {"S1": rows[0]["evidence"][0]})

    class V:
        def verify_claims(self, claims, evidence, te, tc):
            return [{"label": "Supported", "p_entail": 0.9, "p_contra": 0.0, "p_contra_other": 0.0} for _ in claims]
    res = rv.reverify(V(), rows, 0.7, 0.5)
    assert rv.summarise(res)["verbatim_not_supported"] == 0 and res[0]["old"] == "Insufficient"


def _verifier(mode, contested=True):
    from src.verify import Verifier
    import numpy as np
    cfg = load_config()
    cfg.verify["premise_mode"], cfg.verify["contested"], cfg.verify["premise_sentences"] = mode, contested, 3
    v = object.__new__(Verifier)
    v.cfg, v.embedder, v.order = cfg, None, ["contradiction", "entailment", "neutral"]

    def probs(pairs):          # fake NLI: entails only when the premise is a SINGLE sentence containing the claim
        rows = []
        for prem, hyp in pairs:
            single = prem.count(". ") == 0
            rows.append([0.0, 0.95, 0.05] if (hyp.rstrip(".") in prem and single) else [0.0, 0.01, 0.99])
        return np.array(rows)
    v.probs = probs
    return v


EVID = [{"id": "S1", "paper_id": "p", "text": "Alpha was first measured in the original study. Beta came second in the study that followed. Gamma followed much later on in the work."}]
CLAIM = [{"text": "Beta came second in the study that followed.", "evidence_ids": ["S1"]}]


def test_union_premise_recovers_verbatim_claims_the_joined_premise_misses():
    assert _verifier("top_n").verify_claims(CLAIM, EVID)[0]["label"] == "Insufficient"
    assert _verifier("union").verify_claims(CLAIM, EVID)[0]["label"] == "Supported"
    assert _verifier("max_sentence").verify_claims(CLAIM, EVID)[0]["label"] == "Supported"


def test_contested_check_can_be_switched_off():
    ev = EVID + [{"id": "S2", "paper_id": "q", "text": "Unrelated passage about something else entirely."}]
    v = _verifier("union", contested=False)
    calls = []
    orig = v.probs
    v.probs = lambda pairs: (calls.append(len(pairs)), orig(pairs))[1]
    out = v.verify_claims(CLAIM, ev)[0]
    assert out["p_contra_other"] == 0.0 and out["label"] == "Supported"


def _rv():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("reverify", pathlib.Path(__file__).parents[1] / "scripts" / "reverify.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_reverify_abstention_summary(tmp_path):
    rv = _rv()
    rows = [{"qid": q, "insufficient_flag": False, "question": q, "evidence": [{"id": "S1", "text": "t"}]} for q in ("a", "b", "c", "d")]
    res = [{"qid": "a", "new": "Supported", "ids": ["S1"], "text": "x", "p_entail": .9, "p_contra": 0, "verbatim": True},
           {"qid": "b", "new": "Insufficient", "ids": ["S1"], "text": "y", "p_entail": 0, "p_contra": 0, "verbatim": False},
           {"qid": "c", "new": "Supported", "ids": ["S1"], "text": "z", "p_entail": .9, "p_contra": 0, "verbatim": True}]
    s = rv.abstention_summary(res, rows, {"a": True, "b": True, "c": False, "d": False}, 0.5)
    assert s["abstained_questions"] == ["b", "d"] and s["on_unanswerable"] == "1/2" and s["on_answerable(false)"] == "1/2"
    rv.write_label_sheet(res, rows, tmp_path / "sheet.csv")
    sheet = (tmp_path / "sheet.csv").read_text(encoding="utf-8")
    assert "Supported" not in sheet.split("\n", 1)[1] and (tmp_path / "sheet_key.csv").exists()   # blind: no verifier labels


def test_reverify_scores_against_human_sheet(tmp_path):
    rv = _rv()
    rows = [{"qid": "a", "question": "Qa", "evidence": [], "claims": []}, {"qid": "b", "question": "Qb", "evidence": [], "claims": []}]
    res = [{"qid": "a", "text": "c1", "new": "Supported", "p_entail": .9, "p_contra": 0, "ids": [], "verbatim": True},
           {"qid": "a", "text": "c2", "new": "Supported", "p_entail": .9, "p_contra": 0, "ids": [], "verbatim": True},
           {"qid": "b", "text": "c3", "new": "Insufficient", "p_entail": .1, "p_contra": 0, "ids": [], "verbatim": False}]
    sheet = tmp_path / "s.csv"
    sheet.write_text("row_id,question,claim,cited_passage,human_label (Supported/Partly/Unsupported),notes\n"
                     "1,Qa,c1,x,Supported,\n2,Qa,c2,x,Partly,\n3,Qb,c3,x,supported,\n", encoding="utf-8-sig")
    s = rv.score_against_human(res, rows, sheet)
    assert s["matched"] == "3/3" and s["supported_precision"] == 0.5 and s["supported_recall"] == 0.5
    assert s["lenient_precision(Partly counts as right)"] == 1.0 and s["missed_supports"] == 1 and len(s["false_supports"]) == 1


def test_reverify_tau_sweep(tmp_path):
    rv = _rv()
    rows = [{"qid": "a", "question": "Qa", "evidence": [], "claims": []}]
    res = [{"qid": "a", "text": "c1", "new": "Supported", "p_entail": .95, "p_contra": 0.0, "p_contra_other": 0.0, "ids": [], "verbatim": True},
           {"qid": "a", "text": "c2", "new": "Insufficient", "p_entail": .60, "p_contra": 0.0, "p_contra_other": 0.0, "ids": [], "verbatim": True}]
    sheet = tmp_path / "s.csv"
    sheet.write_text("row_id,question,claim,cited_passage,human_label (Supported/Partly/Unsupported),notes\n"
                     "1,Qa,c1,x,Supported,\n2,Qa,c2,x,Supported,\n", encoding="utf-8")
    low, high = rv.sweep_tau(res, rows, sheet, [0.5, 0.9], 0.5)
    assert low["supported"] == 2 and low["recall"] == 1.0 and high["supported"] == 1 and high["missed"] == 1


def test_config_extends_is_resolved_through_a_chain(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\nnest: {p: 1, q: 1}\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nnest: {q: 2}\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("extends: b.yaml\nnest: {r: 3}\n", encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg.x == 1 and cfg.nest.p == 1 and cfg.nest.q == 2 and cfg.nest.r == 3
    (tmp_path / "loop1.yaml").write_text("extends: loop2.yaml\n", encoding="utf-8")
    (tmp_path / "loop2.yaml").write_text("extends: loop1.yaml\n", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError):
        load_config(tmp_path / "loop1.yaml")


def test_real_v2_config_loads_with_every_section():
    import pathlib
    cfg = load_config(pathlib.Path(__file__).parents[1] / "config" / "config.win.v2.yaml")
    assert cfg.privacy.offline is True and cfg.backend == "llamacpp"                      # from config.yaml / topn
    assert cfg.generation.num_predict == 300 and cfg.retrieval.final_k == 6 and cfg.claims_prompt == "system_claims_v2.txt"
    assert cfg.verify.premise_mode == "union" and cfg.verify.contested is False            # v2 overrides
    assert cfg.abstain.count_contested is False and cfg.abstain.min_supported_fraction == 0.5
    assert cfg.verify.tau_entail == 0.7 and cfg.verify.tau_contra == 0.5                  # unchanged from the frozen protocol
    topn = load_config(pathlib.Path(__file__).parents[1] / "config" / "config.win.topn.yaml")
    assert topn.verify.premise_mode == "top_n" and topn.privacy.offline is True            # preregistered config still resolves


def test_reverify_prefill_and_repair_precision(tmp_path):
    rv = _rv()
    rows = [{"qid": "a", "question": "Qa", "evidence": [{"id": "S1", "text": "p1"}, {"id": "S2", "text": "p2"}], "claims": []}]
    res = [{"qid": "a", "text": "old", "new": "Supported", "ids": ["S1"], "p_entail": .9, "p_contra": 0, "p_contra_other": 0, "verbatim": True, "repaired": False},
           {"qid": "a", "text": "fixed", "new": "Supported", "ids": ["S2"], "p_entail": .9, "p_contra": 0, "p_contra_other": 0, "verbatim": False, "repaired": True}]
    earlier = tmp_path / "earlier.csv"
    earlier.write_text("row_id,question,claim,cited_passage,human_label (Supported/Partly/Unsupported),notes\n1,Qa,old,p1,Supported,\n", encoding="utf-8")
    sheet = tmp_path / "new.csv"
    rv.write_label_sheet(res, rows, sheet, prefill=rv.load_prefill(earlier))
    text = sheet.read_text(encoding="utf-8")
    assert "prefilled" in text                                        # the known pair is copied, the new one is left blank
    import csv
    labelled = {r["claim"]: r for r in csv.DictReader(open(sheet, encoding="utf-8"))}
    assert labelled["old"]["human_label (Supported/Partly/Unsupported)"] == "Supported"
    assert labelled["fixed"]["human_label (Supported/Partly/Unsupported)"] == ""
    labelled["fixed"]["human_label (Supported/Partly/Unsupported)"] = "Supported"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(labelled["old"])); w.writeheader(); w.writerows(labelled.values())
    s = rv.score_against_human(res, rows, sheet)
    assert s["repaired_claims"] == {"n": 1, "human_supported": 1, "repair_precision": 1.0, "verifier_supported": 1}