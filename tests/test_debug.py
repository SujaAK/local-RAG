from src.render import render_debug


def test_render_debug_lists_evidence_claims_and_scores():
    result = {
        "abstained": True, "insufficient_flag": False, "n_generated": 2,
        "evidence": [{"id": "S1", "paper_id": "p1", "section": "Method", "page": 3, "text": "alpha  beta\ngamma"}],
        "claims": [{"text": "A claim.", "evidence_ids": ["S1"], "label": "Unsupported",
                    "nli": {"p_entail": 0.1, "p_contra": 0.2, "p_contra_other": 0.0}, "validators": {"ok": True}}],
        "dropped_claims": [{"text": "Bad.", "validators": {"ok": False}}],
    }
    out = render_debug(result, {0: "premise text"})
    for needle in ("DEBUG", "[S1] p1 | Method | p.3: alpha beta gamma", "[Unsupported] 0.10 0.20 0.00", "NLI premise: premise text", "Dropped"):
        assert needle in out