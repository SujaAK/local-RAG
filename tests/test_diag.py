import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("diag_nli", Path(__file__).resolve().parents[1] / "scripts" / "diag_nli.py")
diag = importlib.util.module_from_spec(spec)
sys.modules["diag_nli"] = diag
spec.loader.exec_module(diag)


class FakeVerifier:
    embedder = None

    def probs(self, pairs):    # entails only when the premise holds the supporting wording
        return np.array([[0.01, 0.97, 0.02] if "entirely supported" in prem else [0.01, 0.01, 0.98] for prem, _ in pairs])


def test_diagnose_reports_all_premise_variants_and_finds_best_sentence():
    passage = "Intro words here. We evaluate two metrics: citation recall determines if the output is entirely supported by cited passages. Other text."
    rows = diag.diagnose(FakeVerifier(), "Claim.", passage, variants=["Plain claim."])
    kinds = {r["premise"] for r in rows}
    assert {"top-1 sentences", "top-6 sentences", "whole passage", "best single sentence"} <= kinds
    best = [r for r in rows if r["premise"] == "best single sentence"][0]
    assert best["p_entail"] > 0.9 and "entirely supported" in best["text"]
    assert len(rows) == 2 * 6          # 2 hypotheses x (4 top-n + whole + best sentence)