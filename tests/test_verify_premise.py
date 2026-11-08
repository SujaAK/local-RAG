"""Premise quality: PDF debris must never become NLI evidence; max_sentence mode scores sentences individually."""
import numpy as np
import pytest

from src.utils import load_config
from src.verify import Verifier, top_sentences, usable_sentence

JUNK = [
    "Citation recall.",
    "Fan et al.",
    "Model output String exact match Recall =66.6% Precision =50% Figure 2: Evaluation of correctness (details in § 3.2 ).",
    "Model output “ entailment ” Premise : passage [1][2] Hypothesis : {statement 1} NLI model Citation Recall {statement 2} [3] {statement 1} [1][2] {statement 3} [2][4][5]",
    "Figure 3: Examples of the citation precision computation for three statements.",
    "String exact match Recall =33.3% NLIClaim recall =33.3% QAMPARI Precision, Recall",
    "For each statement s i , its citation recall is 1 if",           # cut off mid-sentence
]
GOOD = [
    "3.3 Citation Quality We evaluate citation qualities using two metrics: (1) citation recall , which determines if the output is entirely supported by cited passages, and (2) citation precision , which identifies any irrelevant citations.",
    "We use an NLI model to verify whether a statement is supported by its citations.",
    "Student loans can affect the debt to income ratio of a borrower in several ways.",
]


@pytest.mark.parametrize("s", JUNK)
def test_debris_is_not_usable(s):
    assert not usable_sentence(s)


@pytest.mark.parametrize("s", GOOD)
def test_real_sentences_are_usable(s):
    assert usable_sentence(s)


def test_top_sentences_skips_debris_even_when_it_matches_the_claim_words():
    passage = " ".join(["Citation recall.", JUNK[2], GOOD[0], "Fan et al."])
    got = top_sentences("Citation recall determines if the output is entirely supported by cited passages.", [passage], 1)
    assert got and got[0].startswith("3.3 Citation Quality")


def make(mode):
    cfg = load_config(overrides={"verify": {"premise_mode": mode}})
    v = object.__new__(Verifier)
    v.cfg, v.embedder = cfg, None
    # entailment only when the premise holds the definition sentence on its own
    v.probs = lambda pairs: np.array([[0.01, 0.98, 0.01] if prem.startswith("3.3 Citation Quality") and "Fan" not in prem
                                      and " We use" not in prem else [0.02, 0.03, 0.95] for prem, _ in pairs])
    return v


def test_max_sentence_mode_finds_the_one_entailing_sentence():
    texts = [" ".join([GOOD[1], "Another ordinary sentence about unrelated experimental details here.", GOOD[0]])]
    prem, p = make("max_sentence").score("Citation recall determines if the output is entirely supported by cited passages.", texts)
    assert prem.startswith("3.3 Citation Quality") and p[1] > 0.9


def test_top_n_mode_is_the_default_and_joins_sentences():
    cfg = load_config("config/config.yaml")      # the base config, whatever RAG_CONFIG currently points to
    assert str(cfg.verify.get("premise_mode", "top_n")) == "top_n"