"""Regression tests for section-heading detection (body lines that start with a keyword must not become sections)."""
import pytest

from src.ingest import is_heading

BODY = 10.0
NOT_HEADINGS = [   # fragments that were wrongly turned into section names in the first real run
    "evaluation is in accordance with the attributable to",
    "related work due to its challenging nature ( Nakano",
    "results in the largest recall improvement. The set",
    "Appendix F for more details",
    "Evaluation is in accordance with the attributable to",
    "Results in the largest recall improvement.",
    "3 Results 85.2",
    "approach, we also",
]
HEADINGS = [
    "Abstract", "1 Introduction", "2 Related Work", "Related work", "3.1 Method", "Experiments",
    "5 Conclusion", "Conclusions and Future Work", "Results and Discussion", "References",
    "Appendix A Prompts", "6 Limitations", "ACKNOWLEDGMENTS", "4 Experimental setup",
]


@pytest.mark.parametrize("text", NOT_HEADINGS)
def test_body_fragments_rejected_even_if_bold_or_large(text):
    assert not is_heading(text, 10.0, False, BODY)
    assert not is_heading(text, 10.0, True, BODY)
    assert not is_heading(text, 11.0, True, BODY)


@pytest.mark.parametrize("text", HEADINGS)
def test_real_headings_accepted(text):
    assert is_heading(text, 11.95, True, BODY)


def test_large_title_font_still_counts():
    assert is_heading("Retrieval Pipelines for Scholarly QA", 20.0, True, BODY)