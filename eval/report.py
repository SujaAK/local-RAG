"""Aggregate scores.csv into tables with bootstrap CIs and paired comparisons (Holm-corrected)."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from . import stats

METRICS = ["recall@5", "recall@10", "allgold@10", "mrr", "ndcg@10", "citation_precision_auto", "citation_recall_auto",
           "fabrication_rate", "fabrication_rate_raw", "coverage", "answered", "parse_error", "n_repaired", "repaired_rate",
           "truncated", "n_supported", "n_contested", "n_unsupported", "n_insufficient", "latency_total"]


def summary_table(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for (system, model), g in df.groupby(["system", "model"]):
        row = {"system": system, "model": model, "n_questions": g["qid"].nunique()}
        for m in METRICS:
            if m in g:
                mean, lo, hi = stats.bootstrap_ci(g[m].dropna().tolist(), n_boot)
                row[m] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]" if mean == mean else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


# Confirmatory family for Holm correction: only (metric, pair) combinations where a difference is possible by design.
#  - retrieval metrics: S1 vs S0 only (S1, S2, S3 share one retriever, so those differences are zero by construction)
#  - coverage / answered: S3 vs S2 and S3R vs S3 only (S0-S2 never abstain on verification, so S1/S2 comparisons are structural)
#  - latency: every consecutive pair
# Everything else in METRICS (label counts, truncated, parse_error, ...) is descriptive: summary table only, no tests.
ALL_PAIRS = (("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S3", "S3R"))
METRIC_PAIRS = {
    **{m: (("S0", "S1"),) for m in ("recall@5", "recall@10", "mrr", "ndcg@10")},
    **{m: (("S2", "S3"), ("S3", "S3R")) for m in ("coverage", "answered", "citation_precision_auto",
                                                   "citation_recall_auto", "fabrication_rate")},
    "latency_total": ALL_PAIRS,
}


def paired_table(df: pd.DataFrame, n_boot: int, metric_pairs: dict | None = None) -> pd.DataFrame:
    metric_pairs = metric_pairs or METRIC_PAIRS
    recs = []
    for model, g in df.groupby("model"):
        for m, pairs in metric_pairs.items():
            if m not in g:
                continue
            tab = g.pivot(index="qid", columns="system", values=m)
            for a, b in pairs:
                if a in tab and b in tab:
                    r = stats.paired_bootstrap(tab[a].values, tab[b].values, n_boot)
                    recs.append({"model": model, "metric": m, "comparison": f"{b} - {a}", **r})
    out = pd.DataFrame(recs)
    if len(out):
        out["p_holm"] = stats.holm(out["p"].tolist())
    return out


def plot_risk_coverage(curves: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (cov, risk) in curves.items():
        ax.plot(cov, risk, label=name)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Unsupported-claim rate among shown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    run = Path(args.run)
    df = pd.read_csv(run / "scores.csv")
    s, p = summary_table(df, args.boot), paired_table(df, args.boot)
    md = ["# Evaluation report", "", f"Run: `{run.name}`. Intervals: 95% bootstrap resampled by question.",
          "Treat small differences as inconclusive (see power warning, C1).",
          "`*_auto` citation metrics come from an NLI model; the human audit (eval.audit_sample) is the primary number.", "", "## Summary", "",
          s.to_markdown(index=False), "", "## Paired comparisons (Holm-corrected over the confirmatory family only)", "",
          p.to_markdown(index=False) if len(p) else "n/a"]
    (run / "report.md").write_text("\n".join(md), encoding="utf-8")
    s.to_csv(run / "summary.csv", index=False)
    p.to_csv(run / "paired.csv", index=False)
    print((run / "report.md").read_text())


if __name__ == "__main__":
    main()