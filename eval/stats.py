"""C5. Paired bootstrap by question, Wilcoxon, Holm correction, Cohen's kappa."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n: int = 10000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray([x for x in values if x == x], float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a, b, n: int = 10000, seed: int = 0):
    """Resamples QUESTIONS (pairs). Returns mean diff (b - a), 95% CI, two-sided bootstrap p, Cohen's d_z."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    d = (b - a)[mask]
    if len(d) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "dz": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    return {"diff": float(d.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975)),
            "p": float(min(1.0, max(p, 1.0 / n))), "dz": float(d.mean() / sd) if sd and sd == sd and sd > 0 else float("nan"),
            "n": int(len(d))}


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as w

    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = (b - a)[m]
    if len(d) == 0 or np.all(d == 0):
        return float("nan")
    return float(w(d).pvalue)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def cohen_kappa(a: list, b: list) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Power warning helper: ~+-10 points for p=0.5, n=100."""
    return z * (p * (1 - p) / n) ** 0.5 if n else float("nan")
