"""S10. Abstention decision and rendering with sources."""
from __future__ import annotations


def shown_indices(claims: list[dict], show_unsupported: bool = False) -> list[int]:
    keep = []
    for i, c in enumerate(claims):
        lab = c.get("label")
        if lab is None or lab in ("Supported", "Contested") or show_unsupported:
            keep.append(i)
    return keep


def decide_abstain(claims: list[dict], insufficient_flag: bool, min_supported_fraction: float,
                   use_fraction: bool = True, count_contested: bool = False) -> bool:
    """count_contested=True: a claim flagged Contested is shown with a warning, so it counts as an answer and
    a purely contested answer is no longer hidden behind an abstention (config: abstain.count_contested, default False until contested detection is validated)."""
    if insufficient_flag or not claims:
        return True
    if not use_fraction:
        return False
    ok = ("Supported", "Contested") if count_contested else ("Supported",)
    sup = sum(1 for c in claims if c.get("label") in ok)
    return (sup / len(claims)) < min_supported_fraction


def source_list(evidence: list[dict]) -> list[str]:
    seen, out = set(), []
    for e in evidence:
        key = (e["paper_id"], e["section"], e["page"])
        if key not in seen:
            seen.add(key)
            out.append(f"{e['title']} ({e['year']}), section: {e['section']}, p. {e['page']}")
    return out


def render_markdown(result: dict, show_unsupported: bool = False) -> str:
    ev = {e["id"]: e for e in result["evidence"]}
    lines = []
    if result["abstained"]:
        lines.append("**The system abstains: the retrieved evidence does not support a reliable answer.**\n")
    else:
        for i in shown_indices(result["claims"], show_unsupported):
            c = result["claims"][i]
            tag = {"Contested": " ⚠ *contested*", "Unsupported": " ✗ *unsupported*",
                   "Insufficient": " ? *insufficient*"}.get(c.get("label"), "")
            refs = ", ".join(f"{x}" for x in c["evidence_ids"] if x in ev)
            lines.append(f"- {c['text']} [{refs}]{tag}")
            if c.get("label") == "Contested":
                lines.append("  - Retrieved passages disagree; see the evidence panel for both sides.")
    lines.append("\n**Sources**")
    for s in source_list(result["evidence"]):
        lines.append(f"- {s}")
    return "\n".join(lines)


def render_debug(result: dict, premises: dict | None = None, snippet: int = 220) -> str:
    """Developer view: what was retrieved, every generated claim with its label and NLI scores, and why it was hidden."""
    premises = premises or {}
    lines = ["", "=== DEBUG ===", f"insufficient_evidence flag from model: {result.get('insufficient_flag')}",
             f"thresholds: {result.get('thresholds')}",
             f"abstained: {result['abstained']}  | generated: {result.get('n_generated')}  | kept after validators: {len(result['claims'])}",
             "", "-- Retrieved evidence (id | paper | section | page):"]
    for e in result["evidence"]:
        text = " ".join(e["text"].split())
        lines.append(f"[{e['id']}] {e['paper_id']} | {e['section']} | p.{e['page']}: {text[:snippet]}...")
    lines += ["", "-- Claims (label | p_entail p_contra p_other | validators | cited ids):"]
    for i, c in enumerate(result["claims"]):
        n = c.get("nli") or {}
        probs = (f"{n.get('p_entail', 0):.2f} {n.get('p_contra', 0):.2f} {n.get('p_contra_other', 0):.2f}"
                 if n else "n/a")
        v = c.get("validators") or {}
        lines.append(f"{i + 1}. [{c.get('label', 'not verified')}] {probs} | validators ok={v.get('ok')} "
                     f"| cites {c.get('evidence_ids')}")
        if c.get("citation_repaired"):
            lines.append(f"   citation repaired: {c.get('original_evidence_ids')} -> {c.get('evidence_ids')}")
        for t in c.get("repair_trace") or []:
            lines.append(f"   repair tried {t['id']} ({t.get('paper_id')}): p_entail={t['p_entail']} p_contra={t['p_contra']}")
        lines.append(f"   claim: {c['text']}")
        if i in premises:
            lines.append(f"   NLI premise: {premises[i][:snippet * 2]}")
    if result.get("dropped_claims"):
        lines.append("")
        lines.append("-- Dropped by the deterministic validators:")
        for c in result["dropped_claims"]:
            lines.append(f"   - {c['text']}  | reasons: {c.get('validators')}")
    return "\n".join(lines)