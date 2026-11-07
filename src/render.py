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
                   use_fraction: bool = True) -> bool:
    if insufficient_flag or not claims:
        return True
    if not use_fraction:
        return False
    sup = sum(1 for c in claims if c.get("label") == "Supported")
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
