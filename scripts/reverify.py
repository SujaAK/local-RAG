#!/usr/bin/env python
"""Re-verify the claims saved in a run's S3 .jsonl with different NLI settings. Offline, no LLM, no regeneration.

The run file already stores every claim and its evidence, so thresholds and premise settings can be compared in
minutes. Use it to (a) see how labels move with tau_entail / tau_contra / premise_mode / premise_sentences,
(b) debug verbatim claims that the verifier rejects (--show-verbatim-fails), (c) later, score labels against your
hand annotations.

  python scripts/reverify.py --run runs\\<id>\\S3__qwen2_5-1_5b-instruct.jsonl --show-verbatim-fails
  python scripts/reverify.py --run <file> --premise-mode max_sentence --tau-entail 0.5
  python scripts/reverify.py --run <file> --premise-sentences 1
Run from the project folder with the venv active and RAG_CONFIG (or --config) set.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `import src` works when run as scripts/reverify.py


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def unique_claims(rows: list[dict]):
    """(row, claim) pairs with exact repeats removed per question."""
    seen = set()
    for r in rows:
        for c in r["claims"]:
            key = (r["qid"], c["text"], tuple(c["evidence_ids"]))
            if key not in seen:
                seen.add(key)
                yield r, c


def is_verbatim(claim: dict, evidence: dict) -> bool:
    from src.utils import norm_text
    cited = " ".join(norm_text(evidence[i]["text"]) for i in claim["evidence_ids"] if i in evidence)
    return norm_text(claim["text"]).rstrip(". ") in cited


def reverify(verifier, rows: list[dict], tau_e: float, tau_c: float) -> list[dict]:
    out = []
    for r in rows:
        claims = [c for _, c in unique_claims([r])]
        if not claims:
            continue
        for c, res in zip(claims, verifier.verify_claims(claims, r["evidence"], tau_e, tau_c)):
            out.append({"qid": r["qid"], "text": c["text"], "ids": c["evidence_ids"], "old": c.get("label"),
                        "new": res["label"], "p_entail": res["p_entail"], "p_contra": res["p_contra"],
                        "p_contra_other": res["p_contra_other"], "repaired": bool(c.get("citation_repaired")),
                        "verbatim": is_verbatim(c, {e["id"]: e for e in r["evidence"]})})
    return out


def summarise(res: list[dict]) -> dict:
    c = collections.Counter(x["new"] for x in res)
    v = [x for x in res if x["verbatim"]]
    return {"claims": len(res), "labels": dict(c),
            "verbatim_claims": len(v),
            "verbatim_not_supported": sum(1 for x in v if x["new"] != "Supported")}


def load_answerable(cfg) -> dict:
    from src.utils import rpath
    path = rpath(cfg, "eval_dir") / "questions.jsonl"
    if not path.exists():
        return {}
    qs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return {q["id"]: bool(q.get("answerable", True)) for q in qs}


def abstention_summary(res: list[dict], rows: list[dict], answerable: dict, min_frac: float,
                       count_contested: bool = False) -> dict:
    """Question-level abstention under the re-labelled claims (same rule as src.render.decide_abstain)."""
    labels = collections.defaultdict(list)
    for x in res:
        labels[x["qid"]].append(x["new"])
    ok = ("Supported", "Contested") if count_contested else ("Supported",)
    abst = {}
    for r in rows:
        labs = labels.get(r["qid"], [])
        abst[r["qid"]] = bool(r.get("insufficient_flag")) or not labs or sum(l in ok for l in labs) / len(labs) < min_frac
    ans = {q: answerable.get(q, True) for q in abst}
    tp = sum(1 for q in abst if abst[q] and not ans[q])
    fp = sum(1 for q in abst if abst[q] and ans[q])
    n_un, n_ans = sum(1 for q in abst if not ans[q]), sum(1 for q in abst if ans[q])
    return {"abstained": sum(abst.values()), "on_unanswerable": f"{tp}/{n_un}", "on_answerable(false)": f"{fp}/{n_ans}",
            "false_abstention_rate": round(fp / n_ans, 3) if n_ans else None,
            "abstention_precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "abstention_recall": round(tp / n_un, 3) if n_un else None,
            "abstained_questions": sorted(q for q in abst if abst[q])}


def _repair_stats(pairs) -> dict:
    """Claims whose citation S3R re-pointed: how many do you judge Supported by the NEW passage?"""
    r = [(h, x) for h, x in pairs if x.get("repaired")]
    ok = sum(1 for h, _ in r if h == "Supported")
    return {"n": len(r), "human_supported": ok, "repair_precision": round(ok / len(r), 3) if r else None,
            "verifier_supported": sum(1 for _, x in r if x["new"] == "Supported")}


def score_against_human(res: list[dict], rows: list[dict], sheet: Path) -> dict:
    """Compare the verifier's labels with your hand labels (matched on question + claim text).
    'Supported' is the positive class. Precision = how many claims the verifier approves are truly supported;
    recall = how many truly supported claims it approves."""
    import csv
    qtext = {r["qid"]: r["question"] for r in rows}
    human = {}
    with open(sheet, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            col = next((k for k in row if k.startswith("human")), None)
            if col and row[col].strip():
                human[(row["question"], row["claim"])] = row[col].strip().capitalize()
    pairs = [(human[(qtext[x["qid"]], x["text"])], x) for x in res if (qtext[x["qid"]], x["text"]) in human]
    conf = collections.Counter((h, x["new"]) for h, x in pairs)
    gold = sum(1 for h, _ in pairs if h == "Supported")
    pred = sum(1 for _, x in pairs if x["new"] == "Supported")
    tp = sum(1 for h, x in pairs if h == "Supported" and x["new"] == "Supported")
    lenient = sum(1 for h, x in pairs if h in ("Supported", "Partly") and x["new"] == "Supported")
    prec, rec = (tp / pred if pred else None), (tp / gold if gold else None)
    return {"matched": f"{len(pairs)}/{len(res)}", "human->verifier": {f"{h}->{v}": n for (h, v), n in sorted(conf.items())},
            "supported_precision": round(prec, 3) if prec is not None else None,
            "supported_recall": round(rec, 3) if rec is not None else None,
            "supported_f1": round(2 * prec * rec / (prec + rec), 3) if prec and rec else None,
            "lenient_precision(Partly counts as right)": round(lenient / pred, 3) if pred else None,
            "false_supports": [f"[{x['qid']}] human={h}: {x['text'][:90]}" for h, x in pairs if x["new"] == "Supported" and h != "Supported"],
            "missed_supports": sum(1 for h, x in pairs if h == "Supported" and x["new"] != "Supported"),
            "repaired_claims": _repair_stats(pairs)}


def sweep_tau(res: list[dict], rows: list[dict], sheet: Path, taus: list[float], tau_c: float) -> list[dict]:
    """Relabel the same NLI probabilities at several tau_entail values and score each against the human labels.
    No model call is needed. Caution: picking tau on the same claims you score on is optimistic."""
    from src.verify import label_from_probs
    out = []
    for te in taus:
        relabeled = [{**x, "new": label_from_probs(x["p_entail"], x["p_contra"], max(0.0, 1 - x["p_entail"] - x["p_contra"]),
                                                   x["p_contra_other"], te, tau_c)} for x in res]
        s = score_against_human(relabeled, rows, sheet)
        out.append({"tau_entail": te, "supported": sum(1 for x in relabeled if x["new"] == "Supported"),
                    "precision": s["supported_precision"], "recall": s["supported_recall"], "f1": s["supported_f1"],
                    "false_supports": len(s["false_supports"]), "missed": s["missed_supports"]})
    return out


def load_prefill(path: Path) -> dict:
    import csv
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            col = next((k for k in row if k.startswith("human")), None)
            if col and row[col].strip():
                out[(row["question"], row["claim"], row["cited_passage"])] = row[col].strip()
    return out


def write_label_sheet(res: list[dict], rows: list[dict], path: Path, seed: int = 42, prefill: dict | None = None) -> None:
    """Blind annotation sheet (no verifier labels, shuffled) + a key file with the verifier's answers."""
    import csv
    import random
    ev = {r["qid"]: {e["id"]: e["text"] for e in r["evidence"]} for r in rows}
    qtext = {r["qid"]: r["question"] for r in rows}
    items = list(res)
    random.Random(seed).shuffle(items)
    with open(path, "w", newline="", encoding="utf-8") as f, \
            open(path.with_name(path.stem + "_key.csv"), "w", newline="", encoding="utf-8") as k:
        w = csv.writer(f)
        kw = csv.writer(k)
        w.writerow(["row_id", "question", "claim", "cited_passage", "human_label (Supported/Partly/Unsupported)", "notes"])
        kw.writerow(["row_id", "qid", "verifier_label", "p_entail", "p_contra", "verbatim", "repaired"])
        for n, x in enumerate(items, 1):
            cited = " || ".join(ev[x["qid"]].get(i, "") for i in x["ids"])
            done = (prefill or {}).get((qtext[x["qid"]], x["text"], cited), "")
            w.writerow([n, qtext[x["qid"]], x["text"], cited, done, "prefilled" if done else ""])
            kw.writerow([n, x["qid"], x["new"], round(x["p_entail"], 4), round(x["p_contra"], 4), x["verbatim"], x.get("repaired", False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to a run's S3 .jsonl")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tau-entail", type=float, default=None)
    ap.add_argument("--tau-contra", type=float, default=None)
    ap.add_argument("--premise-mode", choices=["top_n", "max_sentence", "union"], default=None)
    ap.add_argument("--premise-sentences", type=int, default=None)
    ap.add_argument("--no-contested", action="store_true", help="skip the cross-passage Contested check")
    ap.add_argument("--show-verbatim-fails", action="store_true")
    ap.add_argument("--out", default=None, help="write the per-claim table to this CSV")
    ap.add_argument("--human", default=None, help="your completed label sheet (CSV): scores the verifier against it")
    ap.add_argument("--sweep-tau", default=None, help="comma list, e.g. 0.5,0.6,0.7,0.8,0.9 (needs --human)")
    ap.add_argument("--prefill", default=None, help="earlier completed sheet: copy its labels for identical (claim, passage) rows")
    ap.add_argument("--label-sheet", default=None, help="write a blind, shuffled annotation sheet (+ _key.csv) here")
    a = ap.parse_args()

    from src.embed import load_embedder
    from src.utils import load_config
    from src.verify import Verifier

    cfg = load_config(a.config)
    if a.premise_mode:
        cfg.verify["premise_mode"] = a.premise_mode
    if a.no_contested:
        cfg.verify["contested"] = False
    if a.premise_sentences:
        cfg.verify["premise_sentences"] = a.premise_sentences
    tau_e = cfg.verify.tau_entail if a.tau_entail is None else a.tau_entail
    tau_c = cfg.verify.tau_contra if a.tau_contra is None else a.tau_contra
    rows = load_rows(Path(a.run))
    res = reverify(Verifier(cfg, load_embedder(cfg)), rows, tau_e, tau_c)
    print(f"premise_mode={cfg.verify.get('premise_mode', 'top_n')} premise_sentences={cfg.verify.premise_sentences} "
          f"tau_entail={tau_e} tau_contra={tau_c}")
    print(json.dumps(summarise(res), indent=2))
    moved = collections.Counter((x["old"], x["new"]) for x in res if x["old"] != x["new"])
    print("label changes vs the saved run:", {f"{o}->{n}": k for (o, n), k in moved.items()} or "none")
    ans = load_answerable(cfg)
    if ans:
        cc = bool(cfg.get("abstain", {}).get("count_contested", False))
        print("abstention (question level):", json.dumps(
            abstention_summary(res, rows, ans, float(cfg.abstain.min_supported_fraction), cc), indent=2))
    if a.human:
        print("vs human labels:", json.dumps(score_against_human(res, rows, Path(a.human)), indent=2, ensure_ascii=False))
    if a.sweep_tau and a.human:
        print("tau_entail sweep (same claims, same probabilities):")
        for r in sweep_tau(res, rows, Path(a.human), [float(v) for v in a.sweep_tau.split(",")], tau_c):
            print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    if a.label_sheet:
        pre = load_prefill(Path(a.prefill)) if a.prefill else None
        write_label_sheet(res, rows, Path(a.label_sheet), prefill=pre)
        if pre is not None:
            print("rows already labelled in the earlier sheet are prefilled (notes = prefilled); label only the blank rows")
        print(f"wrote {a.label_sheet} and {Path(a.label_sheet).with_name(Path(a.label_sheet).stem + '_key.csv')}")
    if a.show_verbatim_fails:
        for x in res:
            if x["verbatim"] and x["new"] != "Supported":
                print(f"  [{x['qid']}] {x['new']} pEnt={x['p_entail']:.3f} pContra={x['p_contra']:.3f} | {x['text'][:110]}")
    if a.out:
        import csv
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0]))
            w.writeheader()
            w.writerows(res)


if __name__ == "__main__":
    main()