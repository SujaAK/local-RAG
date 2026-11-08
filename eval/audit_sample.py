"""Human citation audit (PRIMARY citation number). Two steps:
  python -m eval.audit_sample export --run runs/<id> [--n 100]
      -> audit_sheet.csv (blind: claim + cited text, no system/score; give to annotators)
         audit_key.csv   (system, model, qid, auto label; keep away from annotators)
  python -m eval.audit_sample score --run runs/<id> --sheet audit_sheet_annotated.csv
      -> human unsupported-claim rate per system (bootstrap CI by question) + agreement with the NLI check.
Annotators fill `label` with Supported | Partly | Unsupported | Contradicted. Use two annotators (columns label_a1/label_a2)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from src.crypto import get_passphrase
from src.store import read_text_private
from src.utils import load_config

from . import stats


def iter_claims(run: Path, pw):
    for f in sorted(list(run.glob("*.jsonl")) + list(run.glob("*.jsonl.enc"))):
        for line in read_text_private(f, pw).splitlines():
            r = json.loads(line)
            ev = {e["id"]: e["text"] for e in r["evidence"]}
            for i in r["shown"]:
                c = r["claims"][i]
                cited = [ev[x] for x in c["evidence_ids"] if x in ev]
                yield {"system": r["system"], "model": r["model"], "qid": r["qid"], "claim": c["text"],
                       "cited_texts": cited}


def export(run: Path, n: int, seed: int, cfg, pw):
    claims = list(iter_claims(run, pw))
    random.Random(seed).shuffle(claims)
    claims = claims[:n]
    checker = None
    if cfg.eval.nli_check_model:
        from src.embed import load_embedder
        from src.verify import Verifier

        # Same premise builder as eval.run_eval (embedding-based top sentences), so auto labels are comparable.
        checker = Verifier(cfg, load_embedder(cfg), model=cfg.eval.nli_check_model,
                           label_order=cfg.eval.nli_check_label_order)
    sheet, key = [], []
    for k, c in enumerate(claims):
        cid = f"a{k:04d}"
        sheet.append({"audit_id": cid, "claim": c["claim"], "cited_evidence": "\n---\n".join(c["cited_texts"]),
                      "label_a1": "", "label_a2": "", "label": ""})
        auto = bool(checker.entails(c["cited_texts"], c["claim"])) if checker and c["cited_texts"] else None
        key.append({"audit_id": cid, "system": c["system"], "model": c["model"], "qid": c["qid"],
                    "auto_supported": auto})
    pd.DataFrame(sheet).to_csv(run / "audit_sheet.csv", index=False)
    pd.DataFrame(key).to_csv(run / "audit_key.csv", index=False)
    print(f"Wrote {len(sheet)} claims to {run/'audit_sheet.csv'} (blind) and audit_key.csv (keep private).")


def score(run: Path, sheet: Path, boot: int):
    ann = pd.read_csv(sheet).merge(pd.read_csv(run / "audit_key.csv"), on="audit_id")
    if {"label_a1", "label_a2"} <= set(ann) and ann["label_a1"].notna().all() and ann["label_a2"].notna().all():
        print("Cohen's kappa (a1 vs a2):", round(stats.cohen_kappa(ann["label_a1"].tolist(), ann["label_a2"].tolist()), 3))
    ann["bad"] = ann["label"].isin(["Unsupported", "Contradicted"]).astype(float)
    ann["human_supported"] = ann["label"].eq("Supported")
    rows = []
    for (system, model), g in ann.groupby(["system", "model"]):
        by_q = g.groupby("qid")["bad"].mean().tolist()  # resample by question
        mean, lo, hi = stats.bootstrap_ci(by_q, boot)
        rows.append({"system": system, "model": model, "claims": len(g), "questions": len(by_q),
                     "human_unsupported_rate": round(mean, 3), "ci95": f"[{lo:.3f}, {hi:.3f}]"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    g = ann.dropna(subset=["auto_supported"])
    if len(g):
        agree = (g["auto_supported"].astype(bool) == g["human_supported"]).mean()
        print(f"NLI-check vs human agreement on Supported/not: {agree:.3f} (n={len(g)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "score"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    run = Path(a.run)
    if a.cmd == "export":
        pw = get_passphrase() if cfg.privacy.encrypt_at_rest else None
        export(run, a.n or cfg.eval.audit_sample_size, a.seed, cfg, pw)
    else:
        score(run, Path(a.sheet), a.boot)


if __name__ == "__main__":
    main()