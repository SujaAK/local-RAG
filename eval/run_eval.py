"""Evaluation runner. Usage:
  python -m eval.run_eval --split dev --systems S0,S1,S2,S3 --models qwen2.5:7b-instruct
The test split refuses to run unless config/FROZEN exists (see docs/PREREGISTRATION.md, gate G5)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.crypto import get_passphrase
from src.manifest import write_manifest
from src.pipeline import Pipeline
from src.store import write_text_private
from src.utils import ROOT, get_logger, load_config, rpath

from . import metrics as M

log = get_logger("eval")


def load_questions(cfg, split: str) -> list[dict]:
    p = rpath(cfg, "eval_dir") / "questions.jsonl"
    qs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return [q for q in qs if q.get("split") == split]


def score_one(pipe: Pipeline, q: dict, res: dict, ks, checker) -> dict:
    row = {"qid": q["id"], "type": q.get("type"), "answerable": q.get("answerable", True),
           "abstained": res["abstained"], "n_generated": res["n_generated"],
           "n_shown": len(res["shown"]), "coverage": M.coverage(len(res["shown"]), res["n_generated"]),
           "latency_total": res["latency"]["total"],
           "answered": float((not res["abstained"]) and len(res["shown"]) > 0),
           "parse_error": float(bool(res.get("parse_error"))),
           "n_repaired": res.get("n_repaired", 0),
           "repaired_rate": (res.get("n_repaired", 0) / len(res["shown"])) if res["shown"] else float("nan")}
    retrieved = [pipe.chunks[c] for c in res["retrieved"]]
    row.update(M.retrieval_metrics(retrieved, q.get("gold_evidence", []), ks))
    ev = {e["id"]: e for e in res["evidence"]}
    shown = [res["claims"][i] for i in res["shown"]]
    if checker is not None and shown:   # automatic citation metrics, independent NLI model, same premise builder
        row.update(M.citation_scores(shown, ev, checker.entails))
    # Fabrication on raw claims (before validators) so the validators' effect is measurable.
    raw = res["claims"] + res["dropped_claims"]
    row["fabrication_rate_raw"] = (sum(not c["validators"]["V1"] for c in raw) / len(raw)) if raw else float("nan")
    if q.get("conflict_sides"):
        row.update(M.conflict_scores(res, q["conflict_sides"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--systems", default="S0,S1,S2,S3,S3R")
    ap.add_argument("--models", default=None, help="comma-separated Ollama tags (default: config model)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-nli-metrics", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--determinism", type=int, default=0, help="T12: rerun first 10 questions N times, no cache")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.split == "test" and not (ROOT / "config" / "FROZEN").exists():
        sys.exit("Refusing to run the TEST split: create config/FROZEN after committing "
                 "docs/PREREGISTRATION.md (thresholds frozen, gate G5).")
    if cfg.privacy.offline:
        from src.audit import block_network

        block_network()
    pw = get_passphrase() if cfg.privacy.encrypt_at_rest else None
    models = args.models.split(",") if args.models else [cfg.model]
    systems = args.systems.split(",")
    qs = load_questions(cfg, args.split)[: args.limit]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{args.tag}" if args.tag else "")
    run_dir = rpath(cfg, "runs_dir") / run_id
    write_manifest(cfg, run_dir, models, {"split": args.split, "systems": systems, "n_questions": len(qs)})
    pipe = Pipeline(cfg, pw, use_cache=not args.no_cache)
    checker = None
    if not args.no_nli_metrics:
        from src.verify import Verifier

        chk = cfg.eval.nli_check_model
        if not chk or chk == cfg.verify.model:
            log.warning("CIRCULAR: citation metrics use the same NLI model as S3 (%s). "
                        "Set eval.nli_check_model to a different model and rely on the human audit.", cfg.verify.model)
            checker = pipe.verifier
        else:
            checker = Verifier(cfg, pipe.retriever.embedder, model=chk, label_order=cfg.eval.nli_check_label_order)
        write_manifest(cfg, run_dir, models, {"split": args.split, "systems": systems, "n_questions": len(qs),
                                              "citation_check_nli": chk or cfg.verify.model})
    rows = []
    for model in models:
        for system in systems:
            lines = []
            for q in qs:
                res = pipe.answer(q["question"], system, model)
                lines.append(json.dumps({"qid": q["id"], **res}, default=str))
                r = score_one(pipe, q, res, cfg.eval.ks, checker)
                r.update({"system": system, "model": model})
                rows.append(r)
            # Raw outputs hold full evidence text: written encrypted in private mode (plaintext never on disk).
            write_text_private(run_dir / f"{system}__{model.replace(':', '_').replace('/', '_')}.jsonl",
                               "\n".join(lines) + "\n", cfg, pw)
            log.info("done %s / %s (%d questions)", system, model, len(qs))
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "scores.csv", index=False)

    if args.determinism:
        det = {}
        for model in models:
            for system in systems:
                runs = []
                for _ in range(args.determinism):
                    p2 = Pipeline(cfg, pw, use_cache=False)
                    runs.append([p2.answer(q["question"], system, model)["markdown"] for q in qs[:10]])
                det[f"{system}/{model}"] = M.determinism_rate(runs)
        (run_dir / "determinism.json").write_text(json.dumps(det, indent=2))
        log.info("determinism (exact-match rate): %s", det)
    print(f"Run written to {run_dir}")
    print(f"Next: python -m eval.report --run {run_dir}")


if __name__ == "__main__":
    main()
