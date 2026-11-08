"""Command line: python -m src.cli build | ask | review | audit"""
from __future__ import annotations

import argparse
import json

from .crypto import get_passphrase
from .utils import load_config


def main():
    ap = argparse.ArgumentParser(prog="local-rag")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="ingest PDFs, chunk, embed, index")
    b.add_argument("--force", action="store_true", help="re-parse PDFs even if unchanged")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--system", default="S3")
    a.add_argument("--model", default=None)
    a.add_argument("--debug", action="store_true", help="also print retrieved text, every claim with label and NLI scores")
    a.add_argument("--snippet", type=int, default=220, help="characters of each retrieved passage shown by --debug")
    r = sub.add_parser("review")
    r.add_argument("file")
    sub.add_parser("audit")
    h = sub.add_parser("hypergraph", help="optional B8 steps (only after gate G4 decision)")
    h.add_argument("action", choices=["build", "export-pilot", "precision"])
    h.add_argument("--max-papers", type=int, default=None)
    h.add_argument("--model", default=None)
    h.add_argument("--csv", default="runs/hypergraph_pilot.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pw = get_passphrase() if cfg.privacy.encrypt_at_rest and args.cmd != "audit" else None

    if cfg.privacy.offline and args.cmd in ("ask", "review"):
        from .audit import block_network

        block_network()

    if args.cmd == "build":
        from .chunk import chunk_all
        from .embed import build_index
        from .ingest import ingest_all

        print(json.dumps(ingest_all(cfg, pw, args.force)))
        print("chunks:", chunk_all(cfg, pw))
        print("indexed:", build_index(cfg, pw))
    elif args.cmd == "ask":
        from .pipeline import Pipeline

        p = Pipeline(cfg, pw)
        res = p.answer(args.question, args.system, args.model)
        print(res["markdown"])
        if args.debug:
            from .render import render_debug

            premises = {}
            if res["claims"] and any("nli" in c for c in res["claims"]):
                by_id = {e["id"]: e for e in res["evidence"]}
                for i, c in enumerate(res["claims"]):
                    cited = [by_id[x]["text"] for x in c.get("evidence_ids", []) if x in by_id]
                    if cited:
                        premises[i] = p.verifier.score(c["text"], cited)[0]
            print(render_debug(res, premises, args.snippet))
        print("\nlatency:", res["latency"])
    elif args.cmd == "review":
        from pathlib import Path

        from .pipeline import Pipeline
        from .review_draft import review_draft

        p = Pipeline(cfg, pw)
        print(json.dumps(review_draft(p, Path(args.file).read_text(encoding="utf-8")), indent=2)[:20000])
    elif args.cmd == "hypergraph":
        from . import hypergraph as hg

        if args.action == "build":
            print(hg.build_hypergraph(cfg, args.model or cfg.model, pw, args.max_papers))
        elif args.action == "export-pilot":
            print("rows to label:", hg.export_pilot_csv(cfg, args.csv, pw, args.max_papers or 5))
        else:
            print(hg.pilot_precision(args.csv))
    elif args.cmd == "audit":
        from .audit import run_audit

        print(json.dumps(run_audit(cfg), indent=2))


if __name__ == "__main__":
    main()