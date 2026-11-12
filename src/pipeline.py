"""End-to-end pipeline with system switches S0-S5 (each add-on sits behind a flag)."""
from __future__ import annotations

import time

from . import render
from .generate import generate_baseline, generate_claims
from .pack import pack_evidence
from .store import close_db, load_chunks, open_db
from .utils import OutputCache, get_logger, rpath, sha256_text
from .validate import validate_all

log = get_logger("pipeline")

_S3 = dict(mode="hybrid", rerank=True, gen="claims", validate=True, verify=True, abstain="full")
SYSTEMS = {
    "S0": dict(mode="dense", rerank=False, gen="baseline", validate=False, verify=False, abstain=None),
    "S1": dict(mode="hybrid", rerank=True, gen="baseline", validate=False, verify=False, abstain=None),
    "S2": dict(mode="hybrid", rerank=True, gen="claims", validate=True, verify=False, abstain="flag"),
    "S3": dict(_S3),
    "S3R": dict(_S3, repair=True),         # S3 + citation repair: re-point a failed citation to a passage that entails the claim
    "S4": dict(_S3, offline_guard=True),   # S3 + at-rest encryption (privacy.encrypt_at_rest) + offline mode
    "S5": dict(_S3, graph=True),           # only if gate G4 passes
}


class Pipeline:
    def __init__(self, cfg, passphrase=None, use_cache: bool = True, retriever=None, verifier=None,
                 chunks=None):
        self.cfg, self.passphrase = cfg, passphrase
        if chunks is None:
            self.conn = open_db(cfg, passphrase)
            chunks = load_chunks(self.conn)
        else:
            self.conn = None
        self.chunks = chunks
        if retriever is None:
            from .retrieve import Retriever

            retriever = Retriever(cfg, chunks, passphrase)
        self.retriever, self._verifier = retriever, verifier
        # Private mode: the output cache would store prompts/evidence in plaintext, so it is switched off.
        cache_ok = use_cache and cfg.privacy.cache_outputs and not cfg.privacy.encrypt_at_rest
        if use_cache and not cache_ok:
            log.warning("Output cache disabled (privacy.cache_outputs=false or encrypt_at_rest=true)")
        self.cache = OutputCache(rpath(cfg, "runs_dir") / "cache", enabled=cache_ok)

    @property
    def verifier(self):
        if self._verifier is None:
            from .verify import Verifier

            self._verifier = Verifier(self.cfg, self.retriever.embedder)
        return self._verifier

    def answer(self, question: str, system: str = "S3", model: str | None = None, filters=None,
               tau_entail=None, tau_contra=None) -> dict:
        spec, cfg = SYSTEMS[system], self.cfg
        model = model or cfg.model
        t, t0 = {}, time.perf_counter()

        def lap(name, since):
            t[name] = round(time.perf_counter() - since, 4)
            return time.perf_counter()

        s = time.perf_counter()
        retrieved = self.retriever.retrieve(question, spec["mode"], spec["rerank"], filters=filters)
        if spec.get("graph") and cfg.hypergraph.enabled and self.conn is not None:
            from .hypergraph import expand_chunk_ids, route_to_graph

            ents = route_to_graph(self.conn, question)
            have = {c["chunk_id"] for c in retrieved}
            for cid in expand_chunk_ids(self.conn, ents):
                if cid not in have and cid in self.chunks:
                    retrieved.append(dict(self.chunks[cid], rank=len(retrieved) + 1))
        s = lap("retrieve", s)
        evidence, evidence_text = pack_evidence(retrieved, cfg.pack.context_token_budget)
        s = lap("pack", s)
        gen = (generate_claims if spec["gen"] == "claims" else generate_baseline)(
            cfg, model, question, evidence_text, self.cache)
        s = lap("generate", s)
        raw_claims = gen["claims"]
        checks = validate_all(raw_claims, evidence)
        for c, v in zip(raw_claims, checks):
            c["validators"] = v
        claims, dropped = raw_claims, []
        if spec["validate"] and cfg.validators.enabled:
            if cfg.validators.mode == "drop":
                claims = [c for c in raw_claims if c["validators"]["ok"]]
                dropped = [c for c in raw_claims if not c["validators"]["ok"]]
        s = lap("validate", s)
        if spec["verify"] and cfg.verify.enabled and claims:
            kw = {"repair": True} if spec.get("repair") else {}
            for c, r in zip(claims, self.verifier.verify_claims(claims, evidence, tau_entail, tau_contra, **kw)):
                c.update({"label": r["label"],
                          "nli": {k: v for k, v in r.items() if k not in ("label", "repaired_ids", "repair_trace")}})
                if r.get("repair_trace") is not None:
                    c["repair_trace"] = r["repair_trace"]
                if r.get("repaired_ids"):
                    c["original_evidence_ids"], c["evidence_ids"], c["citation_repaired"] = (
                        c["evidence_ids"], r["repaired_ids"], True)
        s = lap("verify", s)
        abstained = False
        if spec["abstain"] == "flag":
            abstained = render.decide_abstain(claims, gen["insufficient_evidence"], 0, use_fraction=False)
        elif spec["abstain"] == "full":
            abstained = render.decide_abstain(claims, gen["insufficient_evidence"],
                                              cfg.abstain.min_supported_fraction,
                                              count_contested=bool(cfg.abstain.get("count_contested", False)))
        result = {
            "question": question, "system": system, "model": model,
            "retrieved": [c["chunk_id"] for c in retrieved], "evidence": evidence,
            "claims": claims, "dropped_claims": dropped, "n_generated": len(raw_claims),
            "n_repaired": sum(1 for c in claims if c.get("citation_repaired")),
            "thresholds": {"tau_entail": cfg.verify.tau_entail if tau_entail is None else tau_entail,
                           "tau_contra": cfg.verify.tau_contra if tau_contra is None else tau_contra},
            "insufficient_flag": gen["insufficient_evidence"], "abstained": abstained,
            "parse_error": bool(gen.get("parse_error")), "truncated": bool(gen.get("truncated")),
            "latency": {**t, "total": round(time.perf_counter() - t0, 4)},
        }
        result["shown"] = [] if abstained else render.shown_indices(claims)
        result["markdown"] = render.render_markdown(result)
        # Privacy: log IDs/hashes only unless explicitly enabled.
        qlog = question if cfg.privacy.log_raw_text else sha256_text(question)[:12]
        log.info("system=%s q=%s retrieved=%d claims=%d abstained=%s total=%.2fs",
                 system, qlog, len(retrieved), len(claims), abstained, result["latency"]["total"])
        return result

    def close(self):
        if self.conn is not None:
            close_db(self.conn, self.cfg, self.passphrase)