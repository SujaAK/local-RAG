# Privacy-Preserving Local RAG for Academic Literature Review

Implementation of the plan in *Research Survey, Gap Analysis, Implementation Plan and Test Protocol*
(Parts B and C). Fully local: PDFs, queries, embeddings and logs never leave the machine.

Needs Python 3.11+ (3.12 recommended) and [Ollama](https://ollama.com).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
# CPU-only machine? install CPU PyTorch FIRST (avoids a multi-GB CUDA download):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip freeze > requirements.lock
bash scripts/setup_models.sh qwen2.5:7b-instruct      # SETUP PHASE (network). 8 GB / no GPU: see Tier 1 below
# put open-access PDFs in data/papers/ (optionally fill data/papers/papers.csv to override metadata)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OLLAMA_HOST=127.0.0.1     # RUN PHASE (offline)
python -m src.cli build                                # ingest -> chunk -> embed -> index
# 8 GB RAM, no GPU:  export RAG_CONFIG=config/config.tier1.yaml   (3B model, num_ctx 4096, smaller budget; UNMEASURED defaults)
python -m src.cli ask "What chunk sizes do the papers compare?" --system S3
python -m src.cli review my_draft.txt
streamlit run app/streamlit_app.py
python -m pytest -q                                    # T1
```

## Map: plan -> code
| Plan | Code |
|---|---|
| S1 Ingest (stages 1-2) | `src/ingest.py` |
| S2 Chunking (3) | `src/chunk.py` |
| S3 Embeddings + FAISS (4-5) | `src/embed.py` |
| S4 BM25 + RRF (add-on A4) | `src/sparse.py`, `src/retrieve.py` |
| S5 Retrieval + rerank (7) | `src/retrieve.py` |
| S6 Evidence packing (8) | `src/pack.py` |
| S7 Claim-form generation (9) | `src/generate.py` |
| S8 Deterministic validators V1-V4 (10) | `src/validate.py` |
| S9 NLI verification (A1) | `src/verify.py` |
| S10 Abstention + rendering | `src/render.py` |
| S11 Draft review (6, 9) | `src/review_draft.py` |
| S12 Interface | `app/streamlit_app.py` |
| B4 Privacy (A2) | `src/audit.py`, `src/crypto.py`, `src/store.py`, `docs/PRIVACY_AUDIT.md` |
| B5 Reproducibility (A3) | `src/manifest.py`, `src/utils.py` (output cache) |
| B8 Hypergraph (optional A5) | `src/hypergraph.py` |
| C3 Systems S0-S5 | `src/pipeline.py` (`SYSTEMS`) |
| C3 Tests T1-T16 | `tests/`, `eval/` (see below) |
| C4/C5 Metrics + statistics | `eval/metrics.py`, `eval/stats.py`, `eval/report.py` |

## Systems (C3)
S0 dense baseline | S1 + hybrid BM25/RRF + rerank | S2 + claim form + validators | S3 + NLI verification + abstention |
S4 = S3 + at-rest encryption (`privacy.encrypt_at_rest: true`, enter passphrase) + offline guard | S5 = S3 + hypergraph (only if gate G4 passes; set `hypergraph.enabled: true`).
Every add-on is a config flag. Prompts, thresholds and chunk sizes live in `config/`.

## Evaluation workflow (gates B7)
1. **G0** Baseline gives real citations on 10 hand-checked questions (`ask --system S0`).
2. **G1** Write `data/eval/questions.jsonl` (~150 questions, split dev/test **by question**) following `data/eval/guidelines.md`. Do not run systems on it yet.
3. **G2** `python -m eval.run_eval --split dev --systems S0,S1 --models qwen2.5:7b-instruct`
4. **G3** Annotate >= 300 dev claims (`claims_annotated.jsonl`); tune `tau_entail`, `tau_contra` on dev only; risk-coverage via `eval.metrics.risk_coverage`.
5. Fill `docs/PREREGISTRATION.md`, commit, `touch config/FROZEN` (**G5**), then run `--split test` **once**.
6. `python -m eval.report --run runs/<id>` -> tables with bootstrap CIs, paired comparisons, Holm correction.
7. Determinism (T12): add `--determinism 5`. Privacy audit (T11): `docs/PRIVACY_AUDIT.md`.

## Test coverage (C3)
| Test | Where |
|---|---|
| T1 code correctness | `python -m pytest` |
| T2 ingestion | `ingest_all` prints metadata completeness (DoD >= 95%); inspect 20 papers manually |
| T3 retrieval | `scores.csv`: recall@5/10, nDCG@10, MRR, all-gold; compare S0 vs S1; sweep `chunk`/`embedding` in config on dev |
| T4 answer quality | human ratings (not automated here) |
| T5 citation fidelity | **Primary: human audit** `python -m eval.audit_sample export/score` (blind sheet, 2 annotators, kappa). Cross-check: `citation_*_auto` in `scores.csv` from an independent NLI model + `fabrication_rate` |
| T6 verifier calibration | `eval.metrics`: `confusion`, `macro_f1`, `ece`, `risk_coverage` on your annotated pairs |
| T7 abstention | `eval.metrics.abstention_prf` over `scores.csv` (`answerable`, `abstained`) |
| T8 conflict | `contested_flag`, `silent_side_picking` columns (questions with `conflict_sides`) |
| T9 draft review | `src.review_draft`; rate outputs with >= 3 people |
| T10 efficiency | `latency_total` and per-stage `latency` in run JSONL |
| T11 privacy | `src.audit`, `scripts/offline_run.sh`, `docs/PRIVACY_AUDIT.md` |
| T12 determinism | `--determinism N` |
| T13 ablation summary | `eval.report` paired table |
| T14 model size | run with 3-4B and 7-8B tags in `--models` |
| T15/T16 optional | `src/hypergraph.py`; copy config bundle for a second corpus |

## Privacy modes: what is and is not protected
`privacy.encrypt_at_rest: true` encrypts the SQLite database, the FAISS index + ID map, and the raw run outputs
(`runs/<id>/*.jsonl.enc`). In that mode the **output cache is switched off** (it would hold prompts/evidence in plaintext)
and logs store only hashes/IDs. Still plaintext: `scores.csv` / `report.md` (metrics and question IDs only), `manifest.json`
(config and hashes), `audit_sheet.csv` (only for the open-access evaluation corpus; do not audit private drafts this way),
the PDFs themselves, and anything Ollama or the OS keeps (check your Ollama server logs/history). Memory is not protected:
the index and database are decrypted into RAM. Out of scope: malware on the host, device theft. Do not claim more than this in the paper.

## Caveats you should know

- **Evaluation status:** The current development dataset contains 23 questions: 19 single-hop and 4 unanswerable. It contains no multi-hop or conflict questions and has no held-out test split. Results are exploratory and must not be presented as test-set performance or evidence of generalization.
- **System comparison:** S1, S2 and S3 were evaluated on the same 23 development questions using Qwen 2.5 1.5B Instruct. Retrieval metrics were identical across systems. S3 had significantly lower claim coverage and answer rate than S2 after Holm correction. This demonstrates more conservative output behaviour, not improved factual correctness.
- **Latency interpretation:** S3's lower observed latency is confounded by its higher abstention rate. Do not interpret it as an independent efficiency improvement.
- **Citation metrics:** Independent automatic citation NLI metrics were disabled in the reported comparison because the independent checker model was unavailable offline. The verifier still operated within S3. Do not report independent automatic citation accuracy from this run.
- **Human audit:** An AI-assisted audit sheet was generated, but its labels have not been independently validated by human annotators. Do not report its scores as human factual-accuracy measurements or claim inter-annotator agreement.
- **Raw fabrication metric:** `fabrication_rate_raw` represents validator failures, not factual hallucination or factual correctness.
- **Claim coverage:** The reported coverage metric is claim retention (`n_shown / n_generated`), not answer coverage.
- **Retrieval metrics:** A chunk hits a gold span only if at least 90% of the span's words appear as one contiguous run in the chunk (OCR typos tolerated). Keep gold spans short (approximately 40 words or fewer) and inside one chunk or its overlap. nDCG credits each gold span once (maximum 1).
- **Token counts:** `approx_tokens` uses word-based approximations. Keep `pack.context_token_budget` well below `num_ctx`.
- **NLI verification:** Verify `verify.label_order`. General NLI models can perform poorly on scientific text; calibrate on domain-specific pairs before making strong reliability claims.
- **Heading/metadata extraction:** This is heuristic. Check a sample of papers and use `papers.csv` to override incorrect metadata.
- **Offline protection:** `offline_guard` blocks non-localhost IP connections in-process (Unix sockets allowed). The OS-level test (`scripts/run_offline.sh --netns`, Linux + root) is the stronger isolation test. Do not describe the in-process guard as complete OS-level network isolation.
- **Hypergraph:** Hypergraph retrieval expansion is implemented but has not been experimentally validated. No claim about its effectiveness is supported by the current results.
- **Privacy:** Encryption at rest does not protect decrypted data in RAM, source PDFs, or data retained by Ollama or the operating system. See the privacy section above.
- **Streamlit and environment:** The local application and model-backed pipeline were exercised in the Windows environment described in the experiment manifest. Results should not be generalized to other hardware without testing.
- This is not a plagiarism tool and gives no novelty score