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
- **Citation metrics:** the `*_auto` columns use an NLI model *different from* the verifier (`eval.nli_check_model`) and the
  same top-sentence premise builder as S3 (`src.verify.build_premise`). If you set both to the same model the run warns
  "CIRCULAR". They are a cross-check; the human audit is the number to report first. Confirm the check model's label order on its model card.
- **Retrieval metrics:** a chunk hits a gold span only if >= 90% of the span's words appear as one contiguous run in the chunk
  (OCR typos tolerated). Keep gold spans short (<= ~40 words) and inside one chunk or the chunk overlap. nDCG credits each gold span once (<= 1).
- **Token counts are word-based approximations** (`approx_tokens`); keep `pack.context_token_budget` well below `num_ctx`.
- **Verify the NLI label order** (`verify.label_order`); general NLI models can drop sharply on scientific text (R20), so calibrate on your own pairs.
- **Heading/metadata extraction is heuristic.** Check 20 papers and use `papers.csv` to override.
- `offline_guard` blocks non-localhost IP connections in-process (Unix sockets allowed); the OS-level test (T11, `scripts/run_offline.sh --netns`, Linux + root) is the real pass criterion.
- **Hypergraph (optional):** `python -m src.cli hypergraph build --max-papers 5` -> `export-pilot` -> label the `correct` column -> `precision`.
  Stop if precision is below your pre-set threshold. Entity normalisation is an alias table only (`config/aliases.yaml`, optional); no embedding-similarity merge.
  Retrieval expansion (S5) is implemented but has no evaluation data.
- **Untested here:** anything needing Ollama, sentence-transformers/FAISS models, the NLI models or Streamlit (no models in the build sandbox). Unit tests cover the pure logic, the pipeline with fakes, the encrypted store, and the offline guard.
- Eval and annotation files in `data/eval/` contain EXAMPLE rows only; replace them.
- This is not a plagiarism tool and gives no novelty score. Numbers in the PDF's Part A come from snippets; verify against the original papers (C9).
