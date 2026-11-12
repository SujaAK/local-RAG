# Privacy-Preserving Local RAG for Academic Literature Review

Implementation of the plan in *Research Survey, Gap Analysis, Implementation Plan and Test Protocol* (Parts B and C).
Fully local: PDFs, queries, embeddings and logs stay on the machine. The pipeline retrieves evidence from a small
paper library, generates **claim-form** answers with citations, checks every claim against its cited passage
(rules plus NLI), and abstains when support is weak.

Status: a working system with a **small, exploratory development-set evaluation** (23 questions, 40 papers,
one 1.5B model, one annotator). Read "Results" and "Limitations" before quoting any number.

## Requirements
- Python 3.11+ (developed and run on 3.13.2, Windows 11, CPU only, about 7 GB RAM).
- A local model server: [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (used for all reported
  results) or [Ollama](https://ollama.com) (default in `config/config.yaml`).
- Models cached locally for offline use: `all-MiniLM-L6-v2` (embeddings), `cross-encoder/ms-marco-MiniLM-L-6-v2`
  (reranker), `cross-encoder/nli-deberta-v3-base` (verifier), and the generator
  (Qwen2.5-1.5B-Instruct, Q4_K_M GGUF).

## Quick start (Windows, PowerShell; the configuration used for the reported results)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# CPU-only machine? install CPU PyTorch FIRST (avoids a multi-GB CUDA download):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt          # exact versions used for the reported runs: requirements.lock
# SETUP PHASE (needs network): download the papers and models once
python scripts/fetch_papers.py --preset core --email you@example.com   # or put open-access PDFs in data/papers/
# RUN PHASE (offline). Terminal 1: model server. Keep -c equal to num_ctx in the config (4096).
llama-server -m "C:\full\path\to\qwen2.5-1.5b-instruct-q4_k_m.gguf" --port 8080 -c 4096
# Terminal 2:
$env:RAG_CONFIG = "config\config.win.v2.yaml"
python -m src.cli build --force           # ingest -> chunk -> embed -> index (--force re-ingests unchanged PDFs)
python -m src.cli ask "What chunk sizes do the papers compare?" --system S3R
streamlit run app/streamlit_app.py
python -m pytest -q                       # one test (Unix sockets) is skipped on Windows
```
Paths that contain spaces must be quoted in PowerShell, for example `--run "runs\<id>"`.

Linux/macOS with Ollama: `bash scripts/setup_models.sh qwen2.5:7b-instruct`, then the same `src.cli` commands. For
8 GB RAM without a GPU use `config/config.tier1.yaml` (3B model; **unmeasured defaults**).

## Systems
| System | What it adds |
|---|---|
| S0 | Dense retrieval baseline, free-text answer |
| S1 | + hybrid BM25/RRF retrieval and cross-encoder rerank |
| S2 | + claim-form generation and deterministic validators V1–V4 |
| S3 | + NLI verification and abstention |
| S3R | S3 + citation repair: a claim that fails against its cited passage is re-pointed to another passage (same paper by default) that entails it |
| S4 | S3 with `privacy.encrypt_at_rest: true` (SQLite, FAISS index and run outputs encrypted; passphrase required) |
| S5 | S3 + hypergraph retrieval expansion (optional; **not evaluated**) |

Every add-on is a config flag; prompts, thresholds and chunk sizes live in `config/`. Configs can extend other
configs through a chain (`extends:`).

## Configurations
| File | Purpose |
|---|---|
| `config/config.yaml` | Base defaults (Ollama backend) |
| `config/config.win.yaml` | Windows, llama.cpp, CPU; `max_sentence` premise |
| `config/config.win.topn.yaml` | `top_n` premise. **Frozen in `docs/PREREGISTRATION.md` (hash recorded); do not edit** |
| `config/config.win.audit.yaml` | As topn, with the independent citation-check model disabled (it is not available offline) |
| `config/config.win.v2.yaml` | **Final configuration**: `premise_mode: union`, Contested check off, `count_contested: false`. Extends topn |
| `config/config.tier1.yaml` | 3B model, 8 GB RAM (unmeasured) |

Why v2 differs from topn (evidence from the development runs, details under Results): a three-sentence joined NLI
premise made the verifier reject claims copied word for word from their cited passage; and the Contested check
fired on unrelated passages. Union premise = a claim counts as supported if either the best single sentence or the
joined top-n premise entails it. `tau_entail` (0.7) and `tau_contra` (0.5) are unchanged from the frozen protocol.

## Map: plan -> code
| Plan | Code |
|---|---|
| S1 Ingest (stages 1-2) | `src/ingest.py` (Unicode NFKC normalisation folds PDF ligatures) |
| S2 Chunking (3) | `src/chunk.py` |
| S3 Embeddings + FAISS (4-5) | `src/embed.py` |
| S4 BM25 + RRF (add-on A4) | `src/sparse.py`, `src/retrieve.py` |
| S5 Retrieval + rerank (7) | `src/retrieve.py` |
| S6 Evidence packing (8) | `src/pack.py` |
| S7 Claim-form generation (9) | `src/generate.py` (JSON retry, salvage of truncated output, duplicate-claim removal) |
| S8 Deterministic validators V1-V4 (10) | `src/validate.py` |
| S9 NLI verification and repair (A1) | `src/verify.py` |
| S10 Abstention + rendering | `src/render.py` |
| S11 Draft review (6, 9) | `src/review_draft.py` (**not evaluated**) |
| S12 Interface | `app/streamlit_app.py`, `src/cli.py` |
| B4 Privacy (A2) | `src/audit.py`, `src/crypto.py`, `src/store.py`, `docs/PRIVACY_AUDIT.md` |
| B5 Reproducibility (A3) | `src/manifest.py`, `src/utils.py` (output cache) |
| B8 Hypergraph (optional A5) | `src/hypergraph.py` (**not evaluated**) |
| C3 Systems | `src/pipeline.py` (`SYSTEMS`) |
| C4/C5 Metrics and statistics | `eval/metrics.py`, `eval/stats.py`, `eval/report.py` |

Scripts: `scripts/fetch_papers.py` (corpus download), `scripts/draft_questions.py` and `scripts/check_questions.py`
(question drafting and validation), `scripts/reverify.py` (re-label saved claims offline and score them against
hand labels), `scripts/diag_nli.py` (NLI diagnostics), `scripts/run_offline.sh` (Linux offline / network-namespace test).

## Evaluation workflow
```powershell
python scripts/check_questions.py --config config\config.win.v2.yaml
# The model server must be running for every system (all of S0-S3R call the LLM).
python -m eval.run_eval --config config\config.win.v2.yaml --split dev --systems S2,S3,S3R --tag my_run --no-nli-metrics --no-cache
python -m eval.report --run runs\<run-folder>
```
- Use `--no-cache` for any latency measurement: S2 and S3 send identical generation prompts, so with the output
  cache on, S3 reuses S2's generations and its latency excludes generation.
- Retrieval metrics are scored at depth `max(ks)` (10), independent of how many chunks are packed into the prompt.
  Keep `retrieval.rerank_pool`, `dense_k` and `sparse_k` at 10 or more.
- `eval.report` runs Holm-corrected paired bootstrap tests over a **confirmatory family only**: retrieval metrics for
  S1 vs S0; coverage and answered rate for S3 vs S2 (and S3R vs S3); latency for consecutive systems. Label counts,
  truncation and similar columns are descriptive.
- Checking the verifier against hand labels (no model server needed):
  ```powershell
  python scripts/reverify.py --config config\config.win.v2.yaml --run "runs\<id>\S3__qwen2.5-1.5b-instruct.jsonl" --label-sheet label_sheet.csv   # blind, shuffled sheet + _key.csv
  # label the sheet (Supported / Partly / Unsupported), save as CSV UTF-8, then:
  python scripts/reverify.py --config config\config.win.v2.yaml --run "runs\<id>\S3__qwen2.5-1.5b-instruct.jsonl" --human label_sheet_completed.csv
  ```
  Optional flags: `--premise-mode`, `--premise-sentences`, `--no-contested`, `--sweep-tau`, `--prefill` (copy labels from an
  earlier sheet).

## Results (development set, exploratory; not test-set performance)
Three runs are kept: `20261109-120850-topn_comparison` (before the bug fixes; superseded and not comparable),
`20261110-214149-fixedA_nocache` (after the code fixes; S0-S3, `config.win.topn.yaml`) and
`20261111-131826-v2_nocache` (final; S2, S3, S3R, `config.win.v2.yaml`). Corpus: 40 papers, 1,250 chunks.
Questions: 23 (19 answerable, 4 unanswerable). Generator: Qwen2.5-1.5B-Instruct Q4_K_M, temperature 0, seed 42.

**Retrieval** (run `20261110-214149`, 19 answerable questions; S1-S3 share one retriever):

| | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| S0 dense | 0.421 | 0.421 | 0.346 | 0.365 |
| S1 hybrid + rerank | 0.737 | 0.895 | 0.493 | 0.590 |

S1 vs S0 recall@10: +0.47, 95% CI 0.26-0.68, Holm-adjusted p < 0.001. The differences in recall@5, MRR and nDCG@10 are not
significant after correction. One question is 5.3 points.

**Answer behaviour** (run `20261111-131826`, 23 questions):

| | Answered | Claim coverage | Questions refused | Refusals on answerable questions |
|---|---|---|---|---|
| S2 | 0.957 (22/23) | 0.957 | 1 | n/a (no verifier) |
| S3 | 0.696 (16/23) | 0.638 | 7 (3 of the 4 unanswerable) | 4/19 (21%) |
| S3R | 0.826 (19/23) | 0.790 | 4 (3 of the 4 unanswerable) | 1/19 (5%) |

S3 shows fewer claims and answers fewer questions than S2 (coverage p_holm 0.0006, answered p_holm 0.013). S3R vs S3
(citation repair) is **not significant** after correction (answered p_holm 0.23). 6 claims were repaired; the one
remaining false refusal (q020) concerns claims that the annotator judged unsupported. Question-level refusal numbers
were recomputed offline from the saved claims with `scripts/reverify.py`.

**Verifier against hand labels** (44 claims, one annotator; setting chosen on these same claims, so **in-sample**):

| Claims | Supported precision | Recall | F1 |
|---|---|---|---|
| S3 claims, labelling pass 1 | 0.875 (28/32) | 0.875 | 0.875 |
| S3 claims, labelling pass 2 | 0.844 (27/32) | 0.900 | 0.871 |
| S3R claims (pass 2 + 6 new rows) | 0.868 (33/38) | 0.971 | 0.917 |
| For comparison: frozen topn setting, pass 1 | 0.944 (17/18) | 0.531 | 0.68 |

95% Wilson intervals are wide (for example precision 27/32: 0.68-0.93). All 6 repaired claims were judged supported by
their new passage (6 of 6). Self-agreement between the two labelling passes: 93.2%, Cohen's kappa 0.85 (0.89 for
Supported vs not); this is one annotator re-labelling, not inter-annotator agreement.

**Reproducibility.** Two independent `--no-cache` runs produced the same claims (after duplicate removal).

**Latency.** Not reliable. Generation on CPU takes roughly 45-60 s per question and dominates; differences between
S2, S3 and S3R are within run-to-run noise. Do not report latency conclusions from these runs.

## Limitations (state these in any report)
- **Evaluation size and design.** 23 development questions (19 single-hop, 4 unanswerable); no multi-hop, conflict or
  held-out test questions. Questions q001-q004 are the template's example rows, kept in the set; most of the others were
  drafted with an LLM from corpus text and reviewed (see the `source` field), which may favour retrieval.
  Treat every result as exploratory. No claim about generalisation is supported.
- **Verifier selection is in-sample.** The union premise and the Contested-off setting were chosen after inspecting
  the same 44 claims that score them. `tau_entail` has almost no effect (NLI probabilities are close to 0 or 1), so
  the precision/recall trade-off cannot be tuned with it. `docs/PREREGISTRATION.md` describes the protocol before the
  test set; the v2 settings are a documented deviation from its `top_n` configuration.
- **What is measured.** Support of a claim by its cited passage, not real-world truth. Single annotator; 44 claims.
- **Known verifier failures.** (1) Text that is quoted as an example or marked as a perturbed claim is approved
  because the words match (mention vs assertion; two q019 claims). (2) Claims that conflate two true facts (q003).
  (3) Claims that depend on an unclear referent ("these operations", "this method") are not self-contained.
  (4) PDF debris (small-caps splits, formula text) breaks some passages and claims.
- **Answerability.** Claim support is not question relevance: a claim can be fully supported yet not answer the
  question (q022). Unanswerable questions are therefore refused only partly (3 of 4).
- **Contested detection** (cross-passage contradiction) produced false positives and is off in v2. It has no
  conflict questions to validate it; the conflict test (T8) was not run.
- **Generator weaknesses.** A 1.5B model can loop (one question produced 19 near-identical claims; duplicates are now
  removed), can exceed the token limit (2 of 23 questions are still flagged `truncated`; their claims are salvaged
  from the cut-off JSON), and sometimes cites the wrong passage (6 of 44 claims; S3R repairs these).
- **Citation metrics.** The independent citation-NLI checker was unavailable offline, so `*_auto` citation metrics were
  not produced (`--no-nli-metrics`). `citation_precision_auto` counts an unsupported claim's citations as irrelevant
  (stricter than the literal ALCE rule).
- **Retrieval metrics.** A chunk hits a gold span only if at least 90% of the span's words appear as one contiguous run in
  the chunk (OCR typos tolerated). Keep gold spans short (about 40 words or fewer) and inside one chunk or its overlap.
  nDCG credits each gold span once.
- **Metadata.** Title and year completeness is 100%, but part of it comes from `data/papers/papers.csv` overrides, so it does
  not validate the extractor. Heading detection is heuristic and noisy (some chunks carry wrong section names).
- **Token counts** use a word-based approximation (`approx_tokens`); keep `pack.context_token_budget` well below `num_ctx`.
- **Environment.** One machine (Windows 11, CPU, about 7 GB RAM). Results should not be generalised to other hardware.
- **Not evaluated:** hypergraph retrieval, draft review, S4 encryption overhead, model sizes other than 1.5B.
- This is not a plagiarism tool and gives no novelty score.

## Privacy modes: what is and is not protected
`privacy.encrypt_at_rest: true` encrypts the SQLite database, the FAISS index and ID map, and the raw run outputs
(`runs/<id>/*.jsonl.enc`). In that mode the **output cache is switched off** (it would hold prompts and evidence in
plaintext) and logs store only hashes and IDs. Still plaintext: `scores.csv` / `report.md` (metrics and question IDs
only), `manifest.json` (configuration and hashes), label and audit sheets (open-access evaluation corpus only; do not
audit private drafts this way), the PDFs themselves, and anything the model server or the OS keeps. Memory is not
protected: the index and database are decrypted into RAM. Out of scope: malware on the host, device theft.

**Offline protection.** `offline_guard` blocks non-loopback IP connections in-process (`connect` and `connect_ex`;
Unix sockets allowed). It is not OS-level isolation. The stronger test, `scripts/run_offline.sh --netns` (Linux and
root), **was not run**. Do not describe the in-process guard as complete network isolation, and do not claim more than this.

## Test coverage
| Test | Where | Status |
|---|---|---|
| T1 code correctness | `python -m pytest` (100+ tests) | done |
| T2 ingestion | `ingest_all` prints metadata completeness | partial (extractor not checked by hand) |
| T3 retrieval | `scores.csv`; S0 vs S1 | done on dev |
| T4 answer quality | human ratings | not done |
| T5 citation fidelity | `eval.audit_sample` and `scripts/reverify.py --human` | partial (44 claims, one annotator; automatic check not run) |
| T6 verifier calibration | `scripts/reverify.py --human`; `eval.metrics` (`ece`, `risk_coverage`) | partial (precision and recall on 44 claims; no calibration curve) |
| T7 abstention | `scripts/reverify.py` (question-level) | done on dev (only 4 unanswerable questions) |
| T8 conflict | `contested_flag`, `silent_side_picking` | not done (no conflict questions) |
| T9 draft review | `src/review_draft.py` | not done |
| T10 efficiency | `latency_total` and per-stage `latency` in run JSONL | measured but unreliable |
| T11 privacy | `src.audit`, `scripts/run_offline.sh`, `docs/PRIVACY_AUDIT.md` | partial (in-process guard only) |
| T12 determinism | `--determinism N` | generation reproduced across two independent runs; the `--determinism` option itself was not run |
| T13 ablation summary | `eval.report` paired table | partial (S0-S3R) |
| T14 model size | `--models` | not done |
| T15/T16 optional | `src/hypergraph.py`; second corpus | not done |

## Reproducing the reported results
1. Check out the commit tagged for the results and install from `requirements.lock`.
2. Download the corpus (`scripts/fetch_papers.py`; `data/papers/papers.csv` and `fetch_report.csv` list what was used) and run
   `python -m src.cli build --force`.
3. Start `llama-server` with the same GGUF and `-c 4096`; the run manifest (`runs/<id>/manifest.json`) records the
   server-reported model path, the configuration and its hash, the lock-file hash, the corpus hash, the seed and the git commit.
4. Run the commands under "Evaluation workflow".