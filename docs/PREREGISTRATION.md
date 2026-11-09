# Test-Set Evaluation Protocol

**Protocol date:** 2026-11-09  
**Protocol status:** Written after the development-set pilot and before any test-set evaluation. This is not a prospective preregistration of the development experiments.

## 1. Question set and gold labels

- Questions: `data/eval/questions.jsonl`
- SHA-256: `54B2EC88631830B55E300B37DC03BD0D961024A7D9C17F8A03673DFBE6AC5F58`
- Annotated claims: `data/eval/claims_annotated.jsonl`
- SHA-256: `3AD5A04A4E3241117B24869D57EAAA3A529E05ED1E9E01E63FBD0E5BB581E8C8`

The current dataset contains 23 development questions: 19 single-hop and 4 unanswerable. No held-out test questions are currently available in the reported dataset.

## 2. Final configuration and thresholds

- Primary configuration: `config/config.win.topn.yaml`
- SHA-256: `0E731E561D5181F105518E516E8CE2A051B769D9DD0516ECD4499F22807FE249`
- Audit configuration: `config/config.win.audit.yaml`
- SHA-256: `A02BD8F95F74E84C53D32A35DC67E354B7660F5F3853F490E59593CA9A8F4B6E`
- Dependency snapshot: `requirements.lock`
- SHA-256: `638D4B0826CC74C362628F0D24E2AEF7603A42200BF12059D96E0A1EDC484DEC`

Thresholds and settings:

- Entailment threshold (`tau_entail`): 0.70
- Contradiction threshold (`tau_contra`): 0.50
- Minimum supported fraction: 0.50
- Chunk size: 384 tokens
- Chunk overlap: 48 tokens
- Embedding model: `all-MiniLM-L6-v2`
- Verification model: `cross-encoder/nli-deberta-v3-base`
- Premise mode: `top_n`
- Generator: Qwen 2.5 1.5B Instruct, Q4_K_M
- Temperature: 0
- Random seed: 42

## 3. Minimum meaningful differences

The following are project-specific decision thresholds, not established scientific standards:

- Verification is considered potentially useful if it lowers the unsupported-claim rate by at least 10 percentage points while reducing claim coverage by no more than 20 percentage points.
- Hypergraph retrieval will only be considered for retention if it improves multi-hop recall by at least 5 percentage points at no more than twice the baseline latency.

These criteria will be interpreted cautiously given the small development dataset.

## 4. Hypergraph pilot

Hypergraph evaluation is deferred. No precision threshold is claimed because a qualifying pilot has not yet been conducted. No conclusion about hypergraph effectiveness will be drawn from the current experiments.

## 5. Statistical analysis

- Paired bootstrap by question.
- 10,000 resamples.
- Holm correction across ablation comparisons.
- Report effect sizes and 95% confidence intervals.
- Treat results from the 23-question development set as exploratory.
- Do not claim test-set generalization without a separate held-out evaluation.

## 6. Interpretation safeguards

- Raw validator failure rate is not equivalent to factual hallucination rate.
- Claim coverage is not equivalent to answer coverage.
- Automated NLI verification is not independent human factual assessment.
- The AI-assisted audit must not be presented as a completed independent human audit.
- Any deviation from this protocol must be documented and explained.

