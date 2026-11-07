# Pre-registration (commit BEFORE touching the test set)  -- date: ____

Fill in, commit, then `touch config/FROZEN` to unlock `--split test` (gate G5).

1. Question set and gold labels: file hash ______
2. Final configs and thresholds (tau_entail, tau_contra, min_supported_fraction, chunk size, embedding model): ______
3. Minimum differences counted as meaningful (placeholders, refine after the dev pilot; proposals, not standards):
   - Verification is useful if it lowers the unsupported-claim rate by >= 10 percentage points while losing <= 20 points of coverage.
   - Hypergraph kept only if it raises multi-hop recall by >= 5 points at <= 2x latency.
4. Hypergraph pilot precision threshold (set in advance, gate G4): ______
5. Statistics: paired bootstrap by question (10,000 resamples), Holm correction across ablation comparisons.
