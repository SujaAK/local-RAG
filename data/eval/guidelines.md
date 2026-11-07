# Annotation guidelines (write and freeze BEFORE running any system)

## Question types
single_hop, multi_hop, conflict, unanswerable, draft. Each question has `gold_evidence`: a list of
`{paper_id, text}` where `text` is a verbatim span from the paper (chunking-independent).
Unanswerable questions have `answerable: false` and no gold evidence.
Conflict questions list `conflict_sides`: the paper_ids of the disagreeing positions.

## Claim labels (claims_annotated.jsonl)
- **Supported**: the cited evidence fully states or directly implies the claim.
- **Partly**: part of the claim is supported, part is not.
- **Unsupported**: the cited evidence does not state or imply the claim.
- **Contradicted**: the cited evidence states the opposite.

## Process
Each author annotates items the other wrote. An independent annotator labels >= 20%.
Report Cohen's kappa; settle disagreements by discussion. Annotate BEFORE running the system.
Split into dev and test by question before any system run. Never tune on test.
