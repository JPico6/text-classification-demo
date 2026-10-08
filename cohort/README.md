# Frozen 11-label credit-card cohort

No predictive models built. Raw narrative strings and Issue labels are preserved exactly. No short-text filtering, leakage-span removal, brand masking, label merging, or semantic relabeling was applied.

## Final counts

| split | before_duplicates | kept | removed_duplicate | quarantined_conflict |
| --- | --- | --- | --- | --- |
| train | 2599 | 2424 | 78 | 97 |
| validation | 2861 | 2695 | 114 | 52 |
| test | 6766 | 6521 | 238 | 7 |

## Retained counts by Issue

| Issue | train | validation | test |
| --- | --- | --- | --- |
| Advertising and marketing, including promotional offers | 129 | 151 | 321 |
| Closing your account | 211 | 232 | 565 |
| Fees or interest | 328 | 359 | 836 |
| Getting a credit card | 291 | 310 | 667 |
| Incorrect information on your report | 94 | 84 | 336 |
| Other features, terms, or problems | 322 | 350 | 810 |
| Problem when making payments | 168 | 212 | 333 |
| Problem with a company's investigation into an existing problem | 46 | 46 | 185 |
| Problem with a purchase shown on your statement | 682 | 782 | 2054 |
| Struggling to pay your bill | 37 | 35 | 99 |
| Trouble using your card | 116 | 134 | 315 |

## Policy and review choices surfaced

- The 11 labels are frozen from the assessment (November 2024 support ≥30 before duplicate removal); support is not re-thresholded after quarantine.

- Receipt splits are unchanged: November 2024 train; December 2024 validation; November–December 2025 test. No future observation moves into an earlier split.

- Duplicate graph covers all 12,363 available credit-card narratives, including the 137 rare-label audit rows. Thus conflicts with an out-of-scope label still quarantine the in-scope records. Other products are outside this cohort graph.

- Exact hashes use stripped raw text, then normalized hashes use the assessment normalization. These transformations affect matching only. Existing near-duplicate evidence adds all 309 pairs; each Jaccard score was recomputed and checked. No new similarity threshold or candidate search was introduced.

- The prior assessment proposed manual confirmation of near-duplicates. Following the current instruction to use already-identified evidence, those pairs are conservatively treated as graph edges, with their unadjudicated status recorded. They are not asserted to be confirmed same-consumer events; review can later reject an edge and trigger a rebuild.

- Contradictory-label connected components are quarantined in full, including their earliest records. No majority vote or relabeling. Non-conflicting components retain only the earliest receipt record; same-day ties use the smallest numeric Complaint ID. This tie-break does not imply a known intraday order.

- Kept singletons remain untouched. All excess copies, including excess training copies, are removed. Normalized matching and near edges are transitive; a component need not have pairwise Jaccard ≥0.80 between every member.

- Leakage and ambiguity flags remain review-only. Representative examples and new leakage samples use development data. Previously identified ambiguity and contradictory groups may expose test examples; no model/feature design is performed.

## Artifacts

- train.csv, validation.csv, test.csv: Complaint ID, original narrative, Issue. Only narrative is intended as input; ID is an audit key.

- *_metadata.csv: source, receipt date, split, label, and group lineage, separated from model inputs.

- counts_overall.csv and counts_by_issue.csv: reconciled mutually exclusive final statuses, with exact/normalized/near attribution. Quarantine takes precedence over removal.

- stage_counts.csv: cumulative status snapshots after exact, normalized, and near edges; do not sum these stages. A row can move from removed to quarantined as evidence accumulates.

- disposition_ledger.csv: every available credit-card narrative, including out-of-scope records; hashes, component membership, representative ID, and final disposition.

- duplicate_edges.csv: complete graph edge evidence. contradictory_label_quarantine.csv: full review population; no records here are retained. rare_label_audit.csv: all four excluded labels, without merging.

- duplicate_components.csv: repeated-component sizes, label sets, date bounds, and split coverage, including out-of-scope members.

- manual_review.html: compact expandable reading artifact. manual_review.csv: same sample with full original text and blank reviewer-decision columns. No Company, ZIP, State, or downstream response metadata is copied.

- manifest.json: fixed labels, source/evidence checksums, source funnel, matching policy, and validation results.

## Validation

{
  "counts_reconcile": true,
  "one_representative_per_component": true,
  "no_normalized_duplicates_in_retained_splits": true,
  "no_evidence_edge_has_two_retained_endpoints": true,
  "receipt_split_unchanged": true,
  "conflicting_groups_fully_quarantined": true,
  "representatives_are_earliest": true,
  "raw_narratives_and_labels_preserved": true
}

Counts reconcile per label and split. Retained components are unique across splits, normalized texts do not repeat across retained datasets, all contradictory components are excluded, and each representative is earliest. Exported narrative and label fields were re-read and compared with source fields. Existing source hashes match the assessment. The approximate evidence remains non-exhaustive; this does not prove independence of all retained complaints.

Duplicate decisions use evidence from the full historical snapshot: a later contradictory label can quarantine an earlier record. This is retrospective decontamination, not a reconstruction of decisions available at the historical training date. No later observation is moved earlier. Investigation training support falls from 142 to 46; the assessed 11-label scope is preserved rather than silently changed.

## Rebuild

Run build_credit_card_cohort.py with Python and pandas from this repository. Deterministic source order, graph IDs, date/ID ordering, review selection, and output sorting are used. Source changes cause a failure rather than silent cohort drift. Reviewer CSV edits are not consumed by this script.
