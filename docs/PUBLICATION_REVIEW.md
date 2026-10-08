# Publication-readiness review

This review concerns presentation and data handling only. Frozen experimental results, labels, splits, duplicate decisions, model artifacts, and figures were not changed.

## Presentation changes

- Added a root README spanning question, result, dataset, design, duplicate/leakage control, five model stages, temporal results, uncertainty, class behavior, runtime, limitations, repository map, and reproduction.
- Reused the three existing final-evaluation figures: aggregate macro-F1, per-label 2025 F1, and validation-to-test per-label F1 changes. No figures were regenerated or altered.
- Added reproduction guidance separating inspection of saved results from an independent historical-data reproduction.
- Tightened ignores for nested Matplotlib caches and narrative-bearing review/context files. Added exceptions preserving narrative-free aggregate cohort summaries.

## Currently tracked files to remove from the public index before pushing

`.gitignore` does not untrack existing files. These remain locally available and tracked; no deletion or index mutation was performed during documentation work:

| File | Reason | Suggested treatment |
|---|---|---|
| `cohort/manual_review.html` | Contains complete complaint narratives for local manual review | Keep locally; remove from Git index |
| `benchmark/shortcut_training_contexts.csv` | Contains complaint-text excerpts for feature diagnostics | Keep locally; remove from Git index |
| `finetune_benchmark/epoch_class_diagnostics/mpl-cache/fontlist-v390.json` | Machine-generated font cache | Remove from index; now ignored |
| `temporal_test_evaluation/mpl-cache/fontlist-v390.json` | Machine-generated font cache | Remove from index; now ignored |

The tracked-file inspection found no raw `CCDB_Export_*.csv` exports, narrative-bearing cohort CSVs, `.joblib`, `.safetensors`, or `.npy` model/embedding files. Prediction CSVs expose complaint IDs and labels but not narrative text; they are retained as lightweight evaluation evidence as requested. Manifests contain historical absolute workspace paths and hardware/package details; they are useful provenance but merit review before publication. Historical provenance was not rewritten.

Optional index cleanup, to execute after reviewing the list:

```powershell
git rm --cached -- cohort/manual_review.html benchmark/shortcut_training_contexts.csv
git rm --cached -- finetune_benchmark/epoch_class_diagnostics/mpl-cache/fontlist-v390.json temporal_test_evaluation/mpl-cache/fontlist-v390.json
```

These commands preserve local files. If narrative artifacts were already pushed, untracking only removes them from the current tree; earlier commits still contain them. Review existing public history separately before deciding whether a history rewrite is warranted. No Git history change was made here.

## Before the next push

Review the staged diff and tracked-file list, apply the index cleanup above, and include the new README/docs/check script plus aggregate cohort summaries. Do not stage ignored raw exports, model weights, or local text reviews with `git add -f`. Keep the three final figures and lightweight CSV/JSON evidence unchanged.

The project currently has no repository license; choose an appropriate code license before advertising unrestricted reuse. Historical CFPB data provenance and pretrained-model licensing are separate from licensing the project code. No license was invented or added in this task.

The main scientific reproduction gap is historical narrative-data access: CFPB removed narratives from its current database in 2026. Do not promise byte-identical reproduction from a current download. The detailed constraints are in [REPRODUCING.md](REPRODUCING.md).
