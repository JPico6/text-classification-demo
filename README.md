# Does more NLP complexity improve complaint classification?

A controlled comparison of lexical baselines, frozen sentence embeddings, and transformer fine-tuning for predicting the CFPB `Issue` label from published credit-card complaint narratives.

## Research question

**When does additional NLP model complexity actually improve real-world text classification?**

This project moves from simple phrase rules to sparse learned features, frozen pretrained representations, and a fine-tuned encoder. Each step is evaluated on the same frozen 11-label cohort. Training uses November 2024, development decisions use December 2024, and the selected models are tested on November–December 2025.

## Key result

Additional representational complexity did not produce a demonstrated macro-F1 improvement in this experiment. TF-IDF and frozen MiniLM had similar temporal-test performance; their small observed difference was not resolved by the paired bootstrap interval. Both outperformed **this specific fine-tuning protocol** under the stated bootstrap assumptions.

| Model | December 2024 validation macro-F1 | 2025 temporal-test macro-F1 | Change |
|---|---:|---:|---:|
| TF-IDF + logistic regression | 0.5111 | 0.5331 | +0.0220 |
| Frozen MiniLM + logistic regression | 0.4983 | 0.5275 | +0.0292 |
| Fine-tuned MiniLM | 0.4588 | 0.4910 | +0.0322 |

![Validation and temporal-test macro-F1, with test bootstrap intervals](temporal_test_evaluation/model_macro_f1_validation_test.png)

The fine-tuned model's weakness was concentrated in particular labels. Its near-absence of `Other` and `Investigation` predictions was visible during validation and replicated on the later test cohort. Aggregate scores alone would obscure this behavior.

**Experiment status: complete and frozen.** The test has been scored. No refitting on training plus validation, test-driven tuning, label changes, or additional predictive development followed the final evaluation.

## Dataset and prediction task

The input is only `Consumer complaint narrative`. The target is the **CFPB-published Issue label**, not an objective determination of misconduct or verified ground truth about a consumer's underlying problem. CFPB describes the taxonomy fields as the consumer's original form selections; this project reproduces that published label. See the [CFPB field/taxonomy release notes](https://cfpb.github.io/api/ccdb/release-notes.html).

The source snapshots contain 1,600,190 records across November–December 2024 and November–December 2025. They cover all products; the modeling cohort retains credit-card complaints with nonempty narratives, including both credit-card sub-products. The 11-label scope was chosen from November 2024 support of at least 30 before duplicate removal. Four low-support labels were excluded, without merging them into `Other`. The scope was not re-thresholded after quarantine.

| Split | Receipt dates | Retained narratives | Purpose |
|---|---|---:|---|
| Training | Nov. 1–30, 2024 | 2,424 | Learned preprocessing, coefficients, encoder updates, loss weights |
| Validation | Dec. 1–31, 2024 | 2,695 | Configuration selection and fine-tuning checkpoint selection |
| Temporal test | Nov. 1–Dec. 31, 2025 | 6,521 | Final evaluation of the actual selected models |

There is no January–October coverage in either snapshot. Narrative availability is selected: the task is not representative of all complaints. Source hashes, missingness, taxonomy changes, and selection effects are documented in the [original assessment](profiling/findings.md); final counts and exact label names are in the [cohort documentation](cohort/README.md).

## Experimental design

Macro-F1 is the primary selection metric because the 11-class task is imbalanced: it gives each Issue equal weight. Weighted-F1, accuracy, balanced accuracy, and per-label precision/recall/F1/support provide complementary views.

All learned vocabulary, IDF, regression coefficients, class weights, and transformer updates use November training data only. December validation selects among small explicit configuration sets; there is no broad automated search. Test inference uses the saved selected models, without a November-plus-December refit. Seeds, package versions, source hashes, model identities, epoch histories, and predictions are recorded in the stage manifests.

The 2025 snapshot was descriptively inspected during the initial data audit and retrospective duplicate control. It was **not** used for predictive feature selection, model selection, or tuning. The holdout was untouched by predictive inference until the final evaluation; it was not unseen in every descriptive sense.

## Leakage and duplicate control

Exact duplicates were linked first, followed by normalized duplicates and the 309 already-identified near-duplicate evidence pairs. Exact hashes strip boundary whitespace; normalized hashes use lowercased alphanumeric tokens. These transformations are for duplicate detection only, not narrative cleaning.

Duplicate evidence forms connected components across the historical snapshots. A nonconflicting component retains its earliest receipt record, with numeric complaint ID as a same-day tie-break. Contradictory-label components are quarantined in full for review, rather than relabeled by majority vote. Future records are never moved into earlier splits. Near-duplicate edges are conservative, unadjudicated evidence, not proof of same-consumer events.

| Split | Before duplicate control | Retained | Duplicate copies removed | Contradictory-label quarantine |
|---|---:|---:|---:|---:|
| Training | 2,599 | 2,424 | 78 | 97 |
| Validation | 2,861 | 2,695 | 114 | 52 |
| Test | 6,766 | 6,521 | 238 | 7 |

Narratives can contain Issue-like wording, procedural language, boilerplate, or multiple topics. Manual review and positive-feature diagnostics surfaced these concerns. No validation-driven brand masking, leakage-span deletion, short-text filtering, or label merging was applied. Company, location, response fields, and complaint IDs are not predictive inputs. The frozen policy is documented in [cohort/README.md](cohort/README.md).

## Models: a progression in complexity

| Stage | Representation and decision rule | Additional complexity | Validation macro-F1 |
|---|---|---|---:|
| Majority baseline | Always predict the most common training Issue | No text representation | 0.0409 |
| Lexical baseline | 67 fixed topic phrases, deterministic ties and training-frequency fallback | Explicit domain rules; no learned semantic representation | 0.2842 |
| TF-IDF + multinomial logistic regression | Word unigrams/bigrams; 50,174 selected features | Learned lexical vocabulary, IDF, and linear class boundaries | 0.5111 |
| Frozen MiniLM + multinomial logistic regression | Fixed 384-dimensional pretrained embeddings | Pretrained contextual encoder; only regression is trained | 0.4983 |
| Fine-tuned MiniLM | Pretrained encoder plus an 11-class task head | Task-specific encoder updates and GPU training | 0.4588 |

The two simple baselines were evaluated on validation only; they were not included in the three-model final test evaluation.

**TF-IDF:** `tfidf_df2_C1.0_balanced`, selected from eight combinations of `min_df ∈ {1,2}`, `C ∈ {1,4}`, and balanced/unweighted classes. The tokenizer preserves negation, numbers, and punctuation tokens; there is no stop-word removal. Vocabulary and IDF are fitted only on training.

**Frozen MiniLM:** `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Complete narratives are tokenized into nonoverlapping 254-wordpiece content chunks plus special tokens. Normalized chunk embeddings are averaged with content-token-count weights, then L2-normalized. Transformer weights remain fixed. `embedding_C4.0_balanced` was selected from four regression configurations (`C ∈ {1,4}`, balanced/unweighted).

**Fine-tuned MiniLM:** the same pretrained encoder, masked mean pooling/L2 normalization, dropout 0.1, and an 11-class linear head. Native input support was verified from configuration, the 512×384 position tensor, and a 512-position forward pass. The encoder supports 512 total positions; the SentenceTransformer wrapper's default is 256. Fine-tuning uses the first 510 content wordpieces plus special tokens, without extending positional embeddings.

One predefined protocol used inverse-frequency weighted cross-entropy `N/(11 × class_count)`, AdamW, encoder learning rate `2e-5`, head learning rate `1e-3`, batch size 8 with four-step accumulation, seed `20261008`, and at most eight epochs. December macro-F1 selected the checkpoint, with patience 2. Epoch 8 was selected; the epoch cap was reached before early stopping triggered. See [the full protocol and truncation audit](finetune_benchmark/README.md).

## Validation and frozen temporal-test results

All three selected models improved their aggregate metrics on this particular later cohort. That observation does not imply that future temporal transfer generally improves performance.

| Model | Test macro-F1 | Test weighted-F1 | Test accuracy | Test balanced accuracy |
|---|---:|---:|---:|---:|
| TF-IDF | 0.5331 | 0.6386 | 0.6528 | 0.5405 |
| Frozen MiniLM | 0.5275 | 0.6274 | 0.6235 | 0.5695 |
| Fine-tuned MiniLM | 0.4910 | 0.6027 | 0.6433 | 0.5448 |

The [final evaluation report](temporal_test_evaluation/README.md) contains both periods' full metrics, per-label temporal differences, prediction shares, and confusion matrices. The authoritative machine-readable results are [aggregate_metrics.csv](temporal_test_evaluation/aggregate_metrics.csv) and [per_label_metrics.csv](temporal_test_evaluation/per_label_metrics.csv).

## Statistical uncertainty

The final analysis uses 10,000 ordinary complaint-level bootstrap replicates, seed `20261009`. Each replicate resamples 6,521 indices with replacement and uses the **same indices across all models**. Intervals are 2.5th/97.5th percentiles of fixed 11-label macro-F1 or paired differences.

| Test macro-F1 difference | Estimate | Paired 95% interval |
|---|---:|---:|
| TF-IDF − frozen MiniLM | +0.0056 | [−0.0104, +0.0211] |
| TF-IDF − fine-tuned MiniLM | +0.0421 | [+0.0269, +0.0569] |
| Frozen MiniLM − fine-tuned MiniLM | +0.0365 | [+0.0223, +0.0507] |

TF-IDF is **not established as statistically superior to frozen MiniLM**. An interval containing zero also does not prove equivalence. Both comparisons against the tested fine-tuning protocol exclude zero under the stated assumptions. These intervals condition on the fitted models and frozen sample; they exclude development-selection uncertainty, label uncertainty, and future population change. Undetected related complaints could weaken the independence assumption. See [all model and paired intervals](temporal_test_evaluation/macro_f1_bootstrap_intervals.csv).

## Class-level findings

![Per-label 2025 F1 across the three selected models](temporal_test_evaluation/per_label_2025_f1.png)

The fine-tuned model reproduced several topical labels well, but almost ceased assigning two classes:

| Fine-tuned label on 2025 | True examples | Predicted examples | Recall | F1 |
|---|---:|---:|---:|---:|
| Other features, terms, or problems | 810 | 33 | 0.0296 | 0.0569 |
| Problem with a company's investigation into an existing problem | 185 | 6 | 0.0000 | 0.0000 |

This behavior was already visible on December validation. `Other` predictions fell from 17 at epoch 1 to 5 at epochs 4–5, then partly recovered to 12 at epoch 8. Investigation received 0–2 predictions per epoch and no correct predictions. Thus suppression was present by the first recorded epoch; it was not a steady disappearance first caused by temporal transfer. See the [all-epoch flow diagnostic](finetune_benchmark/epoch_class_diagnostics/README.md).

`Other` had 322 training examples, the third-largest training class. Its broad/residual definition, heterogeneous narratives, and overlapping topical content are plausible contributors to difficult class boundaries. `Investigation` had only 46 training examples and encodes a procedural distinction that can overlap the underlying topic. These interpretations are consistent with the diagnostics, **not causally established explanations**. Optimization and input handling are also possible contributors.

The later cohort does not preserve every class-level ordering: frozen MiniLM has higher Investigation F1 than TF-IDF, while fine-tuned MiniLM improves markedly on Incorrect report information. Struggling-to-pay F1 improves for all three models. These are descriptive differences, not additional model-selection decisions.

![Per-label F1 change from validation to temporal test](temporal_test_evaluation/per_label_f1_temporal_change.png)

## Complexity and runtime

| Approach | Test representation/inference cost | Lightweight classifier prediction | Hardware |
|---|---:|---:|---|
| TF-IDF | 1.33 s transform | 0.016 s | CPU |
| Frozen MiniLM | 238.58 s complete-document encoding | 0.015 s | CPU |
| Fine-tuned MiniLM | 3.13 s tokenization + 27.87 s joint encoder/head inference | Included in forward pass | RTX 3070 Laptop GPU |

These costs exclude model loading. Frozen encoding includes tokenization, 11,104 chunk forwards, and aggregation. Fine-tuned GPU inference is **not a hardware-normalized speed comparison** with the CPU runs. Exact environments and cost boundaries are in [runtime_summary.json](temporal_test_evaluation/runtime_summary.json).

The practical finding is not just that sparse features are cheap: considerably greater representational complexity did not produce a demonstrated macro-F1 advantage here. Runtime, class behavior, and uncertainty all matter when assessing whether added complexity is useful.

## What the experiment suggests

Strong lexical baselines deserve to be measured before adding transformer complexity. Frozen semantic representations can be competitive without improving the primary metric. Fine-tuning can improve individual topical classes while worsening broad or procedural labels enough to lower macro-F1. Per-class prediction mass and confusion flows expose failures that aggregate scores can conceal.

This is evidence about one controlled task and protocol, not a general verdict against transformer fine-tuning. No further development was undertaken after the final test was scored.

## Limitations

- CFPB Issue is the published complaint label, not verified ground truth; complaints may legitimately contain several issues despite single-label modeling.
- Narratives are published/redacted text, and narrative availability is selected rather than representative of all complaints.
- Duplicate control is retrospective: later contradictory evidence can quarantine earlier records. It is not a point-in-time deployment simulation, and it cannot establish complete complaint independence.
- The temporal test covers November–December 2025 only, approximately a year after development. It does not establish performance on every future population.
- 2025 was descriptively inspected during the original audit, but not used for predictive feature selection, model selection, or tuning.
- Only one controlled fine-tuning protocol was evaluated. Its weakness does not imply that fine-tuning inherently performs worse than TF-IDF.
- Frozen MiniLM covers complete narratives via chunks; fine-tuned MiniLM truncates after 510 content tokens. Their difference combines training strategy and input-handling policy.
- Source exports are retrospectively updated snapshots. Exact acquisition timestamps are unavailable; filename numbers do not define receipt coverage.
- Public data access has changed, and the original narrative snapshots are excluded from this repository. Exact end-to-end reproduction requires those historical inputs; a current export is not a substitute.

## Repository structure

| Path | Contents |
|---|---|
| `profiling/` | Original assessment, schema/selection/taxonomy summaries, duplicate evidence |
| `cohort/` | Frozen cohort policy and source/integrity manifest; narrative-bearing CSVs are local only |
| `benchmark/` | Majority, lexical, and TF-IDF validation experiments and feature diagnostics |
| `embedding_benchmark/` | Frozen MiniLM configuration, validation results, model revision, environment lock |
| `finetune_benchmark/` | Protocol, epoch metrics, native-length checks, retention and class-flow diagnostics |
| `temporal_test_evaluation/` | Authoritative final metrics, paired bootstrap, predictions without narrative text, and figures |
| Root Python scripts | Profiling, cohort construction, development-stage runs, evaluation, and verification |
| `docs/` | Reproduction guide and publication-readiness review |

Stage reports describe what was known when that stage finished; their historical statements about an unscored test are superseded by the final evaluation above.

Raw exports, narrative-bearing derived files, trained `.joblib`/`.safetensors` weights, and cached `.npy` embeddings are intentionally excluded. Lightweight results and manifests remain public. No complaint narrative examples are reproduced in this README.

## Reproducing the experiment

The saved results can be read without downloading data or loading models. See [docs/REPRODUCING.md](docs/REPRODUCING.md) for source acquisition, historical-snapshot constraints, environment setup, workflow order, and checks. Rebuilding models is an independent reproduction exercise, not authorization to resume development against this scored holdout.

**Data-access caveat:** CFPB announced cessation of narrative publication on August 14, 2026; its September 2026 API release notes confirm removal from the database. The announcement directs previously published narratives to its FOIA Reading Room. See the [official announcement](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/) and [release notes](https://cfpb.github.io/api/ccdb/release-notes.html). An exact historical export cannot be promised from today's database. This repository makes that reproducibility gap explicit rather than silently substituting a changed dataset.

Publication cleanup and local-only review artifacts are listed in [docs/PUBLICATION_REVIEW.md](docs/PUBLICATION_REVIEW.md).
