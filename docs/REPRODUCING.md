# Reproducing the frozen experiment

## Read the saved analysis first

The root README, stage reports, result CSVs, JSON manifests, and figures are sufficient to inspect the conclusions without complaint narratives or model weights. The final authoritative outputs are in `temporal_test_evaluation/`. Viewing them is different from regenerating the experiment.

No models were trained, selected, or scored during the portfolio-documentation work. Reproduction commands below are instructions for an independent checkout with the original inputs, not steps executed here and not a new development cycle.

## Source acquisition and historical snapshots

CFPB's [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) provides structured complaint data and links to download/API facilities. However, the [August 14, 2026 announcement](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/) and [September 2026 release notes](https://cfpb.github.io/api/ccdb/release-notes.html) document removal of narratives from the current database. The announcement points to the FOIA Reading Room for previously published narratives. Availability of the exact two historical narrative snapshots has not been verified.

For exact reproduction, obtain the original historical narrative-bearing exports through an appropriate archive/data custodian and verify their SHA-256 values against `profiling/summary.json` and `cohort/manifest.json`. A current structured-data download, a reconstructed archive, or identical date filters do not guarantee the same rows, redactions, labels, or bytes. Do not disable the cohort hash checks to make a changed snapshot pass.

Place the original exports in the repository root using these identifiers:

- `CCDB_Export_8_November_2024_through_December_2024.csv`
- `CCDB_Export_14_November_2025_through_December_2025.csv`

Both cover November 1–December 31 in their respective years despite the numeric filename prefixes. The inputs have 16 fields, including `Consumer complaint narrative`, `Date received`, `Product`, `Sub-product`, `Issue`, and `Complaint ID`. Preserve original strings and read IDs as strings. Do not add these exports or derived narratives to Git.

## Environments

The recorded runs used Python 3.12.14 on Windows and three separate environments. Exact package inventories are in:

- `benchmark/requirements-lock.txt`: pandas/scikit-learn TF-IDF baseline environment.
- `embedding_benchmark/requirements-lock.txt`: PyTorch 2.9.0+cpu, SentenceTransformers 5.1.2, Transformers 4.57.1.
- `finetune_benchmark/requirements-lock.txt`: PyTorch 2.5.1+cu121 and Transformers 4.57.1; RTX 3070 Laptop GPU, CUDA runtime 12.1.

Example PowerShell setup, in an independent checkout:

```powershell
py -3.12 -m venv .venv-benchmark
py -3.12 -m venv .venv-embedding
py -3.12 -m venv .venv-finetune
.\.venv-benchmark\Scripts\python.exe -m pip install -r benchmark/requirements-lock.txt
.\.venv-embedding\Scripts\python.exe -m pip install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv-embedding\Scripts\python.exe -m pip install -r embedding_benchmark/requirements-lock.txt
.\.venv-finetune\Scripts\python.exe -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
.\.venv-finetune\Scripts\python.exe -m pip install -r finetune_benchmark/requirements-lock.txt
```

These are recorded environments, not a portable package specification or a claim that every pinned wheel remains available. CUDA requires a compatible driver/GPU. Different operating systems, CPU libraries, drivers, or hardware can change runtime and floating-point results. Reproduction installation has not been tested in a clean public checkout during this documentation task.

## Workflow and script order

Execute stages only in a separate reproduction workspace; scripts write outputs at repository-relative locations and several historical manifests hash earlier artifacts. Do not run the rebuilding stages over the published frozen results.

| Stage | Scripts | Dependencies / notes |
|---|---|---|
| Original audit | `profile_ccdb.py`, `profiling_supplement.py`, `build_profile_report.py` | Original exports; profiling emits narrative review snippets to the console, so keep logs local |
| Frozen cohort | `build_credit_card_cohort.py`, `test_credit_card_cohort.py` | Original exports and `profiling/near_duplicate_pairs.csv`; preserve the assessed near-pair evidence rather than inventing a new duplicate search |
| Lexical and TF-IDF development | `run_validation_benchmark.py`, `verify_validation_benchmark.py` | November/December cohort files; imports `benchmark_text.py` |
| Feature review | `inspect_validation_diagnostics.py` | Saved selected pipeline and training narratives; writes local-only context excerpts |
| Pretrained download | `fetch_embedding_model.py` | Network access; recorded revision and download metadata |
| Frozen embeddings | `run_embedding_benchmark.py`, `verify_embedding_benchmark.py`, `summarize_embedding_comparison.py` | Imports `frozen_embedding.py`; November/December only; original CPU environment |
| Native length and fine-tuning | `verify_native_length.py`, `run_finetune_benchmark.py`, `verify_finetune_benchmark.py`, `report_finetune_benchmark.py` | Imports `finetune_model.py`; fixed protocol and original CUDA environment |
| Epoch diagnostics | `diagnose_epoch_class_flows.py` | Saved validation predictions only; no additional training |
| Final frozen evaluation | `run_final_temporal_test.py`, `report_final_temporal_test.py`, `verify_final_temporal_test.py` | Actual selected pipelines/checkpoint and 2025 cohort; orchestrates the three recorded environments |

The fine-tuning protocol and bootstrap procedure are predefined in the corresponding configuration files. The final bootstrap uses 10,000 paired IID complaint resamples, seed 20261009. Do not refit on November plus December or select a different epoch/configuration before comparing a reproduction with these final results.

The final inference runner deliberately refuses an already-completed evaluation directory. A public checkout includes that directory as evidence; use an isolated reproduction copy/output workspace and preserve the published results before regenerating anything. This guard prevents accidental predictive re-access; the scripts are not a one-command pipeline for rerunning the scored holdout in place.

## Known reproducibility gaps

1. **Historical data availability:** the raw snapshots are excluded, current database narratives have been removed, and matching historical byte-for-byte inputs have not been located for public download.
2. **Machine-specific paths:** `embedding_benchmark/model_download.json` records an absolute local model path. Running `fetch_embedding_model.py` in the reproduction workspace rebuilds local metadata for the pinned revision; do not edit the published historical manifest in place.
3. **Hardware and package distribution:** lock files record exact inventories, including platform-specific CPU/CUDA PyTorch builds. They are not validated cross-platform installers.
4. **Artifact lifecycle:** development scripts overwrite outputs, while the final runner rejects completed output directories. Existing evidence needs a separate workspace; there is no configurable end-to-end output-root option.
5. **Protected local evidence:** historical manifests include hashes of excluded narrative/model files. Reading results is possible without them; full verification requires the original local artifacts. Changing a historical report or removing a hashed review artifact locally can invalidate those checks even though aggregate results have not changed.
6. **Model artifacts excluded:** the selected regression pipelines, trained fine-tuned checkpoint, and embeddings must be regenerated from the original data in an independent reproduction. Their recorded hashes preserve identity; this repository does not supply runnable trained weights.

## Checks that do not continue modeling

Synthetic tests (`test_final_evaluation_metrics.py`, `test_finetune_protocol.py`, `test_frozen_embedding.py`) exercise metric, loss-accumulation, and chunk-aggregation behavior. Stage verification scripts differ: some load models and reproduce development predictions and require excluded artifacts. `verify_final_temporal_test.py` checks saved final predictions/metrics/bootstrap and protected hashes without predictive inference, but still needs protected local files.

Documentation and publication checks are provided by `check_portfolio_docs.py`. It checks root/documentation links, key reported results, ignore rules, and whether frozen final outputs changed during the documentation work. It never loads models or accesses complaint narrative rows.
