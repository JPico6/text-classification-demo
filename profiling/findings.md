# CFPB complaint data assessment

Descriptive inspection of the two repository CSVs. No predictive models, vectorizers, fitted preprocessing, or model performance estimates were created. Counts are CSV records, not physical lines. Blank/whitespace-only fields are missing; literal redactions and placeholder values are not counted as missing. Original files are unchanged.

## Recommendation

Proceed with a historical, narrative-only credit-card Issue classification benchmark, subject to duplicate quarantine and a small manual label/leakage audit. The goal is to reproduce the consumer-selected issue for published narratives, not determine the objectively correct issue, verify allegations, or classify every credit-card complaint. Use both credit-card sub-products. Retain an explicitly scoped 11-label primary task using training support, and report the excluded rare-label coverage separately. Do not silently merge the rare labels into “Other features, terms, or problems,” which is already a distinct label.

## File coverage and schema

| File year | Rows | Received range | Sent-to-company range | Nonempty narratives |
| --- | --- | --- | --- | --- |
| 2024 | 591,058 | 2024-11-01 – 2024-12-31 | 2024-11-01 – 2026-03-09 | 154,916 (26.21%) |
| 2025 | 1,009,132 | 2025-11-01 – 2025-12-31 | 2025-11-01 – 2026-08-18 | 134,641 (13.34%) |

Combined: **1,600,190 records**. Both files cover all 61 receipt dates in November–December, starting on November 1 despite the “8” and “14” in their filenames. There is no January–October data in either year. Treat the filenames as identifiers, not evidence of an export timestamp or date filter. Receipt dates and sent dates parse without errors; none has a negative sent lag. Median and 90th-percentile sent lag are zero days; 99th percentile is 8 days, maximum 446 days. These are retrospectively updated extracts, not reconstructable point-in-time snapshots. File hashes are in summary.json.

The headers match exactly: 16 columns. Dates: `Date received`, `Date sent to company`. Taxonomy: `Product`, `Sub-product`, `Issue`, `Sub-issue`. Free text: `Consumer complaint narrative`, `Company public response`. Other categorical/context fields: `Company`, `State`, `ZIP code`, `Tags`, `Submitted via`, `Company response to consumer`, `Timely response?`. Identifier: `Complaint ID`. Read IDs and ZIP codes as strings; preserve leading zeros and redaction placeholders. There is no consumer identifier, consent field, narrative publication date, or extraction timestamp. Each file has 11 products and 54 sub-products; 2024 has 88 distinct Issue values and 2025 has 90.

## Missingness

| Field | 2024 missing | 2025 missing |
| --- | --- | --- |
| Sub-issue | 6,519 (1.10%) | 11,727 (1.16%) |
| Consumer complaint narrative | 436,142 (73.79%) | 874,491 (86.66%) |
| Company public response | 225,289 (38.12%) | 422,077 (41.83%) |
| State | 1,274 (0.22%) | 1,243 (0.12%) |
| Tags | 571,447 (96.68%) | 987,337 (97.84%) |

All other fields have zero blank values, including Product, Issue, Company, both dates, Complaint ID, and ZIP code. Zero blank ZIP codes does not establish geographic completeness: redaction/placeholder strings are retained. Tags and public responses are optional; Sub-issue can be structurally absent. Missing narratives cannot be attributed specifically to non-consent because the export lacks consent/publication status. Historical publication required opt-in consent and de-identification; availability also reflects selection and publication processes. [CFPB publication policy](https://files.consumerfinance.gov/f/documents/06022023_Publication-of-Consumer-Response-Complaint-Narratives.pdf).

## Product and issue distributions

| Product | 2024 | 2025 |
| --- | --- | --- |
| Checking or savings account | 9,973 | 13,769 |
| Credit card | 12,792 | 14,720 |
| Credit reporting or other personal consumer reports | 521,672 | 903,508 |
| Debt collection | 31,879 | 54,989 |
| Debt or credit management | 682 | 681 |
| Money transfer, virtual currency, or money service | 3,147 | 5,752 |
| Mortgage | 3,501 | 4,866 |
| Payday loan, title loan, personal loan, or advance loan | 1,609 | 2,681 |
| Prepaid card | 1,178 | 1,125 |
| Student loan | 2,239 | 2,851 |
| Vehicle loan or lease | 2,386 | 4,190 |

Credit reporting dominates: 88.3% of 2024 records and 89.5% of 2025 records. Credit card accounts for only 2.16% and 1.46%, respectively. A full-database text task would be dominated by reporting disputes and repeated templates; filtering by exact `Product == "Credit card"` provides a clearer task. Do not infer the product by searching narrative text.

| Year | Top database Issue | Rows | Share |
| --- | --- | --- | --- |
| 2024 | Incorrect information on your report | 289,126 | 48.92% |
| 2024 | Improper use of your report | 139,843 | 23.66% |
| 2024 | Problem with a company's investigation into an existing problem | 93,119 | 15.75% |
| 2024 | Attempts to collect debt not owed | 13,040 | 2.21% |
| 2024 | Written notification about debt | 7,162 | 1.21% |
| 2025 | Incorrect information on your report | 552,295 | 54.73% |
| 2025 | Improper use of your report | 178,127 | 17.65% |
| 2025 | Problem with a company's investigation into an existing problem | 170,194 | 16.87% |
| 2025 | Attempts to collect debt not owed | 19,662 | 1.95% |
| 2025 | Took or threatened to take negative or legal action | 17,672 | 1.75% |

Full distributions and Product/Issue/Sub-issue combinations are supplied in Issue_counts.csv and taxonomy_counts.csv. Labels should be interpreted within Product: identical or very similar issue strings can appear under multiple products.

## Credit-card cohort

There are **27,512 credit-card complaints**, of which **12,363 (44.94%)** have narratives. 2024: 12,792 complaints / 5,527 narratives (43.21%); 2025: 14,720 / 6,836 (46.44%). Both General-purpose credit card or charge card and Store credit card are present.

| Issue | 2024 rows | 2024 narratives / availability | 2025 rows | 2025 narratives / availability |
| --- | --- | --- | --- | --- |
| Advertising and marketing, including promotional offers | 420 | 282 (67.1%) | 582 | 322 (55.3%) |
| Closing your account | 785 | 456 (58.1%) | 1,128 | 570 (50.5%) |
| Credit monitoring or identity theft protection services | 31 | 13 (41.9%) | 23 | 8 (34.8%) |
| Fees or interest | 1,096 | 697 (63.6%) | 1,516 | 837 (55.2%) |
| Getting a credit card | 1,575 | 700 (44.4%) | 1,821 | 741 (40.7%) |
| Improper use of your report | 106 | 43 (40.6%) | 114 | 51 (44.7%) |
| Incorrect information on your report | 2,074 | 204 (9.8%) | 1,513 | 475 (31.4%) |
| Other features, terms, or problems | 1,170 | 680 (58.1%) | 1,516 | 814 (53.7%) |
| Problem when making payments | 681 | 384 (56.4%) | 743 | 335 (45.1%) |
| Problem with a company's investigation into an existing problem | 1,784 | 254 (14.2%) | 627 | 194 (30.9%) |
| Problem with a purchase shown on your statement | 2,479 | 1,473 (59.4%) | 4,275 | 2,060 (48.2%) |
| Problem with fraud alerts or security freezes | 10 | 7 (70.0%) | 17 | 7 (41.2%) |
| Struggling to pay your bill | 172 | 80 (46.5%) | 244 | 101 (41.4%) |
| Trouble using your card | 399 | 250 (62.7%) | 590 | 317 (53.7%) |
| Unable to get your credit report or credit score | 10 | 4 (40.0%) | 11 | 4 (36.4%) |

All available credit-card narratives were submitted via Web. For 2024, Web accounts for 11,968 of 12,792 complaints; for 2025, 13,661 of 14,720. Phone, postal mail, and referral credit-card complaints have no narratives. Conditioning on narrative availability therefore also conditions on channel.

Selection is strongly issue-dependent. In 2024, Incorrect information on your report has only 9.8% narrative availability, compared with 59.4% for purchase-statement problems. Across the full credit-card cohort, incorrect-report complaints fall from 16.2% to 10.3%; among narratives their share instead rises from 3.7% to 6.9%. Purchase-statement problems rise from 26.7% to 30.1% of narratives; investigation problems fall from 4.6% to 2.8%. Overall database narrative availability falls from 26.2% to 13.3%, even though credit-card availability rises. These are material composition changes, not evidence of changing underlying consumer incidence.

Narrative length has useful substance: median normalized word count is 160 in 2024 and 185.5 in 2025; 90th percentiles are 439 and 447. There are 244 narratives under 20 normalized words (1.97%). Keep short but meaningful narratives initially; a length cutoff could introduce issue-dependent selection. The longest text is 32,400 characters / 5,343 normalized words, so future truncation choices need validation. Counts include redaction tokens, not just semantic words.

Credit-card Product does not imply an issuer-only population: the 2024 full cohort includes TransUnion (1,253), Experian (1,212), and Equifax (1,205). The narrative cohort is more issuer-heavy. Capital One, Citi, and Synchrony contribute 33.3% of 2024 narratives and 36.7% of 2025 narratives. Treat company mix as a potential shortcut and drift factor; retain these complaints unless the intended business population is specifically card issuers.

## Labels across time

The 15 credit-card Issue strings are identical across the two windows. No observed credit-card Issue rename requires harmonization. Stability of names does not establish stability of meaning or consumer interpretation. Check label pairs with overlapping semantics, especially incorrect reporting versus investigation, card use versus purchase disputes, and fees versus the broad “Other” category. Two credit-card Sub-issue values occur only in 2024: Received unsolicited financial product or insurance offers after opting out, and Received unwanted marketing or advertising. There are no 2025-only credit-card Sub-issue values. Differences in observed categories alone do not prove a form change.

Across all products, the observed Issue sets differ:

- Only in 2024: .

- Only in 2025: Problem with credit report or credit score; Property was damaged or destroyed property.

The vehicle-property wording variants are candidates for taxonomy review; rare absent labels may simply have no observations. Do not automatically equate observed set changes with official renaming. CFPB retains consumers’ original taxonomy selections and documents form changes; consult the relevant taxonomy before extending to other years. [CFPB database and taxonomy documentation](https://www.consumerfinance.gov/data-research/consumer-complaints/).

## Duplicate and near-duplicate audit

There are zero duplicate Complaint IDs, zero duplicate full rows within either file, and zero Complaint ID overlap between files. This does not establish independent consumers or events.

Across all products, 289,557 rows have narratives. Exact stripped text yields 19,098 repeated groups affecting 140,814 rows (48.63%) and 121,716 excess copies. There are 253 groups spanning years, 127 spanning products, and 696 with conflicting Issue labels. This is a strong warning against random splits of the full database.

For credit-card narratives:

| Text match | Repeated groups | Affected rows | Excess copies | Cross-year groups | Conflicting Issue groups |
| --- | --- | --- | --- | --- | --- |
| exact_hash | 115 | 565 | 450 | 2 | 2 |
| norm_hash | 123 | 593 | 470 | 2 | 3 |

Exact means case-sensitive stripped text; normalized means lowercase alphanumeric tokens with collapsed whitespace. Normalized repeated groups affect 4.80% of credit-card narratives. Different IDs with the same narrative may reflect template use, multiple companies, repeat submissions, or shared events; they cannot safely be assumed independent or fraudulent.

An approximate screen used 64-bit SimHash of token 5-shingle sets, eight 8-bit candidate buckets, a length-ratio filter, and exact set Jaccard ≥0.80. It identified 309 distinct-normalized-text near-duplicate pairs involving 314 IDs; 24 pairs have different Issue labels. None of the detected distinct-text pairs spans years, but exact/normalized matches do span years. This is a lower-bound screening result: short texts with fewer than 20 unique shingles are not screened, and candidate generation can miss similar pairs. It is not an exhaustive or semantic duplicate audit. See near_duplicate_pairs.csv and hashed duplicate-group files.

Before modeling, form connected components from exact/normalized matches plus manually confirmed near-duplicates. Quarantine contradictory-label components for review. Preserve one earliest representative of non-conflicting components; exclude later copies from validation/test and report excluded counts by issue and split. Do not move future complaints into training to keep a group together. Report performance on unique narratives and, separately, operational complaint-weighted results if needed. Check reused long passages and short-template cases beyond this screen. No deduplicated modeling dataset has been constructed yet.

## Narrative leakage and label quality

Only 14 / 12,363 narratives (0.11%) contain their complete normalized Issue label verbatim; 5 contain their own full Sub-issue string. A broad field-marker screen (`issue:`, `sub-issue:`, `product:`, `complaint type:`) flags 234 narratives. These flags are not confirmed leakage: ordinary phrases such as “Issue:” and short sub-issue strings can be natural prose. Exact-string checks also miss paraphrases, partial labels, appended form answers, and differently formatted headers.

A deterministic spot-check of the first 400 characters of up to two complaints per Issue/year (60 narrative prefixes, first IDs within each cell) showed substantive stories about promotional bonuses, disputed purchases, late-payment reporting, account closures, and investigations; it also showed copied reporting-dispute language under separate IDs. Some label mismatch or ambiguity is visible: complaint 10994030 concerns a closed paid balance still appearing on a report but is labeled Unable to get your credit report or credit score; complaint 16975970 concerns multiple debt collectors but is labeled Credit monitoring or identity theft protection services. Credit-limit reductions appear under both Other features and Trouble using your card (10648203 and 10651960). These are examples requiring review, not adjudicated errors. The sample supports useful semantic content but is not a full-text, adjudicated label-quality or leakage audit. No raw narrative excerpts are redistributed in this report.

Issue is a consumer-selected form category. A narrative describing fees is valid predictive evidence for Fees or interest; ordinary topic overlap is not leakage. Explicit copied form headings/answers that reveal the selected Issue are shortcuts and should be reviewed and removed or quarantined using a documented policy. If the product is meant to recommend an Issue before the consumer selects one, this dataset does not establish that narratives were authored independently of that selection.

For the future model, allow only Consumer complaint narrative as input. Sub-issue is directly related to the target; Product/Sub-product can reveal the applicable label menu; response fields and Date sent to company are downstream; Company, IDs, dates, channel, location, and Tags can encode spurious cohort/label correlations. Use these fields only for filtering, splitting, grouping, and audit. Never concatenate the exported row or form selections into text. Company names, legal citations, prior complaint IDs, response quotations, and copied boilerplate can also occur inside narratives: audit brand masking and boilerplate/label-span removal as sensitivity analyses after approval. Ordinary references to a prior response are legitimate historical input if they existed at submission; their presence alone is not proof that a later outcome leaked.

Published/redacted narratives are not identical to live intake text. No edit history or publication timestamps are available to verify narrative availability at historical receipt time. Treat the result as a retrospective benchmark. Plan a manual stratified audit of label fit, ambiguous multi-issue stories, form echoes, and template families before interpreting model scores; keep the 2025 test text out of feature-design iteration.

## Recommended cohort and temporal evaluation

Primary cohort: exact Product == Credit card; both observed sub-products; nonempty narrative; valid receipt date and Issue; stable 11-label support rule fixed from November 2024: at least 30 training examples per label. The 30-example threshold is a pragmatic initial scope rule, not evidence of adequate statistical power for every class; the smallest retained class needs uncertainty reporting. This retains Struggling to pay your bill (43 training examples) while excluding Improper use of your report, credit-monitoring/identity-protection services, fraud-alert/security-freeze problems, and unable-to-get-report/score problems. This is a restricted 11-label problem, not a claim to classify every credit-card issue. Retain the rare labels in an audit set and report out-of-scope/abstention coverage; do not fabricate a heterogeneous “Other” label.

| Issue | test | train | validation |
| --- | --- | --- | --- |
| Advertising and marketing, including promotional offers | 322 | 131 | 151 |
| Closing your account | 570 | 218 | 238 |
| Credit monitoring or identity theft protection services | 8 | 6 | 7 |
| Fees or interest | 837 | 337 | 360 |
| Getting a credit card | 741 | 330 | 370 |
| Improper use of your report | 51 | 19 | 24 |
| Incorrect information on your report | 475 | 108 | 96 |
| Other features, terms, or problems | 814 | 323 | 357 |
| Problem when making payments | 335 | 168 | 216 |
| Problem with a company's investigation into an existing problem | 194 | 142 | 112 |
| Problem with a purchase shown on your statement | 2060 | 683 | 790 |
| Problem with fraud alerts or security freezes | 7 | 6 | 1 |
| Struggling to pay your bill | 101 | 43 | 37 |
| Trouble using your card | 317 | 116 | 134 |
| Unable to get your credit report or credit score | 4 | 0 | 4 |

Proposed split, based exclusively on Date received:

| Split | Receipt dates | All 15 labels, raw narratives | Primary 11 labels, before deduplication |
| --- | --- | --- | --- |
| train | 2024-11-01 through 2024-11-30 | 2630 | 2599 |
| validation | 2024-12-01 through 2024-12-31 | 2897 | 2861 |
| test | 2025-11-01 through 2025-12-31 | 6836 | 6766 |

The 11-label scope covers 12,226 / 12,363 narratives (98.89%) before deduplication. The 2025 two-month test is a genuine later-period holdout separated by a ten-month gap; it tests year-over-year transfer in the same season. It cannot measure all-season robustness or identify when changes occurred. Both 2025 months were inspected descriptively for this assessment, so this is not a pristine blind holdout. Freeze the scope now and exclude both November and December 2025 from subsequent model/feature design; report month-specific scores after the final evaluation. A November-2025-validation / December-2025-test design would adapt to the test year and answers a different question; it is not the primary recommendation.

Tune only on December 2024, then optionally refit approved choices on all retained November–December 2024 data for a single final 2025 evaluation. Learn tokenization/vocabulary, weights, thresholds, and data-dependent cleaning only from training; use validation for decisions. Deduplicate temporally before fitting and publish the final post-quarantine split counts. Acquisition of more historical months would strengthen both training and temporal validation, but is outside the inspected files.

Once modeling is approved, report macro-F1, per-label precision/recall/F1 and support, balanced accuracy, confusion matrix, and micro/weighted-F1 or accuracy as secondary measures. Include majority-class and simple lexical baselines, with uncertainty and company/sub-product/month slices. The largest class is 26.7% of 2024 narratives and 30.1% of 2025 narratives before restricting labels; accuracy alone would conceal weak minority performance. No baseline or model has been implemented.

## Provenance and limits

The CFPB says the database is not a representative statistical sample and preserves original consumer taxonomy selections. These findings concern published complaints, not prevalence of consumer harm or legally verified violations. [CFPB database guidance](https://www.consumerfinance.gov/data-research/consumer-complaints/).

As of this assessment date (October 8, 2026), CFPB announced on August 14, 2026 that it ceased discretionary public publication of narratives and directed previously published narratives to its FOIA Reading Room. Freeze these repository files and their hashes for reproducibility; do not promise a continuing public narrative feed. [CFPB announcement](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/).

The audit uses pandas plus standard-library hashing and similarity screening. summary.json contains file checksums, schema counts, missingness, lengths, and duplicate summaries. CSV outputs contain full distributions, taxonomy, label echoes, split support, and hashed/ID-only duplicate evidence. profile_ccdb.py and profiling_supplement.py reproduce the measurements; build_profile_report.py assembles this report. Existing CSVs are neither modified nor relabeled.
