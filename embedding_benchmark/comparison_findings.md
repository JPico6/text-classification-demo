# Embedding comparison findings

Selected frozen MiniLM uses C=4 and balanced training-derived class weights. Validation macro-F1 is 0.4983 versus 0.5111 for the unchanged selected TF-IDF model (difference −0.0129). TF-IDF remains stronger on the primary metric. Embedding weighted-F1 is 0.6101, accuracy 0.6037, and balanced accuracy 0.5337. Balanced accuracy improves by 0.0220, reflecting a different recall/precision tradeoff rather than an across-the-board gain.

Three of eleven labels improve in F1: hardship (+0.0717), account closure (+0.0094), and investigation (+0.0088). The largest losses are incorrect-report information (−0.0543), marketing (−0.0533), and payments (−0.0501). These are descriptive December differences after model selection, not established semantic or causal effects.

For hardship, recall rises from 0.2286 to 0.5143 while precision falls from 0.3333 to 0.2571 (35 validation examples). For investigation, recall rises from 0.1304 to 0.3261 while precision falls from 0.4000 to 0.1500 (46 examples); F1 therefore gains only slightly. The broad Other label stays weak for both models. No cohort, cleaning, or label decisions were changed.

## Long narratives and runtime

| Split | Narratives | Chunked | Chunked share | Chunks | Encoding seconds |
| --- | --- | --- | --- | --- | --- |
| train | 2424 | 958 | 39.52% | 3949 | 90.27 |
| validation | 2695 | 1075 | 39.89% | 4376 | 109.77 |

Each narrative becomes one 384-dimensional vector. The fixed policy uses all original wordpieces in contiguous chunks of at most 254 content tokens plus two special tokens, a content-length-weighted mean of normalized chunk vectors, and final L2 normalization. No silent truncation occurred. The longest training narrative uses 31 chunks; the longest validation narrative uses 16. Coverage is complete at the pretrained tokenizer level, with ordinary pretrained normalization/unknown-token behavior still applying.

Training encoding took 90.27s and validation encoding took 109.77s. Selected classifier fitting took 0.170s. Encoder load time was 0.19s; benchmark runtime was 203.82s. Download/setup and separate verification are excluded. Lower classifier cost comes with an encoder inference cost that the lexical benchmark does not incur.

## Per-label metrics for both selected models

### TF-IDF

| Issue | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Advertising and marketing, including promotional offers | 0.6090 | 0.5364 | 0.5704 | 151 |
| Closing your account | 0.5983 | 0.6164 | 0.6072 | 232 |
| Fees or interest | 0.6425 | 0.7159 | 0.6772 | 359 |
| Getting a credit card | 0.6292 | 0.7935 | 0.7019 | 310 |
| Incorrect information on your report | 0.3564 | 0.4286 | 0.3892 | 84 |
| Other features, terms, or problems | 0.4747 | 0.2686 | 0.3431 | 350 |
| Problem when making payments | 0.4858 | 0.5660 | 0.5229 | 212 |
| Problem with a company's investigation into an existing problem | 0.4000 | 0.1304 | 0.1967 | 46 |
| Problem with a purchase shown on your statement | 0.8155 | 0.8593 | 0.8369 | 782 |
| Struggling to pay your bill | 0.3333 | 0.2286 | 0.2712 | 35 |
| Trouble using your card | 0.5285 | 0.4851 | 0.5058 | 134 |

### Frozen embedding

| Issue | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Advertising and marketing, including promotional offers | 0.4882 | 0.5497 | 0.5171 | 151 |
| Closing your account | 0.5968 | 0.6379 | 0.6167 | 232 |
| Fees or interest | 0.7073 | 0.6462 | 0.6754 | 359 |
| Getting a credit card | 0.6561 | 0.6645 | 0.6603 | 310 |
| Incorrect information on your report | 0.2701 | 0.4405 | 0.3348 | 84 |
| Other features, terms, or problems | 0.4217 | 0.2771 | 0.3345 | 350 |
| Problem when making payments | 0.4561 | 0.4906 | 0.4727 | 212 |
| Problem with a company's investigation into an existing problem | 0.1500 | 0.3261 | 0.2055 | 46 |
| Problem with a purchase shown on your statement | 0.8589 | 0.7864 | 0.8211 | 782 |
| Struggling to pay your bill | 0.2571 | 0.5143 | 0.3429 | 35 |
| Trouble using your card | 0.4675 | 0.5373 | 0.5000 | 134 |

All eleven F1 differences and a bar chart are in README.md, per_label_comparison.csv, and per_label_f1_differences.png. Full configurations, revision, hashes, cached embeddings, timings, classifier artifacts, predictions, and confusion matrices are saved in this directory.

Three chunk-policy tests passed. Saved metrics/confusions were independently recomputed for all four classifiers, native SentenceTransformer output matches the custom path on a short synthetic input, pretrained state hashes match, and protected cohort/TF-IDF hashes are unchanged. No transformer fine-tuning and no 2025 test access occurred.
