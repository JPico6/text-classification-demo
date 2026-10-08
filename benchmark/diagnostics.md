# Feature and confusion diagnostics — review only

Inspected the selected model’s top 25 positive features for each class. Coefficients reflect November training associations, not verified leakage or causal relevance. No feature, narrative, label, cohort record, or rule was modified after inspection.

| Issue | Diagnostic finding |
| --- | --- |
| Advertising and marketing, including promotional offers | Offer/promotion/bonus are sensible topic signals. 0 %, spend { and punctuation-bearing features can also encode offer formatting. |
| Closing your account | Closed/close/closure dominate. No obvious company or explicit form-heading shortcut among the top 25; this is not a full leakage audit. |
| Fees or interest | Interest/fee/late are sensible. Currency and CFPB amount wrappers ($, { $, {, }) appear in the top 25: plausible amount context plus a publication-format shortcut. |
| Getting a credit card | Application/applied and identity/name features capture both applications and cards opened without knowledge. Credit-report terms overlap with reporting labels. |
| Incorrect information on your report | 1099 and a tax may reflect a specific complaint subtype. company wrote occurs in only five training documents, all in this class; inspect recurring phrasing/template or response quotations. |
| Other features, terms, or problems | Rewards/points dominate the broad label. pnc (10 training documents, 5 in class) is a possible brand shortcut; xxxx points mixes topic information with redaction formatting. issue is generic prose/form language, not proof of target copying. |
| Problem when making payments | Payment/payments/autopay are sensible. Bank/account/late-fee terms create overlap with card use and fee complaints. |
| Problem with a company's investigation into an existing problem | Legal-citation fragments 1666 b and usc 1666 each occur in five class documents among only 46 training examples. target may be a retailer/card brand or ordinary prose. . certified, is yet and be fixed are low-support phrasing candidates; investigate boilerplate, not automatic deletion. |
| Problem with a purchase shown on your statement | Dispute/merchant/refund are sensible. xxxx xxxx and from xxxx are redaction artifacts with positive weights; legal dispute language overlaps with investigation. |
| Struggling to pay your bill | Hardship/settlement/negotiate are sensible. declared disaster occurs in two training documents, both in class; continental occurs in three, two in class and may be a company name. These are fragile/event/template correlations in a 37-example class. |
| Trouble using your card | Limit/credit limit/blocked/declined are sensible but overlap the broad Other label. ! and complaints may encode writing style or form language rather than mechanism. |

Company-name candidates, legal citations, publication redactions, low-support phrases, and generic form words deserve review. No complete copied Issue label is apparent in the top-25 lists, but unigram/bigram inspection cannot establish absence of longer copied answers or form-language leakage. Ordinary topical overlap with the target is intended signal.

shortcut_training_contexts.csv contains up to two training-only contexts per selected suspicious feature; these are short excerpts, not full complaints or a representative sample. Brand interpretations such as target and continental require context. No test text is included.

Context inspection confirms PNC and Continental Finance occur as company names. The company wrote snippets repeat tax-write-off/1099 wording, and USC 1666b snippets repeat on-time-payment/legal-error arguments with small wording changes. These are template-family candidates left after the frozen duplicate policy, not grounds to alter that policy. Amount wrappers are CFPB publication formatting. Such associations are shortcut risks rather than proof of copied target labels.

## Largest validation confusions

| true_Issue | predicted_Issue | count | fraction_of_true_label |
| --- | --- | --- | --- |
| Other features, terms, or problems | Problem with a purchase shown on your statement | 68 | 19.4% |
| Fees or interest | Problem when making payments | 39 | 10.9% |
| Other features, terms, or problems | Closing your account | 39 | 11.1% |
| Other features, terms, or problems | Getting a credit card | 33 | 9.4% |
| Problem when making payments | Fees or interest | 33 | 15.6% |
| Other features, terms, or problems | Advertising and marketing, including promotional offers | 30 | 8.6% |
| Advertising and marketing, including promotional offers | Fees or interest | 27 | 17.9% |
| Incorrect information on your report | Getting a credit card | 27 | 32.1% |
| Other features, terms, or problems | Problem when making payments | 26 | 7.4% |
| Other features, terms, or problems | Fees or interest | 25 | 7.1% |

## Lowest validation F1 labels

| Issue | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| Problem with a company's investigation into an existing problem | 0.4000 | 0.1304 | 0.1967 | 46 |
| Struggling to pay your bill | 0.3333 | 0.2286 | 0.2712 | 35 |
| Other features, terms, or problems | 0.4747 | 0.2686 | 0.3431 | 350 |
| Incorrect information on your report | 0.3564 | 0.4286 | 0.3892 | 84 |

These weaknesses and potentially fragile associations should be reviewed before deciding whether to run additional development experiments. Existing small class supports remain unchanged. Validation model selection does not establish temporal generalization; 2025 remains unscored.
