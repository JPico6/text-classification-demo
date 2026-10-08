"""Review-only interpretation of saved validation results and training features."""
from pathlib import Path
import json, re, hashlib, time
import pandas as pd
from benchmark_text import TOKEN_RE

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'benchmark'
NOTES={
 'Advertising and marketing, including promotional offers':
    'Offer/promotion/bonus are sensible topic signals. 0 %, spend { and punctuation-bearing features can also encode offer formatting.',
 'Closing your account':
    'Closed/close/closure dominate. No obvious company or explicit form-heading shortcut among the top 25; this is not a full leakage audit.',
 'Fees or interest':
    'Interest/fee/late are sensible. Currency and CFPB amount wrappers ($, { $, {, }) appear in the top 25: plausible amount context plus a publication-format shortcut.',
 'Getting a credit card':
    'Application/applied and identity/name features capture both applications and cards opened without knowledge. Credit-report terms overlap with reporting labels.',
 'Incorrect information on your report':
    '1099 and a tax may reflect a specific complaint subtype. company wrote occurs in only five training documents, all in this class; inspect recurring phrasing/template or response quotations.',
 'Other features, terms, or problems':
    'Rewards/points dominate the broad label. pnc (10 training documents, 5 in class) is a possible brand shortcut; xxxx points mixes topic information with redaction formatting. issue is generic prose/form language, not proof of target copying.',
 'Problem when making payments':
    'Payment/payments/autopay are sensible. Bank/account/late-fee terms create overlap with card use and fee complaints.',
 "Problem with a company's investigation into an existing problem":
    'Legal-citation fragments 1666 b and usc 1666 each occur in five class documents among only 46 training examples. target may be a retailer/card brand or ordinary prose. . certified, is yet and be fixed are low-support phrasing candidates; investigate boilerplate, not automatic deletion.',
 'Problem with a purchase shown on your statement':
    'Dispute/merchant/refund are sensible. xxxx xxxx and from xxxx are redaction artifacts with positive weights; legal dispute language overlaps with investigation.',
 'Struggling to pay your bill':
    'Hardship/settlement/negotiate are sensible. declared disaster occurs in two training documents, both in class; continental occurs in three, two in class and may be a company name. These are fragile/event/template correlations in a 37-example class.',
 'Trouble using your card':
    'Limit/credit limit/blocked/declined are sensible but overlap the broad Other label. ! and complaints may encode writing style or form language rather than mechanism.',
}
SUSPECT_FEATURES=['company wrote','1099','1666 b','usc 1666','pnc','target','continental',
                  'xxxx xxxx','xxxx points','{ $','declared disaster','. certified','issue']

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in df.values])

def main():
    start=time.perf_counter()
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    labels=json.loads((OUT/'experiment_config.json').read_text())['labels']
    name=manifest['selection']['experiment']
    features=pd.read_csv(OUT/'selected_positive_features.csv',keep_default_na=False)
    # Training-only short contexts; no test or validation narrative reading here.
    train=pd.read_csv(ROOT/'cohort/train.csv',dtype=str,keep_default_na=False)
    contexts=[]
    for feature in SUSPECT_FEATURES:
        if feature not in set(features.feature): continue
        positive_classes=set(features[features.feature==feature].Issue)
        ordered=pd.concat([train[train.Issue.isin(positive_classes)],train[~train.Issue.isin(positive_classes)]])
        wanted=feature.split(' '); count=0
        for _,r in ordered.iterrows():
            raw=r['Consumer complaint narrative']; matches=list(TOKEN_RE.finditer(raw.lower()))
            toks=[m.group() for m in matches]
            for j in range(len(toks)-len(wanted)+1):
                if toks[j:j+len(wanted)]==wanted:
                    lo=max(0,matches[j].start()-160); hi=min(len(raw),matches[j+len(wanted)-1].end()+160)
                    contexts.append({'feature':feature,'positive_for_Issues':' | '.join(sorted(positive_classes)),
                        'Complaint ID':r['Complaint ID'],'Issue':r.Issue,
                        'training_context':raw[lo:hi],'context_is_excerpt':True,'reviewer_notes':''})
                    count+=1; break
            if count>=2: break
    pd.DataFrame(contexts).to_csv(OUT/'shortcut_training_contexts.csv',index=False)
    cm=pd.read_csv(OUT/f'{name}_confusion.csv',index_col=0)
    pairs=[]
    for label in cm.index:
        for predicted in cm.columns:
            if label!=predicted and cm.at[label,predicted]>0:
                pairs.append({'true_Issue':label,'predicted_Issue':predicted,'count':int(cm.at[label,predicted]),
                              'fraction_of_true_label':float(cm.at[label,predicted]/cm.loc[label].sum())})
    errors=pd.DataFrame(pairs).sort_values(['count','true_Issue','predicted_Issue'],ascending=[False,True,True])
    errors.to_csv(OUT/'validation_confusion_pairs.csv',index=False)
    detail=pd.read_csv(OUT/f'{name}_per_label.csv')
    report=['# Feature and confusion diagnostics — review only',
        'Inspected the selected model’s top 25 positive features for each class. Coefficients reflect November training associations, not verified leakage or causal relevance. No feature, narrative, label, cohort record, or rule was modified after inspection.',
        table(pd.DataFrame([{'Issue':label,'Diagnostic finding':NOTES[label]} for label in labels])),
        'Company-name candidates, legal citations, publication redactions, low-support phrases, and generic form words deserve review. No complete copied Issue label is apparent in the top-25 lists, but unigram/bigram inspection cannot establish absence of longer copied answers or form-language leakage. Ordinary topical overlap with the target is intended signal.',
        'shortcut_training_contexts.csv contains up to two training-only contexts per selected suspicious feature; these are short excerpts, not full complaints or a representative sample. Brand interpretations such as target and continental require context. No test text is included.',
        'Context inspection confirms PNC and Continental Finance occur as company names. The company wrote snippets repeat tax-write-off/1099 wording, and USC 1666b snippets repeat on-time-payment/legal-error arguments with small wording changes. These are template-family candidates left after the frozen duplicate policy, not grounds to alter that policy. Amount wrappers are CFPB publication formatting. Such associations are shortcut risks rather than proof of copied target labels.',
        '## Largest validation confusions',table(errors.head(10).assign(fraction_of_true_label=lambda x:x.fraction_of_true_label.map(lambda v:f'{v:.1%}'))),
        '## Lowest validation F1 labels',table(detail.sort_values('f1').head(4).assign(
            precision=lambda x:x.precision.map(lambda v:f'{v:.4f}'),recall=lambda x:x.recall.map(lambda v:f'{v:.4f}'),
            f1=lambda x:x.f1.map(lambda v:f'{v:.4f}'))),
        'These weaknesses and potentially fragile associations should be reviewed before deciding whether to run additional development experiments. Existing small class supports remain unchanged. Validation model selection does not establish temporal generalization; 2025 remains unscored.']
    (OUT/'diagnostics.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    readme=OUT/'README.md'; contents=readme.read_text(encoding='utf-8')
    note='\n\n## Reviewed diagnostic findings\n\nSee [diagnostics.md](diagnostics.md) for class-by-class shortcut/leakage observations and validation confusions, and shortcut_training_contexts.csv for short training-only contexts. Reproduce this review artifact with inspect_validation_diagnostics.py; it never fits or changes a model.\n'
    if '## Reviewed diagnostic findings' not in contents: readme.write_text(contents+note,encoding='utf-8')
    inputs=[Path(__file__),ROOT/'benchmark_text.py',ROOT/'cohort/train.csv',
            OUT/'selected_positive_features.csv',OUT/'run_manifest.json']
    (OUT/'diagnostic_manifest.json').write_text(json.dumps({
        'source_and_input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'model_changed':False,'test_accessed':False,'elapsed_seconds':time.perf_counter()-start},indent=2),encoding='utf-8')
    print(errors.head(10).to_string(index=False)); print(detail.to_string(index=False))

if __name__=='__main__': main()
