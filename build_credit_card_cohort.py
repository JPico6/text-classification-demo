"""Freeze the assessed cohort and duplicate decisions; no predictive modeling.

Run with pandas. Narratives and labels are exported unchanged from the CSVs.
Normalization is used only as duplicate evidence, never as model input cleaning.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import html
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'cohort'
SPLITS = ['train', 'validation', 'test']
LABELS = [
    'Advertising and marketing, including promotional offers',
    'Closing your account', 'Fees or interest', 'Getting a credit card',
    'Incorrect information on your report', 'Other features, terms, or problems',
    'Problem when making payments',
    "Problem with a company's investigation into an existing problem",
    'Problem with a purchase shown on your statement',
    'Struggling to pay your bill', 'Trouble using your card',
]

def normalize(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def assign_split(date):
    if '2024-11-01' <= date <= '2024-11-30': return 'train'
    if '2024-12-01' <= date <= '2024-12-31': return 'validation'
    if '2025-11-01' <= date <= '2025-12-31': return 'test'
    raise ValueError(f'Unexpected receipt date: {date}')

class Components:
    def __init__(self, ids): self.parent = {i:i for i in ids}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)
    def groups(self):
        groups = defaultdict(list)
        for i in self.parent: groups[self.find(i)].append(i)
        return list(groups.values())

def decisions(frame, graph):
    result = {}
    for members in graph.groups():
        ordered = sorted(members, key=lambda i:(frame.at[i,'Date received'], int(i)))
        representative = ordered[0]
        conflict = frame.loc[members,'Issue'].nunique() > 1
        group_id = digest('|'.join(sorted(members)))[:20]
        for i in members:
            result[i] = {'group_id':group_id, 'group_size':len(members),
                'representative_id':representative,
                'duplicate_status': 'quarantined_conflict' if conflict else
                    ('kept' if i == representative else 'removed_duplicate')}
    return pd.DataFrame.from_dict(result, orient='index')

def main():
    OUT.mkdir(exist_ok=True)
    profile = json.loads((ROOT/'profiling/summary.json').read_text())
    hashes = {}
    parts = []
    source_counts = []
    for name in sorted(k for k in profile if k.endswith('.csv')):
        path = ROOT/name
        hashes[name] = digest_bytes(path)
        if hashes[name] != profile[name]['sha256']:
            raise ValueError(f'Source changed since assessment: {name}')
        total = card = present = 0
        for c in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=100000):
            total += len(c)
            c = c[c.Product == 'Credit card'].copy()
            card += len(c)
            c = c[c['Consumer complaint narrative'].str.strip() != ''].copy()
            present += len(c)
            c['source_file'] = name
            parts.append(c)
        source_counts.append({'source_file':name, 'total_rows':total,
                             'credit_card_rows':card,'credit_card_narratives':present})
    n = pd.concat(parts, ignore_index=True).set_index('Complaint ID', drop=False)
    assert n.index.is_unique and len(n) == 12363
    assert pd.to_datetime(n['Date received'],format='%Y-%m-%d',errors='coerce').notna().all()
    assert n.Issue.str.strip().ne('').all()
    n['split'] = n['Date received'].map(assign_split)
    n['in_scope'] = n.Issue.isin(LABELS)
    support = n[n.split=='train'].Issue.value_counts()
    assert set(support[support>=30].index) == set(LABELS)
    raw_counts = n[n.in_scope].groupby('split').size().to_dict()
    assert raw_counts == {'train':2599,'validation':2861,'test':6766}
    n['exact_hash'] = n['Consumer complaint narrative'].map(lambda t:digest(t.strip()))
    n['normalized_hash'] = n['Consumer complaint narrative'].map(lambda t:digest(normalize(t)))
    graph = Components(n.index)
    edges = []
    snapshots = {}
    for stage, column in [('exact','exact_hash'),('normalized','normalized_hash')]:
        for _, g in n.groupby(column,sort=True):
            ids = sorted(g.index)
            for other in ids[1:]:
                edges.append({'id_a':ids[0],'id_b':other,'evidence':stage,'jaccard':''})
                graph.union(ids[0],other)
        snapshots[stage] = decisions(n,graph)
    evidence_path = ROOT/'profiling/near_duplicate_pairs.csv'
    pairs = pd.read_csv(evidence_path, dtype=str, keep_default_na=False)
    assert len(pairs)==309
    for r in pairs.sort_values(['id_a','id_b']).to_dict('records'):
        a,b = r['id_a'],r['id_b']
        assert a in n.index and b in n.index
        assert n.at[a,'Issue']==r['issue_a'] and n.at[b,'Issue']==r['issue_b']
        def shingles(i):
            tokens=normalize(n.at[i,'Consumer complaint narrative']).split()
            return set(' '.join(tokens[j:j+5]) for j in range(len(tokens)-4))
        sa,sb=shingles(a),shingles(b)
        similarity=len(sa&sb)/len(sa|sb)
        assert similarity>=.8 and abs(similarity-float(r['jaccard']))<1e-12
        edges.append({'id_a':a,'id_b':b,'evidence':'near','jaccard':similarity})
        graph.union(a,b)
    snapshots['near'] = decisions(n,graph)
    final = snapshots['near']
    ledger = n[['Complaint ID','source_file','Date received','split','Issue','in_scope',
                'exact_hash','normalized_hash']].join(final)
    ledger['decision_basis'] = ''
    for i,r in final.iterrows():
        if r.duplicate_status == 'kept': continue
        if r.duplicate_status == 'quarantined_conflict':
            stage = next(s for s in ['exact','normalized','near']
                         if snapshots[s].at[i,'duplicate_status']=='quarantined_conflict')
        else:
            representative = r.representative_id
            stage = next(s for s in ['exact','normalized','near']
                         if snapshots[s].at[i,'group_id']==snapshots[s].at[representative,'group_id'])
        ledger.at[i,'decision_basis'] = stage
    ledger['disposition'] = ledger.duplicate_status
    ledger.loc[~ledger.in_scope,'disposition'] = 'out_of_scope_rare_label'
    # Duplicate status remains visible for rare-label audit records; no relabeling.
    ordered_ids = sorted(n.index,key=lambda i:(n.at[i,'Date received'],int(i)))
    ledger = ledger.loc[ordered_ids]
    metadata_cols=['Complaint ID','Date received','split','Issue','source_file']
    for split in SPLITS:
        ids=ledger[(ledger.split==split)&(ledger.disposition=='kept')].index
        n.loc[ids,['Complaint ID','Consumer complaint narrative','Issue']].to_csv(OUT/f'{split}.csv',index=False)
        ledger.loc[ids,metadata_cols+['group_id','representative_id']].to_csv(OUT/f'{split}_metadata.csv',index=False)
    ledger.to_csv(OUT/'disposition_ledger.csv',index=False)
    component_rows=[]
    for group,g in ledger.groupby('group_id',sort=True):
        if len(g)<2: continue
        component_rows.append({'group_id':group,'rows_all_labels':len(g),
            'rows_in_scope':int(g.in_scope.sum()),'Issue_count':g.Issue.nunique(),
            'Issues':' | '.join(sorted(g.Issue.unique())),
            'splits':' | '.join(s for s in SPLITS if s in set(g.split)),
            'first_date':g['Date received'].min(),'last_date':g['Date received'].max(),
            'duplicate_status':'quarantined_conflict' if g.Issue.nunique()>1 else 'earliest_representative_retained',
            'earliest_id':g.representative_id.iloc[0]})
    component_summary=pd.DataFrame(component_rows)
    component_summary.to_csv(OUT/'duplicate_components.csv',index=False)
    audit_cols=metadata_cols+['Sub-product','Sub-issue','Consumer complaint narrative']
    n.loc[ledger[ledger.duplicate_status=='quarantined_conflict'].index,audit_cols].join(
        ledger[['group_id','group_size','in_scope','decision_basis']]).to_csv(OUT/'contradictory_label_quarantine.csv',index=False)
    n.loc[ledger[~ledger.in_scope].index,audit_cols].join(
        ledger[['group_id','duplicate_status','decision_basis']]).to_csv(OUT/'rare_label_audit.csv',index=False)
    pd.DataFrame(edges).to_csv(OUT/'duplicate_edges.csv',index=False)
    stage_rows=[]
    for stage,status in snapshots.items():
        for split in SPLITS:
            ids=n[(n.split==split)&n.in_scope].index
            cnt=status.loc[ids,'duplicate_status'].value_counts()
            stage_rows.append({'stage':stage,'split':split,'eligible':len(ids),
                **{s:int(cnt.get(s,0)) for s in ['kept','removed_duplicate','quarantined_conflict']}})
    pd.DataFrame(stage_rows).to_csv(OUT/'stage_counts.csv',index=False)
    counts=[]
    for split in SPLITS:
        for issue in LABELS:
            g=ledger[(ledger.split==split)&(ledger.Issue==issue)]
            counts.append({'split':split,'Issue':issue,'before_duplicates':len(g),
                'kept':int((g.disposition=='kept').sum()),
                **{f'{status}_{stage}':int(((g.disposition==status)&(g.decision_basis==stage)).sum())
                   for status in ['removed_duplicate','quarantined_conflict']
                   for stage in ['exact','normalized','near']}})
    counts=pd.DataFrame(counts)
    counts['removed_duplicate']=counts[[c for c in counts if c.startswith('removed_duplicate_')]].sum(axis=1)
    counts['quarantined_conflict']=counts[[c for c in counts if c.startswith('quarantined_conflict_')]].sum(axis=1)
    counts.to_csv(OUT/'counts_by_issue.csv',index=False)
    overall=counts.drop(columns='Issue').groupby('split').sum().reindex(SPLITS)
    overall.to_csv(OUT/'counts_overall.csv')
    checks={}
    checks['counts_reconcile']=bool((counts.before_duplicates==counts.kept+counts.removed_duplicate+counts.quarantined_conflict).all())
    kept=ledger[ledger.disposition=='kept']
    checks['one_representative_per_component']=not kept.group_id.duplicated().any()
    checks['no_normalized_duplicates_in_retained_splits']=not kept.normalized_hash.duplicated().any()
    checks['no_evidence_edge_has_two_retained_endpoints']=all(
        not (e['id_a'] in kept.index and e['id_b'] in kept.index) for e in edges)
    checks['receipt_split_unchanged']=all(assign_split(r['Date received'])==r['split'] for r in ledger.to_dict('records'))
    checks['conflicting_groups_fully_quarantined']=all(
        (g.duplicate_status=='quarantined_conflict').all()
        for _,g in ledger.groupby('group_id') if g.Issue.nunique()>1)
    checks['representatives_are_earliest']=all(
        (n.at[r.representative_id,'Date received'],int(r.representative_id)) <= (r['Date received'],int(i))
        for i,r in ledger.iterrows())
    checks['raw_narratives_and_labels_preserved']=True
    for split in SPLITS:
        exported=pd.read_csv(OUT/f'{split}.csv',dtype=str,keep_default_na=False).set_index('Complaint ID')
        checks['raw_narratives_and_labels_preserved'] &= exported[['Consumer complaint narrative','Issue']].equals(
            n.loc[exported.index,['Consumer complaint narrative','Issue']])
        assert set(exported.Issue)==set(LABELS)
    assert all(checks.values()),checks
    review_count=make_review(n,ledger)
    manifest={'source_sha256':hashes,'near_evidence_sha256':digest_bytes(evidence_path),
        'labels':LABELS,'source_counts':source_counts,'eligible_all_15':len(n),
        'eligible_11':int(n.in_scope.sum()),'rare_label_rows':int((~n.in_scope).sum()),
        'graph_scope':'All 15 credit-card narrative labels; rare-label conflicts can quarantine in-scope members.',
        'near_policy':'All 309 already-identified pairs used as evidence as requested; similarity recomputed, not manually adjudicated.',
        'tie_break':'Earliest Date received, then smallest numeric Complaint ID on the same day.',
        'normalization':'lowercase [a-z0-9]+ tokens, joined by spaces, for hashes only; exact strips boundary whitespace for hashes only',
        'conflict_policy':'Entire connected component quarantined if it has more than one Issue; no representative retained.',
        'review_records':review_count,'checks':checks}
    manifest['duplicate_component_counts']={
        'repeated_components_all_labels':len(component_summary),
        'contradictory_components_all_labels':int((component_summary.Issue_count>1).sum()),
        'contradictory_rows_all_labels':int((ledger.duplicate_status=='quarantined_conflict').sum()),
        'contradictory_rows_in_scope':int(((ledger.duplicate_status=='quarantined_conflict')&ledger.in_scope).sum()),
        'duplicate_copies_removed_in_scope':int((ledger.disposition=='removed_duplicate').sum())}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    make_readme(overall,counts,manifest)
    print(overall[['before_duplicates','kept','removed_duplicate','quarantined_conflict']].to_string())
    print('Review records:',review_count,'Checks:',checks)

def digest_bytes(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def make_review(n,ledger):
    reasons=defaultdict(list)
    def add(i,reason):
        if reason not in reasons[i]: reasons[i].append(reason)
    dev=ledger[ledger.split.isin(['train','validation'])]
    # One retained example per Issue, closest to that Issue's median text length.
    for issue in LABELS:
        candidates=dev[(dev.Issue==issue)&(dev.disposition=='kept')].index
        lengths=n.loc[candidates,'Consumer complaint narrative'].str.len()
        target=lengths.median()
        i=min(candidates,key=lambda i:(abs(lengths[i]-target),n.at[i,'Date received'],int(i)))
        add(i,'Representative development example: near median narrative length; not a certified correct label')
    # All literal Issue echoes in development; up to two field-marker cases per Issue.
    marker=re.compile(r'\b(issue|sub[ -]?issue|product|complaint type)\s*:',re.I)
    flagged=[]
    for i,r in n.loc[dev.index].iterrows():
        t=r['Consumer complaint narrative']; normalized=normalize(t)
        if normalize(r.Issue) in normalized: add(i,'Potential leakage: complete normalized Issue phrase occurs in narrative')
        if r['Sub-issue'] and normalize(r['Sub-issue']) in normalized:
            add(i,'Potential leakage: complete normalized Sub-issue phrase occurs in narrative')
        if marker.search(t): flagged.append(i)
    for issue in LABELS:
        ids=[i for i in flagged if n.at[i,'Issue']==issue]
        for i in sorted(ids,key=lambda i:(n.at[i,'Date received'],int(i)))[:2]:
            add(i,'Potential leakage: broad form-field marker; ordinary prose can also trigger this flag')
    for i in ['10994030','16975970','10648203','10651960']:
        add(i,'Previously surfaced ambiguous label example; assess entire narrative without automatic relabeling')
    # Show one member per distinct label in each contradictory component.
    conflict=ledger[ledger.duplicate_status=='quarantined_conflict']
    for group,g in conflict.groupby('group_id'):
        for issue,gg in g.groupby('Issue'):
            i=gg.index[0]
            add(i,f'Contradictory duplicate component {group}: one example for this label')
    records=[]
    for i in sorted(reasons,key=lambda i:(n.at[i,'Date received'],int(i))):
        r=n.loc[i]; l=ledger.loc[i]
        records.append({'Complaint ID':i,'Date received':r['Date received'],'split':r.split,
            'Issue':r.Issue,'Sub-issue':r['Sub-issue'],'disposition':l.disposition,
            'duplicate_status':l.duplicate_status,'group_id':l.group_id if l.group_size>1 else '',
            'review_reason':' | '.join(reasons[i]),'Consumer complaint narrative':r['Consumer complaint narrative'],
            'reviewer_leakage':'','reviewer_label_fit':'','reviewer_duplicate_confirmation':'','reviewer_notes':''})
    review=pd.DataFrame(records)
    review.to_csv(OUT/'manual_review.csv',index=False)
    sections=['<!doctype html><meta charset="utf-8"><title>Credit-card manual review</title>',
        '<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 16px}details{border:1px solid #ccc;padding:12px;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px system-ui}small{color:#555}</style>',
        '<h1>Credit-card manual review</h1>',
        f'<p>{len(records)} deduplicated review records. Expand to read the complete original narrative. Flags are review prompts, not findings of leakage or erroneous labels. No text was removed or label changed.</p>',
        '<p>Representative and new leakage examples use training/validation only. Previously surfaced ambiguity examples and contradictory components can include test records; review reasons identify them. This is a purposive sample, not an estimate of error prevalence.</p>',
        '<p>For each record, assess: does copied form information reveal the selected label? Does the narrative support the assigned Issue or several Issues? For duplicate components, are the accounts genuinely duplicates? Record decisions in the blank reviewer columns of manual_review.csv. Decisions are not applied automatically.</p>']
    for r in records:
        e=lambda k:html.escape(str(r[k]))
        sections += [f'<details><summary>{e("Complaint ID")} · {e("split")} · {e("Issue")}</summary>',
            f'<p><strong>Review reason:</strong> {e("review_reason")}</p>',
            f'<small>Received: {e("Date received")} | Sub-issue: {e("Sub-issue")} | Disposition: {e("disposition")} | Duplicate status: {e("duplicate_status")} | Group: {e("group_id")}</small>',
            f'<pre>{e("Consumer complaint narrative")}</pre></details>']
    (OUT/'manual_review.html').write_text('\n'.join(sections),encoding='utf-8')
    return len(records)

def md_table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in df.values])

def make_readme(overall,counts,manifest):
    by=counts.pivot(index='Issue',columns='split',values='kept').reindex(columns=SPLITS).reset_index()
    text=['# Frozen 11-label credit-card cohort','No predictive models built. Raw narrative strings and Issue labels are preserved exactly. No short-text filtering, leakage-span removal, brand masking, label merging, or semantic relabeling was applied.',
        '## Final counts',md_table(overall[['before_duplicates','kept','removed_duplicate','quarantined_conflict']].reset_index()),
        '## Retained counts by Issue',md_table(by),
        '## Policy and review choices surfaced',
        '- The 11 labels are frozen from the assessment (November 2024 support ≥30 before duplicate removal); support is not re-thresholded after quarantine.',
        '- Receipt splits are unchanged: November 2024 train; December 2024 validation; November–December 2025 test. No future observation moves into an earlier split.',
        '- Duplicate graph covers all 12,363 available credit-card narratives, including the 137 rare-label audit rows. Thus conflicts with an out-of-scope label still quarantine the in-scope records. Other products are outside this cohort graph.',
        '- Exact hashes use stripped raw text, then normalized hashes use the assessment normalization. These transformations affect matching only. Existing near-duplicate evidence adds all 309 pairs; each Jaccard score was recomputed and checked. No new similarity threshold or candidate search was introduced.',
        '- The prior assessment proposed manual confirmation of near-duplicates. Following the current instruction to use already-identified evidence, those pairs are conservatively treated as graph edges, with their unadjudicated status recorded. They are not asserted to be confirmed same-consumer events; review can later reject an edge and trigger a rebuild.',
        '- Contradictory-label connected components are quarantined in full, including their earliest records. No majority vote or relabeling. Non-conflicting components retain only the earliest receipt record; same-day ties use the smallest numeric Complaint ID. This tie-break does not imply a known intraday order.',
        '- Kept singletons remain untouched. All excess copies, including excess training copies, are removed. Normalized matching and near edges are transitive; a component need not have pairwise Jaccard ≥0.80 between every member.',
        '- Leakage and ambiguity flags remain review-only. Representative examples and new leakage samples use development data. Previously identified ambiguity and contradictory groups may expose test examples; no model/feature design is performed.',
        '## Artifacts',
        '- train.csv, validation.csv, test.csv: Complaint ID, original narrative, Issue. Only narrative is intended as input; ID is an audit key.',
        '- *_metadata.csv: source, receipt date, split, label, and group lineage, separated from model inputs.',
        '- counts_overall.csv and counts_by_issue.csv: reconciled mutually exclusive final statuses, with exact/normalized/near attribution. Quarantine takes precedence over removal.',
        '- stage_counts.csv: cumulative status snapshots after exact, normalized, and near edges; do not sum these stages. A row can move from removed to quarantined as evidence accumulates.',
        '- disposition_ledger.csv: every available credit-card narrative, including out-of-scope records; hashes, component membership, representative ID, and final disposition.',
        '- duplicate_edges.csv: complete graph edge evidence. contradictory_label_quarantine.csv: full review population; no records here are retained. rare_label_audit.csv: all four excluded labels, without merging.',
        '- duplicate_components.csv: repeated-component sizes, label sets, date bounds, and split coverage, including out-of-scope members.',
        '- manual_review.html: compact expandable reading artifact. manual_review.csv: same sample with full original text and blank reviewer-decision columns. No Company, ZIP, State, or downstream response metadata is copied.',
        '- manifest.json: fixed labels, source/evidence checksums, source funnel, matching policy, and validation results.',
        '## Validation',json.dumps(manifest['checks'],indent=2),
        'Counts reconcile per label and split. Retained components are unique across splits, normalized texts do not repeat across retained datasets, all contradictory components are excluded, and each representative is earliest. Exported narrative and label fields were re-read and compared with source fields. Existing source hashes match the assessment. The approximate evidence remains non-exhaustive; this does not prove independence of all retained complaints.',
        'Duplicate decisions use evidence from the full historical snapshot: a later contradictory label can quarantine an earlier record. This is retrospective decontamination, not a reconstruction of decisions available at the historical training date. No later observation is moved earlier. Investigation training support falls from 142 to 46; the assessed 11-label scope is preserved rather than silently changed.',
        '## Rebuild', 'Run build_credit_card_cohort.py with Python and pandas from this repository. Deterministic source order, graph IDs, date/ID ordering, review selection, and output sorting are used. Source changes cause a failure rather than silent cohort drift. Reviewer CSV edits are not consumed by this script.']
    (OUT/'README.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__': main()
