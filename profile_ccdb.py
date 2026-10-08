"""Descriptive profiling only; no predictive models. Run with Python + pandas/numpy."""
from pathlib import Path
import hashlib, json, re
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'profiling'
OUT.mkdir(exist_ok=True)
files = sorted(ROOT.glob('CCDB*.csv'))
frames = []
summary = {}
for p in files:
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    d = d.apply(lambda s: s.str.strip())
    d['source'] = p.name
    frames.append(d)
    summary[p.name] = {'rows':len(d), 'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'columns':list(d.columns[:-1]), 'date_ranges':{c:[d.loc[d[c]!='',c].min(),d.loc[d[c]!='',c].max()] for c in ['Date received','Date sent to company']},
        'missing':{c:int((d[c]=='').sum()) for c in d.columns[:-1]},
        'distinct':{c:int(d[c].nunique()) for c in d.columns[:-1]},
        'duplicate_ids':int(d['Complaint ID'].duplicated().sum()),
        'duplicate_rows':int(d.drop(columns='source').duplicated().sum())}
    print(p.name, len(d), flush=True)
d = pd.concat(frames,ignore_index=True)
d['year'] = d['Date received'].str[:4]
d['month'] = d['Date received'].str[:7]
d['has_narrative'] = d['Consumer complaint narrative']!=''
for col in ['Product','Issue','Submitted via','Company response to consumer','Timely response?']:
    d.groupby(['source',col],dropna=False).agg(rows=('Complaint ID','size'),narratives=('has_narrative','sum')).reset_index().to_csv(OUT/(col.replace(' ','_').replace('?','')+'_counts.csv'),index=False)
d.groupby(['source','month']).agg(rows=('Complaint ID','size'),narratives=('has_narrative','sum')).reset_index().to_csv(OUT/'monthly_counts.csv',index=False)
d.groupby(['source','Product','Issue','Sub-issue']).size().rename('rows').reset_index().to_csv(OUT/'taxonomy_counts.csv',index=False)
summary['combined']={'rows':len(d),'duplicate_ids':int(d['Complaint ID'].duplicated().sum()),'cross_file_ids':len(set(frames[0]['Complaint ID']) & set(frames[1]['Complaint ID'])), 'products':sorted(d.Product.unique())}
cc=d[d.Product.str.contains('credit card',case=False)].copy()
cc.groupby(['year','Product','Sub-product','Issue']).agg(rows=('Complaint ID','size'),narratives=('has_narrative','sum')).reset_index().to_csv(OUT/'credit_card_issue_counts.csv',index=False)
cc.groupby(['year','Issue','Sub-issue']).size().rename('rows').reset_index().to_csv(OUT/'credit_card_subissue_counts.csv',index=False)
cc.groupby(['year','Submitted via']).agg(rows=('Complaint ID','size'),narratives=('has_narrative','sum')).reset_index().to_csv(OUT/'credit_card_channels.csv',index=False)
cc.groupby(['year','Issue','has_narrative']).size().rename('rows').reset_index().to_csv(OUT/'credit_card_selection.csv',index=False)
n=cc[cc.has_narrative].copy()
def norm(t): return ' '.join(re.findall(r'[a-z0-9]+',t.lower()))
n['norm']=n['Consumer complaint narrative'].map(norm)
n['words']=n.norm.str.split().str.len()
n['chars']=n['Consumer complaint narrative'].str.len()
n['exact_hash']=n['Consumer complaint narrative'].map(lambda t:hashlib.sha256(t.encode()).hexdigest())
n['norm_hash']=n.norm.map(lambda t:hashlib.sha256(t.encode()).hexdigest())
summary['credit_card']={'rows':len(cc),'narratives':len(n),'products':cc.Product.value_counts().to_dict(),'years':{},'lengths':{},'duplicates':{}}
for year,g in cc.groupby('year'):
    ng=n[n.year==year]
    summary['credit_card']['years'][year]={'rows':len(g),'narratives':len(ng),'date_range':[g['Date received'].min(),g['Date received'].max()], 'company_top10':g.Company.value_counts().head(10).to_dict(), 'narrative_company_top10':ng.Company.value_counts().head(10).to_dict()}
    summary['credit_card']['lengths'][year]={c:ng[c].quantile([0,.01,.1,.25,.5,.75,.9,.99,1]).to_dict() for c in ['words','chars']}
for col in ['exact_hash','norm_hash']:
    groups=n.groupby(col)
    sz=groups.size()
    summary['credit_card']['duplicates'][col]={'groups_repeated':int((sz>1).sum()),'rows_in_repeated_groups':int(sz[sz>1].sum()),'excess_rows':int((sz-1).clip(lower=0).sum()),'cross_year_groups':int((groups.year.nunique()>1).sum()),'conflicting_issue_groups':int((groups.Issue.nunique()>1).sum())}
    n[n[col].isin(sz[sz>1].index)][['Complaint ID','year','Issue',col]].to_csv(OUT/(col+'_groups.csv'),index=False)
# Exact narrative reuse across every product; hashes only are exported.
alln=d[d.has_narrative].copy()
alln['hash']=alln['Consumer complaint narrative'].map(lambda t:hashlib.sha256(t.encode()).hexdigest())
ag=alln.groupby('hash'); sizes=ag.size()
summary['all_narrative_duplicates']={'narratives':len(alln),'repeated_groups':int((sizes>1).sum()),'rows_in_repeated_groups':int(sizes[sizes>1].sum()),'excess_rows':int((sizes-1).clip(lower=0).sum()),'cross_year_groups':int((ag.year.nunique()>1).sum()),'cross_product_groups':int((ag.Product.nunique()>1).sum()),'conflicting_issue_groups':int((ag.Issue.nunique()>1).sum())}
# High-similarity screen: SimHash candidate buckets then exact token 5-shingle Jaccard.
# This is a duplicate audit, not an exhaustive near-duplicate search or a model.
sets=[]; buckets=defaultdict(list); candidates=set()
for i,t in enumerate(n.norm):
    toks=t.split(); shingles=set(' '.join(toks[j:j+5]) for j in range(max(0,len(toks)-4)))
    sets.append(shingles)
    if len(shingles)<20: continue
    vec=[0]*64
    for s in shingles:
        h=int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'little')
        for bit in range(64): vec[bit]+=1 if h & (1<<bit) else -1
    sig=sum(1<<bit for bit,v in enumerate(vec) if v>=0)
    for band in range(8):
        key=(band,(sig>>(band*8))&255)
        for j in buckets[key]:
            if min(len(sets[j]),len(shingles))/max(len(sets[j]),len(shingles))>=.8: candidates.add((j,i))
        buckets[key].append(i)
print('Near-duplicate candidate pairs',len(candidates),flush=True)
rows=[]; nn=n.reset_index(drop=True)
for i,j in candidates:
    a,b=sets[i],sets[j]; jac=len(a&b)/len(a|b)
    if jac>=.8 and nn.at[i,'norm_hash']!=nn.at[j,'norm_hash']:
        rows.append({'id_a':nn.at[i,'Complaint ID'],'id_b':nn.at[j,'Complaint ID'],'year_a':nn.at[i,'year'],'year_b':nn.at[j,'year'],'issue_a':nn.at[i,'Issue'],'issue_b':nn.at[j,'Issue'],'jaccard':jac})
pd.DataFrame(rows).to_csv(OUT/'near_duplicate_pairs.csv',index=False)
summary['credit_card']['near_duplicates']={'candidate_pairs':len(candidates),'distinct_text_pairs_jaccard_ge_0.8':len(rows),'cross_year_pairs':sum(r['year_a']!=r['year_b'] for r in rows),'conflicting_issue_pairs':sum(r['issue_a']!=r['issue_b'] for r in rows),'affected_ids':len(set(v for r in rows for v in [r['id_a'],r['id_b']]))}
# Literal label echo and form-field markers; ordinary semantic overlap is not leakage.
echo=[]
for year,g in n.groupby('year'):
    for issue,gg in g.groupby('Issue'):
        literal=sum(norm(issue) in t for t in gg.norm)
        sub=sum(bool(s) and norm(s) in t for s,t in zip(gg['Sub-issue'],gg.norm))
        markers=sum(bool(re.search(r'\b(issue|sub[ -]?issue|product|complaint type)\s*:',t,re.I)) for t in gg['Consumer complaint narrative'])
        echo.append({'year':year,'Issue':issue,'narratives':len(gg),'literal_issue':literal,'literal_subissue':sub,'field_markers':markers})
pd.DataFrame(echo).to_csv(OUT/'label_echo_audit.csv',index=False)
# Candidate time windows, before any duplicate quarantine.
n['split']=np.select([n['Date received']<'2024-12-01',n['Date received']<'2025-01-01',n['Date received']<'2025-12-01'],['train','validation','gap_2025_nov'],default='test')
n.groupby(['split','Issue']).size().rename('rows').reset_index().to_csv(OUT/'proposed_split_counts.csv',index=False)
n.groupby(['year','month']).agg(rows=('Complaint ID','size'),short_under20=('words',lambda s:int((s<20).sum()))).reset_index().to_csv(OUT/'credit_card_monthly.csv',index=False)
summary['credit_card']['short_under20']=int((n.words<20).sum())
summary['dates']={c:{'invalid':int(pd.to_datetime(d[c],errors='coerce').isna().sum())} for c in ['Date received','Date sent to company']}
lag=(pd.to_datetime(d['Date sent to company'])-pd.to_datetime(d['Date received'])).dt.days
summary['sent_lag']={'negative':int((lag<0).sum()),'quantiles':lag.quantile([0,.5,.9,.99,1]).to_dict()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary['credit_card'],indent=2),flush=True)
