from pathlib import Path
import json, re
import pandas as pd
R=Path(__file__).resolve().parent; O=R/'profiling'
ds=[]
for f in sorted(R.glob('CCDB*.csv')):
    chunks=pd.read_csv(f,dtype=str,keep_default_na=False,chunksize=100000)
    ds.extend(c[c.Product=='Credit card'] for c in chunks)
d=pd.concat(ds,ignore_index=True); d['year']=d['Date received'].str[:4]
n=d[d['Consumer complaint narrative'].str.strip()!=''].copy()
n['split']=n['Date received'].map(lambda t:'train' if t<'2024-12-01' else ('validation' if t<'2025-01-01' else 'test'))
pd.crosstab(n.Issue,n.split).to_csv(O/'recommended_split_counts.csv')
stats={}
for split,g in n.groupby('split'):
    stats[split]={'rows':len(g),'labels':g.Issue.value_counts().to_dict(),'min_date':g['Date received'].min(),'max_date':g['Date received'].max()}
    trainlabels=n[n.split=='train'].Issue.value_counts()
    supported=set(trainlabels[trainlabels>=50].index)
    stats[split]['supported_labels_train_ge50_rows']=int(g.Issue.isin(supported).sum())
stats['label_sets']={str(y):sorted(g.Issue.unique()) for y,g in d.groupby('year')}
stats['subissue_sets']={str(y):sorted(g['Sub-issue'].unique()) for y,g in d.groupby('year')}
(O/'supplement.json').write_text(json.dumps(stats,indent=2))
# Deterministic review sample: first by complaint ID within each issue/year,
# plus literal-label echoes and field markers. Display only locally for review.
sample=n.sort_values('Complaint ID').groupby(['year','Issue']).head(2)
for _,r in sample.iterrows():
    print('\n',r['Complaint ID'],r.year,r.Issue,'\n',r['Consumer complaint narrative'][:400])
print('\nSPLIT COUNTS', {k:v['rows'] for k,v in stats.items() if 'rows' in v})
