"""Audit saved December predictions only; no model loading, fitting, or test access."""
from pathlib import Path
import os, json, hashlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'finetune_benchmark' / 'epoch_class_diagnostics'
os.environ['MPLCONFIGDIR'] = str(OUT / 'mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |', '| '+' | '.join(['---']*len(df.columns))+' |'] +
                     ['| '+' | '.join(map(str,row))+' |' for row in df.values])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [(f'Epoch {i}', ROOT/'finetune_benchmark'/f'epoch_{i}_validation_predictions.csv') for i in range(1,9)] + [
        ('TF-IDF', ROOT/'benchmark'/'tfidf_df2_C1.0_balanced_validation_predictions.csv'),
        ('Frozen MiniLM', ROOT/'embedding_benchmark'/'embedding_C4.0_balanced_validation_predictions.csv')]
    label_path = ROOT/'finetune_benchmark'/'selected_per_label.csv'
    labels = pd.read_csv(label_path).Issue.tolist()
    short = ['Marketing','Closing account','Fees / interest','Getting card','Report information','Other',
             'Making payments','Investigation','Purchase / statement','Struggling to pay','Using card']
    aliases = dict(zip(labels, short))
    focal = [labels[5], labels[7]]
    sources = [p for _,p in paths] + [label_path, ROOT/'finetune_benchmark'/'epoch_history.csv',
                                      ROOT/'finetune_benchmark'/'loss_weights.csv']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    rows, flows, matrices = [], [], []
    reference = None
    for name,path in paths:
        df = pd.read_csv(path)
        assert not df['Complaint ID'].duplicated().any()
        if reference is None: reference = df[['Complaint ID','true_Issue']]
        assert reference.equals(df[['Complaint ID','true_Issue']])
        assert set(df.true_Issue) <= set(labels) and set(df.predicted_Issue) <= set(labels)
        cm = confusion_matrix(df.true_Issue,df.predicted_Issue,labels=labels)
        p,r,f,s = precision_recall_fscore_support(df.true_Issue,df.predicted_Issue,labels=labels,zero_division=0)
        for j,label in enumerate(labels):
            rows.append(dict(model=name,epoch=int(name.split()[1]) if name.startswith('Epoch') else None,
                Issue=label,true_support=int(s[j]),true_share_percent=100*s[j]/len(df),
                predicted_count=int(cm[:,j].sum()),predicted_share_percent=100*cm[:,j].sum()/len(df),
                true_positive_count=int(cm[j,j]),precision=p[j],recall=r[j],f1=f[j]))
        for a,true in enumerate(labels):
            for b,pred in enumerate(labels):
                matrices.append(dict(model=name,true_Issue=true,predicted_Issue=pred,count=int(cm[a,b])))
        for label in focal:
            j=labels.index(label)
            for direction,counts,denominator in [('true_to_predicted',cm[j,:],int(cm[j,:].sum())),
                                                  ('predicted_from_true',cm[:,j],int(cm[:,j].sum()))]:
                for k,count in enumerate(counts):
                    flows.append(dict(model=name,focal_Issue=label,direction=direction,other_Issue=labels[k],
                        count=int(count),denominator=denominator,
                        conditional_percent=100*count/denominator if denominator else np.nan))
        assert cm.sum()==len(df)
    metrics=pd.DataFrame(rows); flow=pd.DataFrame(flows)
    history=pd.read_csv(ROOT/'finetune_benchmark'/'epoch_history.csv')
    training_counts=pd.read_csv(ROOT/'finetune_benchmark'/'loss_weights.csv').set_index('Issue').train_support
    assert training_counts.loc[labels[5]]==322 and training_counts.rank(ascending=False).loc[labels[5]]==3
    for name,_ in paths:
        g=metrics[metrics.model==name]
        assert g.predicted_count.sum()==len(reference)
        assert np.isclose(g.predicted_share_percent.sum(),100)
        assert g.true_support.sum()==len(reference)
        if name.startswith('Epoch'):
            e=int(name.split()[1])
            assert np.isclose(g.f1.mean(),history.set_index('epoch').loc[e,'macro_f1'],atol=1e-12)
    for _,g in flow.groupby(['model','focal_Issue','direction']):
        assert g['count'].sum()==g.denominator.iloc[0]
        if g.denominator.iloc[0]: assert np.isclose(g.conditional_percent.sum(),100)
        else: assert g.conditional_percent.isna().all()
    metrics.to_csv(OUT/'per_class_by_epoch_and_baseline.csv',index=False)
    flow.to_csv(OUT/'focal_class_flows.csv',index=False)
    pd.DataFrame(matrices).to_csv(OUT/'all_confusion_counts.csv',index=False)
    flags=[]
    for label in focal:
        g=metrics[(metrics.Issue==label)&metrics.epoch.notna()].sort_values('epoch')
        c=g.predicted_count.to_numpy(); tp=g.true_positive_count.to_numpy()
        flags.append(dict(Issue=label,predicted_counts_by_epoch=c.tolist(),true_positives_by_epoch=tp.tolist(),
            monotonically_nonincreasing=bool(np.all(np.diff(c)<=0)),
            first_zero_prediction_epoch=int(np.flatnonzero(c==0)[0]+1) if np.any(c==0) else None,
            absent_at_final_epoch=bool(c[-1]==0),
            final_vs_first_prediction_count_change_percent=100*(c[-1]-c[0])/c[0] if c[0] else None))
    (OUT/'disappearance_flags.json').write_text(json.dumps(flags,indent=2))
    fig=plt.figure(figsize=(12,7.5)); gs=fig.add_gridspec(2,2,height_ratios=[1.5,1])
    ax=fig.add_subplot(gs[0,:])
    colors=['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#d62728',
            '#17becf','#111111','#9467bd','#c5b0d5','#8c564b']
    shares=np.array([metrics[(metrics.model==f'Epoch {e}')].set_index('Issue').loc[labels].predicted_share_percent for e in range(1,9)]).T
    ax.stackplot(range(1,9),shares,labels=short,colors=colors)
    ax.set(xlim=(1,8),ylim=(0,100),ylabel='Predicted share (%)',title='December validation: predicted class share during fine-tuning')
    ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=8,frameon=False)
    for col,label in enumerate(focal):
        a=fig.add_subplot(gs[1,col]);g=metrics[(metrics.Issue==label)&metrics.epoch.notna()]
        a.plot(g.epoch,g.predicted_share_percent,'o-',color=colors[labels.index(label)],label='Fine-tuned')
        for name,style,color in [('TF-IDF','--','#2166ac'),('Frozen MiniLM','-.','#1b9e77')]:
            a.axhline(metrics[(metrics.model==name)&(metrics.Issue==label)].predicted_share_percent.iloc[0],ls=style,color=color,label=name)
        a.axhline(g.true_share_percent.iloc[0],ls=':',color='#777777',label='True class share')
        a.set(xlim=(1,8),ylim=(0,None),xticks=range(1,9),xlabel='Epoch',ylabel='Predicted share (%)',title=aliases[label])
        a.legend(fontsize=8);a.grid(alpha=.15)
    fig.subplots_adjust(right=.78,hspace=.42,wspace=.28,left=.07,bottom=.08,top=.94)
    fig.savefig(OUT/'predicted_class_share_by_epoch.png',dpi=160);plt.close(fig)
    content=['# December validation: epoch class shares and flows',
        'Saved predictions only: no training, model loading, new epochs, loss changes, narrative access, or 2025 access. All ten prediction files have identical December complaint IDs and true labels. Class names in tables are shortened only for display; CSVs retain the exact frozen labels.',
        '## Findings',
        'Other is severely under-predicted from epoch 1: counts are 17, 9, 8, 5, 5, 9, 11, 12 (0.19–0.63% of predictions), against 350 true examples (12.99%). This is not monotonic disappearance: prediction mass bottoms out at epochs 4–5, then partly recovers. Correct Other predictions are 7, 7, 4, 2, 2, 3, 3, 4. Epoch-8 recall is 1.14%, compared with 26.86% for TF-IDF and 27.71% for frozen MiniLM. Other is the third-largest training label (322 examples), so its suppression cannot be explained simply as rarity.',
        'Investigation is effectively absent from the start: prediction counts 0, 0, 0, 1, 0, 2, 2, 2, with zero correct predictions at every epoch. Its 46 true examples represent 1.71% of validation. TF-IDF predicts 15 Investigation examples with 6 correct; frozen MiniLM predicts 100 with 15 correct. Frozen MiniLM increases Investigation recall at the cost of precision (15%).',
        'At epoch 8, true Other examples disperse chiefly to Purchase (76/350, 21.71%), Marketing (72/350, 20.57%), and Payments (43/350, 12.29%). For those destinations TF-IDF counts are 68, 30, 26, and frozen MiniLM counts are 45, 36, 24. Other suppression is spread across multiple alternative labels, rather than a single replacement class.',
        'Investigation has a more concentrated substitution: 24/46 (52.17%) go to Incorrect information on your report at both epochs 1 and 8, compared with 10/46 (21.74%) for TF-IDF and 14/46 (30.43%) for frozen MiniLM. At epoch 8, another 8 go to Purchase and 7 to Getting a credit card. The two epoch-8 Investigation predictions are actually one Payments and one Purchase example. The 12 epoch-8 Other predictions comprise 4 true Other, 3 Fees, 3 Purchase, and 2 Closing-account examples.',
        'Interpretation: the visible failure is specific class suppression already present after the first epoch, not progressive loss of all classes as aggregate macro-F1 improves. Other is not a minority training class; Investigation is. Their different outgoing patterns suggest distinct label-boundary difficulties. These predictions alone cannot separate optimization, representation changes, loss behavior, input truncation, or label overlap as causes. Baselines have stronger recognition of both labels, so the frozen labels are not wholly unlearnable. Epoch 0 predictions were not saved, so the onset within epoch 1 is unknown. No additional training or label decision is justified solely by this audit.',
        '![Predicted class shares](predicted_class_share_by_epoch.png)',
        'Top: all 11 prediction shares sum to 100% at each epoch. Bottom: separate scales expose Other and Investigation, with unchanged baseline prediction shares and true prevalence for context.',
        '## Predictions and per-class metrics at every epoch',
        'Share is percentage of all 2,695 validation predictions. Metrics use the fixed 11-label order and zero_division=0; zero predicted examples makes precision undefined, reported as 0.']
    for name,_ in paths:
        g=metrics[metrics.model==name][['Issue','true_support','predicted_count','predicted_share_percent','precision','recall','f1']].copy()
        g.Issue=g.Issue.map(aliases)
        for col in ['predicted_share_percent','precision','recall','f1']:g[col]=g[col].map(lambda x:f'{x:.4f}')
        content += [f'### {name}',table(g)]
    content += ['## Two-way flows and baseline comparisons',
        'Each cell is count (conditional percentage). Outgoing rows condition on the true focal class and include correct predictions. Incoming rows condition on predictions assigned to the focal class and include correct predictions. An empty predicted class has denominator 0: its incoming percentages are undefined (NA), not 0%.']
    for label in focal:
        for direction in ['true_to_predicted','predicted_from_true']:
            data=[]
            for name,_ in paths:
                g=flow[(flow.model==name)&(flow.focal_Issue==label)&(flow.direction==direction)].set_index('other_Issue').loc[labels]
                data.append([name,int(g.denominator.iloc[0])]+[f'{int(row["count"])} ({row.conditional_percent:.1f}%)' if pd.notna(row.conditional_percent) else '0 (NA)' for _,row in g.iterrows()])
            content += [f'### {aliases[label]}: '+('true examples → predicted labels' if direction=='true_to_predicted' else 'predicted examples ← true labels'),
                        table(pd.DataFrame(data,columns=['Model','Denominator']+short))]
    content += ['## Disappearance flags',table(pd.DataFrame(flags).drop(columns=['predicted_counts_by_epoch','true_positives_by_epoch']).astype(str)),
                'These are descriptive validation diagnostics. A shrinking prediction share does not by itself identify the causal mechanism; confusion flows show which labels receive the displaced examples. No further training or cleaning decision is made here.']
    (OUT/'README.md').write_text('\n\n'.join(content)+'\n',encoding='utf-8')
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in sources}
    manifest=dict(source_sha256=sha(Path(__file__)),input_sha256=hashes,validation_rows=len(reference),
                  prediction_alignment_verified=True,input_files_unchanged=True,test_accessed=False,
                  epoch_macro_f1_matches_history=True,class_counts_and_flow_percentages_reconcile=True,
                  model_loaded=False,training_performed=False,
                  packages={'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,
                            'scikit-learn':sklearn.__version__})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(flags,indent=2))

if __name__=='__main__':main()
