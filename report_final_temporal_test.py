"""Final frozen test metrics, paired bootstrap and publication figures; no fitting."""
from pathlib import Path
import os,json,hashlib,time,sys,importlib.metadata
R=Path(__file__).resolve().parent;O=R/'temporal_test_evaluation'
os.environ['MPLCONFIGDIR']=str(O/'mpl-cache')
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,accuracy_score,balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'tfidf':'TF-IDF','frozen':'Frozen MiniLM','finetuned':'Fine-tuned MiniLM'}
COLORS=['#2166ac','#1b9e77','#d95f02']
SHORT=['Marketing','Closing account','Fees / interest','Getting a credit card','Incorrect report information','Other',
       'Making payments','Investigation','Purchase / statement','Struggling to pay','Using card']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in df.values])
def measures(df,labels):
    p,r,f,s=precision_recall_fscore_support(df.true_Issue,df.predicted_Issue,labels=labels,zero_division=0)
    cm=confusion_matrix(df.true_Issue,df.predicted_Issue,labels=labels)
    aggregate={'macro_f1':f.mean(),'weighted_f1':np.average(f,weights=s),
               'accuracy':accuracy_score(df.true_Issue,df.predicted_Issue),
               'balanced_accuracy':balanced_accuracy_score(df.true_Issue,df.predicted_Issue)}
    detail=pd.DataFrame({'Issue':labels,'precision':p,'recall':r,'f1':f,'support':s,
        'predicted_count':cm.sum(axis=0),'predicted_share_percent':100*cm.sum(axis=0)/len(df),
        'true_share_percent':100*s/len(df),'true_positive_count':np.diag(cm)})
    return aggregate,detail,cm
def macro_from_codes(y,p,k):
    cm=np.bincount(k*y+p,minlength=k*k).reshape(k,k)
    denom=cm.sum(axis=0)+cm.sum(axis=1)
    return np.divide(2*np.diag(cm),denom,out=np.zeros(k,dtype=float),where=denom>0).mean()
def savefig(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(O/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
def main():
    assert (O/'post_inference_integrity.json').exists(), 'Inference must finish before scoring.'
    config=json.loads((O/'evaluation_config.json').read_text())
    pre=json.loads((O/'pre_inference_integrity.json').read_text());labels=pre['labels'];k=len(labels)
    aliases=dict(zip(labels,SHORT));metrics=[];details=[];matrices={};tests={};source=[]
    validation={'tfidf':R/'benchmark/tfidf_df2_C1.0_balanced_validation_predictions.csv',
        'frozen':R/'embedding_benchmark/embedding_C4.0_balanced_validation_predictions.csv',
        'finetuned':R/'finetune_benchmark/epoch_8_validation_predictions.csv'}
    for mode in NAMES:
        for split,path in [('validation',validation[mode]),('test',O/f'{mode}_test_predictions.csv')]:
            source.append(path);df=pd.read_csv(path,dtype={'Complaint ID':str})
            if split=='test':
                assert len(df)==6521
                if tests:assert df[['Complaint ID','true_Issue']].equals(next(iter(tests.values()))[['Complaint ID','true_Issue']])
                tests[mode]=df
            a,d,cm=measures(df,labels)
            metrics.append({'model':NAMES[mode],'model_key':mode,'split':split,**a})
            d.insert(0,'split',split);d.insert(0,'model',NAMES[mode]);details.append(d)
            matrices[(mode,split)]=cm
            pd.DataFrame(cm,index=labels,columns=labels).rename_axis('true_Issue').to_csv(O/f'{mode}_{split}_confusion.csv')
    metrics=pd.DataFrame(metrics);detail=pd.concat(details,ignore_index=True)
    metrics.to_csv(O/'aggregate_metrics.csv',index=False);detail.to_csv(O/'per_label_metrics.csv',index=False)
    delta=[];class_delta=[]
    for mode,name in NAMES.items():
        g=metrics[metrics.model_key==mode].set_index('split')
        delta.append({'model':name,**{f'{metric}_{split}':g.loc[split,metric] for metric in ['macro_f1','weighted_f1','accuracy','balanced_accuracy'] for split in ['validation','test']},
            **{f'{metric}_change':g.loc['test',metric]-g.loc['validation',metric] for metric in ['macro_f1','weighted_f1','accuracy','balanced_accuracy']}})
        d=detail[detail.model==name].pivot(index='Issue',columns='split',values='f1').loc[labels]
        for label,row in d.iterrows():class_delta.append({'model':name,'Issue':label,'validation_f1':row.validation,'test_f1':row.test,'f1_change':row.test-row.validation})
    delta=pd.DataFrame(delta);class_delta=pd.DataFrame(class_delta)
    delta.to_csv(O/'validation_test_metric_changes.csv',index=False)
    class_delta.to_csv(O/'validation_test_per_label_f1.csv',index=False)
    # Predeclared IID complaint-level paired bootstrap, fixed labels even if a resample lacks a class.
    seed=config['bootstrap']['seed'];reps=config['bootstrap']['replicates'];rng=np.random.default_rng(seed)
    y=pd.Categorical(tests['tfidf'].true_Issue,categories=labels).codes.astype(np.int64)
    pred=np.stack([pd.Categorical(tests[mode].predicted_Issue,categories=labels).codes.astype(np.int64) for mode in NAMES])
    assert (y>=0).all() and (pred>=0).all()
    for j,mode in enumerate(NAMES):
        assert np.isclose(macro_from_codes(y,pred[j],k),metrics[(metrics.model_key==mode)&(metrics.split=='test')].macro_f1.iloc[0])
    t=time.perf_counter();scores=np.empty((reps,3))
    for b in range(reps):
        indices=rng.integers(0,len(y),size=len(y))
        for j in range(3):scores[b,j]=macro_from_codes(y[indices],pred[j,indices],k)
    bootstrap_seconds=time.perf_counter()-t
    boot=pd.DataFrame(scores,columns=list(NAMES));pairs=[(0,1),(0,2),(1,2)];ci=[]
    for j,(mode,name) in enumerate(NAMES.items()):
        lo,hi=np.quantile(scores[:,j],[.025,.975])
        ci.append({'estimand':name,'estimate':metrics[(metrics.model_key==mode)&(metrics.split=='test')].macro_f1.iloc[0],
                   'lower_95':lo,'upper_95':hi,'bootstrap_se':scores[:,j].std(ddof=1)})
    for a,b in pairs:
        name=list(NAMES.values())[a]+' minus '+list(NAMES.values())[b];values=scores[:,a]-scores[:,b]
        boot[name]=values;lo,hi=np.quantile(values,[.025,.975])
        ci.append({'estimand':name,'estimate':ci[a]['estimate']-ci[b]['estimate'],'lower_95':lo,'upper_95':hi,'bootstrap_se':values.std(ddof=1)})
    boot.insert(0,'replicate',range(1,reps+1));boot.to_csv(O/'paired_bootstrap_replicates.csv',index=False)
    ci=pd.DataFrame(ci);ci.to_csv(O/'macro_f1_bootstrap_intervals.csv',index=False)
    # Focal flow reporting after fixed predictions, not used by any model.
    flows=[]
    for mode,name in NAMES.items():
        for split in ['validation','test']:
            cm=matrices[(mode,split)]
            for j in [5,7]:
                for direction,counts in [('true_to_predicted',cm[j,:]),('predicted_from_true',cm[:,j])]:
                    denom=int(counts.sum())
                    for b,count in enumerate(counts):flows.append({'model':name,'split':split,'focal_Issue':labels[j],
                        'direction':direction,'other_Issue':labels[b],'count':int(count),'denominator':denom,
                        'conditional_percent':100*count/denom if denom else np.nan})
    flow=pd.DataFrame(flows);flow.to_csv(O/'other_investigation_flows.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(3);width=.34
    val=metrics[metrics.split=='validation'].set_index('model_key').loc[list(NAMES)].macro_f1.to_numpy()
    test=metrics[metrics.split=='test'].set_index('model_key').loc[list(NAMES)].macro_f1.to_numpy()
    ax.bar(x-width/2,val,width,color=COLORS,alpha=.4,label='December 2024 validation')
    ax.bar(x+width/2,test,width,color=COLORS,label='2025 temporal test')
    modelci=ci.iloc[:3];err=np.vstack([test-modelci.lower_95.to_numpy(),modelci.upper_95.to_numpy()-test])
    ax.errorbar(x+width/2,test,yerr=err,fmt='none',ecolor='#222222',capsize=4)
    for j in range(3):
        ax.text(x[j]-width/2,val[j]+.009,f'{val[j]:.3f}',ha='center',fontsize=9)
        ax.text(x[j]+width/2,test[j]+.029,f'{test[j]:.3f}',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=list(NAMES.values()),ylabel='Macro-F1',ylim=(0,max(val.max(),test.max())+.11),
           title='Frozen models on a later complaint cohort')
    ax.legend(loc='upper right',fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.5,.01,'Test whiskers: 95% paired complaint-bootstrap percentile intervals',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1));savefig(fig,'model_macro_f1_validation_test')
    fig,ax=plt.subplots(figsize=(9,6));ypos=np.arange(k)
    for j,(mode,name) in enumerate(NAMES.items()):
        f=detail[(detail.model==name)&(detail.split=='test')].set_index('Issue').loc[labels].f1
        ax.barh(ypos+(j-1)*.23,f,height=.22,color=COLORS[j],label=name)
    ax.set(yticks=ypos,yticklabels=SHORT,xlim=(0,1),xlabel='2025 F1',title='Per-label temporal test performance')
    ax.invert_yaxis();ax.legend(loc='lower right',fontsize=9);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();savefig(fig,'per_label_2025_f1')
    changes=class_delta.pivot(index='Issue',columns='model',values='f1_change').loc[labels,list(NAMES.values())].to_numpy()
    bound=max(.01,np.abs(changes).max());fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(changes,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
    for row in range(k):
        for col in range(3):ax.text(col,row,f'{changes[row,col]:+.3f}',ha='center',va='center',color='white' if abs(changes[row,col])>.65*bound else '#111111')
    ax.set(xticks=range(3),xticklabels=list(NAMES.values()),yticks=range(k),yticklabels=SHORT,
        title='F1 change: 2025 test minus December validation')
    fig.colorbar(im,ax=ax,label='F1 change',shrink=.8);fig.tight_layout();savefig(fig,'per_label_f1_temporal_change')
    report=['# Final frozen temporal test evaluation',
        'The actual development-selected models were evaluated once on 6,521 frozen 2025 narratives. No November+December refit, retraining, tuning, calibration, threshold/class-weight change, label change, or feature cleaning occurred. Predictive input was narrative text only. Test integrity checks were completed before inference. Earlier artifacts remain unchanged.',
        '## Validation and temporal test metrics']
    for metric in ['macro_f1','weighted_f1','accuracy','balanced_accuracy']:
        g=delta[['model',f'{metric}_validation',f'{metric}_test',f'{metric}_change']].copy()
        for col in g.columns[1:]:g[col]=g[col].map(lambda v:f'{v:.4f}')
        report += [f'### {metric}',table(g)]
    ranking=metrics[metrics.split=='test'].sort_values('macro_f1',ascending=False).model.tolist()
    report += ['Test macro-F1 ranking: '+' > '.join(ranking)+'. This is a descriptive ranking; paired uncertainty is reported below.',
        '![Macro-F1](model_macro_f1_validation_test.png)','## Paired bootstrap uncertainty']
    fmt=ci.copy()
    for col in fmt.columns[1:]:fmt[col]=fmt[col].map(lambda v:f'{v:.4f}')
    report += [table(fmt),f'{reps:,} ordinary IID bootstrap replicates, seed {seed}. Each draws {len(y):,} complaint indices with replacement; the same indices are used for all three models. Fixed 11-label macro-F1, zero_division=0, percentile 2.5/97.5 intervals. No stratification, tuning, or selection from bootstrap results. Full replicate scores/differences are saved. Runtime {bootstrap_seconds:.2f}s.',
        'Intervals quantify resampling uncertainty conditional on this frozen sample and fitted models. They do not include development selection, label uncertainty, retrospective cohort design, or future population uncertainty. Complaint-level resampling assumes retained complaints are sufficiently independent; undetected related narratives could make intervals optimistic.',
        '## Descriptive generalization findings']
    persists=ranking==list(NAMES.values())
    report += ['The development macro-F1 ranking '+('persists' if persists else 'does not persist')+' on the temporal test cohort.']
    for _,row in ci.iloc[3:].iterrows():
        report += [f'{row.estimand}: {row.estimate:+.4f}, 95% paired interval [{row.lower_95:+.4f}, {row.upper_95:+.4f}]. '+
            ('The interval includes zero; the numerical ordering does not establish a difference.' if row.lower_95<=0<=row.upper_95 else
             'The interval excludes zero under the stated complaint-bootstrap assumptions; this does not establish superiority across future populations.')]
    for label in [labels[5],labels[7],labels[4],labels[9]]:
        rows=class_delta[class_delta.Issue==label]
        report += [f'{label}: '+ '; '.join(f'{row.model} F1 {row.validation_f1:.4f} → {row.test_f1:.4f} ({row.f1_change:+.4f})' for _,row in rows.iterrows())+'.']
    for label in [labels[5],labels[7]]:
        d=detail[(detail.model=='Fine-tuned MiniLM')&(detail.split=='test')&(detail.Issue==label)].iloc[0]
        v=detail[(detail.model=='Fine-tuned MiniLM')&(detail.split=='validation')&(detail.Issue==label)].iloc[0]
        report += [f'Fine-tuned {aliases[label]}: {int(d.predicted_count)} test predictions ({d.predicted_share_percent:.3f}%) versus {int(d.support)} true examples ({d.true_share_percent:.3f}%). '+
            f'December prediction share was {v.predicted_share_percent:.3f}%. Test recall {d.recall:.4f}, precision {d.precision:.4f}. '+
            f'Prediction/true-support ratio is {d.predicted_count/d.support:.3f}; these counts directly show the extent of under-prediction without changing the classifier.']
    report += ['These are post-prediction diagnostics of CFPB label reproduction. Findings are reported only: no scored-test-driven development action, replacement model selection, or refit follows this evaluation.',
        '## Full per-label metrics and prediction shares',
        'Tables include all precision/recall/F1/support and predicted count/share for both periods. Undefined precision for an empty predicted class is displayed as 0 (zero_division=0). CSVs use exact frozen labels; figure/table display names are shortened.']
    for name in NAMES.values():
        for split in ['validation','test']:
            g=detail[(detail.model==name)&(detail.split==split)].drop(columns=['model','split','true_positive_count','true_share_percent']).copy()
            g.Issue=g.Issue.map(aliases)
            for col in ['precision','recall','f1','predicted_share_percent']:g[col]=g[col].map(lambda v:f'{v:.4f}')
            report += [f'### {name}: {split}',table(g)]
    report += ['![2025 per-label F1](per_label_2025_f1.png)','## Per-label F1 comparison across periods']
    comparison=class_delta.pivot(index='Issue',columns='model',values=['validation_f1','test_f1','f1_change']).loc[labels]
    comparison.columns=[f'{model}: {metric}' for metric,model in comparison.columns]
    comparison.insert(0,'Issue',[aliases[x] for x in comparison.index]);comparison=comparison.reset_index(drop=True)
    for col in comparison.columns[1:]:comparison[col]=comparison[col].map(lambda v:f'{v:.4f}')
    report += [table(comparison),'![F1 changes](per_label_f1_temporal_change.png)',
        '## Other and Investigation: two-way flows',
        'Cell = count (conditional percentage). Outgoing rows condition on true focal class; incoming rows condition on predictions assigned to it. Correct examples are included. Zero-denominator percentages are NA.']
    for label in [labels[5],labels[7]]:
        for direction in ['true_to_predicted','predicted_from_true']:
            rows=[]
            for name in NAMES.values():
                for split in ['validation','test']:
                    g=flow[(flow.model==name)&(flow.split==split)&(flow.focal_Issue==label)&(flow.direction==direction)].set_index('other_Issue').loc[labels]
                    rows.append([name,split,int(g.denominator.iloc[0])]+[f'{int(row["count"])} ({row.conditional_percent:.1f}%)' if pd.notna(row.conditional_percent) else '0 (NA)' for _,row in g.iterrows()])
            report += [f'### {aliases[label]}: {direction}',table(pd.DataFrame(rows,columns=['Model','Period','Denominator']+SHORT))]
    runtimes={mode:json.loads((O/f'{mode}_runtime_manifest.json').read_text()) for mode in NAMES}
    (O/'runtime_summary.json').write_text(json.dumps(runtimes,indent=2))
    report += ['## Runtime and exact model identities',
        'TF-IDF and frozen embeddings use their original CPU environments. Fine-tuned MiniLM uses its original CUDA environment/GPU. No runtime optimization was selected using test data. Costs across CPU and GPU are not hardware-normalized. TF-IDF and frozen logistic-regression prediction costs are separated from representation costs; the fine-tuned encoder and task head are measured jointly because they execute in one forward pass. Model loading and tokenization are reported separately where possible.',
        '```json\n'+json.dumps({mode:{'timings':r['timings'],'details':r['details']} for mode,r in runtimes.items()},indent=2)+'\n```',
        '## Integrity and interpretation safeguards',
        'No pre-existing whole-test-file SHA-256 was found in the cohort-building manifest. The hash recorded immediately before inference is therefore a new evaluation commitment, not an earlier commitment. Independently, all 6,521 IDs/labels match frozen test metadata and kept ledger membership, each narrative matches the ledger’s pre-existing exact-text SHA-256, and receipt years are 2025. The fine-tuned checkpoint matches its development-recorded hash. All cohort, training, validation, model, and prior benchmark files are hashed before/after evaluation.',
        'The target is the CFPB Issue attached to a published/redacted consumer complaint narrative, not verified ground truth about the underlying problem. Narrative availability is selected, labels can be ambiguous, and one complaint can contain multiple issues. This is a temporal generalization test on the specified 2025 cohort, not a guarantee for every future complaint population. The cohort was retrospectively deduplicated/decontaminated; it does not reconstruct information available at historical training time. The 2025 period was descriptively inspected in the original audit, but it was not used for predictive feature selection, modeling, or tuning.',
        '## Reproducibility',
        'evaluation_config.json fixes the procedure; pre_inference_integrity.json and protected_before_sha256.json lock integrity. All test predictions, metrics, confusion matrices, class flows, bootstrap replicates/intervals, runtime records, and PNG/SVG/PDF figures are saved. test_evaluation_manifest.json contains versions, source/model hashes, environment identities, seeds, and final protection verification. Do not rerun inference or begin development using this scored holdout without an explicitly authorized new development cycle and new future holdout.']
    (O/'README.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    before=json.loads((O/'protected_before_sha256.json').read_text())
    assert before=={path:sha(R/path) for path in before}
    manifest={'evaluation_config':config,'test_integrity':pre,'test_ranking':ranking,'bootstrap_seconds':bootstrap_seconds,
        'all_protected_artifacts_unchanged':True,'protected_files':len(before),'training_or_adaptation_performed':False,
        'prediction_and_validation_sha256':{str(p.relative_to(R)):sha(p) for p in source},
        'source_sha256':{p.name:sha(p) for p in [Path(__file__),R/'final_test_worker.py',R/'run_final_temporal_test.py']},
        'models_and_runtime':runtimes,'python':sys.version,
        'analysis_packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    model_paths={'tfidf':R/'benchmark/selected_pipeline.joblib',
                 'frozen_classifier':R/'embedding_benchmark/selected_classifier.joblib',
                 'finetuned':R/'finetune_benchmark/best_model.safetensors'}
    manifest['selected_model_sha256']={name:sha(path) for name,path in model_paths.items()}
    manifest['pretrained_identity']=json.loads((R/'embedding_benchmark/model_download.json').read_text())
    manifest['evaluation_output_sha256']={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in
        ['test_evaluation_manifest.json','verification.json']}
    (O/'test_evaluation_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(metrics.to_string(index=False));print(ci.to_string(index=False))

if __name__=='__main__':main()
