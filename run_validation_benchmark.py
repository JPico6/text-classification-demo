"""Train/validation benchmark. No code path reads, hashes, or scores test data."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'benchmark/mpl-cache'))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import hashlib, json, platform, time, warnings, sys
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits, threadpool_info
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_text import tokenize, LEXICAL_CUES, lexical_predict

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'benchmark'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(y,pred,labels):
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    summary={'macro_f1':f1_score(y,pred,labels=labels,average='macro',zero_division=0),
        'weighted_f1':f1_score(y,pred,labels=labels,average='weighted',zero_division=0),
        'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred)}
    detail=pd.DataFrame({'Issue':labels,'precision':p,'recall':r,'f1':f,'support':s})
    return summary,detail,confusion_matrix(y,pred,labels=labels)

def write_eval(name,y,pred,labels,ids):
    summary,detail,cm=metrics(y,pred,labels)
    detail.to_csv(OUT/f'{name}_per_label.csv',index=False)
    pd.DataFrame(cm,index=labels,columns=labels).rename_axis('true_Issue').to_csv(OUT/f'{name}_confusion.csv')
    pd.DataFrame({'Complaint ID':ids,'true_Issue':y,'predicted_Issue':pred}).to_csv(OUT/f'{name}_validation_predictions.csv',index=False)
    return summary

def main():
    total_start=time.perf_counter()
    OUT.mkdir(exist_ok=True)
    labels=json.loads((ROOT/'cohort/manifest.json').read_text())['labels']
    configs=[{'id':f'tfidf_df{df}_C{c}_{weight or "none"}', 'min_df':df,'C':c,'class_weight':weight}
             for df in [1,2] for c in [1.0,4.0] for weight in [None,'balanced']]
    cfg={'seed':20261008,'inputs':['Consumer complaint narrative'],
        'data_access':['cohort/train.csv','cohort/validation.csv','cohort/manifest.json'],
        'training_period':'November 2024','validation_period':'December 2024',
        'test_accessed':False,'labels':labels,'lexical_cues':LEXICAL_CUES,
        'lexical_rule':'distinct cue count per label; ties/no cue resolved by training label frequency, then fixed label order',
        'tokenizer':"Unicode word tokens incl contractions; numeric tokens incl comma/decimal groups; individual non-word punctuation tokens",
        'tfidf_fixed':{'ngram_range':[1,2],'lowercase':True,'stop_words':None,'strip_accents':None,
            'sublinear_tf':True,'norm':'l2','smooth_idf':True,'use_idf':True,'max_df':1.0,'max_features':None},
        'lr_fixed':{'solver':'lbfgs','penalty':'l2','max_iter':1000,'tol':1e-4,'fit_intercept':True,
            'objective':'multinomial softmax (lbfgs, 11 classes)','threads':1},
        'experiments':configs,'selection':'maximum validation macro-F1; exact ties use listed experiment order'}
    (OUT/'experiment_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    train_path=ROOT/'cohort/train.csv'; val_path=ROOT/'cohort/validation.csv'
    input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [train_path,val_path,ROOT/'cohort/manifest.json']}
    train=pd.read_csv(train_path,dtype=str,keep_default_na=False)
    validation=pd.read_csv(val_path,dtype=str,keep_default_na=False)
    assert len(train)==2424 and len(validation)==2695
    assert set(train.Issue)==set(validation.Issue)==set(labels)
    assert set(train['Complaint ID']).isdisjoint(validation['Complaint ID'])
    texts=train['Consumer complaint narrative']; vtexts=validation['Consumer complaint narrative']
    y=train.Issue; vy=validation.Issue; ids=validation['Complaint ID']
    counts=y.value_counts().to_dict()
    majority=sorted(labels,key=lambda label:(-counts[label],labels.index(label)))[0]
    results=[]
    start=time.perf_counter(); pred=[majority]*len(validation)
    elapsed=time.perf_counter()-start
    result=write_eval('majority',vy,pred,labels,ids)
    result.update({'experiment':'majority','fit_seconds':0.0,'validation_predict_seconds':elapsed,
        'feature_dim':0,'majority_label':majority,'converged':True})
    results.append(result)
    start=time.perf_counter(); pred,evidence=lexical_predict(vtexts,labels,counts)
    elapsed=time.perf_counter()-start
    result=write_eval('lexical',vy,pred,labels,ids)
    result.update({'experiment':'lexical','fit_seconds':0.0,'validation_predict_seconds':elapsed,
        'feature_dim':sum(len(v) for v in LEXICAL_CUES.values()),'converged':True,
        'fallback_count':sum(e['majority_fallback'] for e in evidence)})
    results.append(result)
    pd.DataFrame(evidence).assign(complaint_id=ids.values).to_csv(OUT/'lexical_validation_evidence.csv',index=False)
    best=None; feature_rows=[]
    with threadpool_limits(limits=1):
        for min_df in [1,2]:
            vectorizer=TfidfVectorizer(tokenizer=tokenize,token_pattern=None,ngram_range=(1,2),
                lowercase=True,stop_words=None,strip_accents=None,min_df=min_df,max_df=1.0,
                max_features=None,sublinear_tf=True,norm='l2',smooth_idf=True,dtype=np.float64)
            start=time.perf_counter(); X=vectorizer.fit_transform(texts)
            vector_fit=time.perf_counter()-start
            vocabulary_before=dict(vectorizer.vocabulary_); idf_before=vectorizer.idf_.copy()
            start=time.perf_counter(); V=vectorizer.transform(vtexts)
            vector_validation=time.perf_counter()-start
            assert vectorizer.vocabulary_==vocabulary_before and np.array_equal(vectorizer.idf_,idf_before)
            for setting in [c for c in configs if c['min_df']==min_df]:
                name=setting['id']
                model=LogisticRegression(C=setting['C'],class_weight=setting['class_weight'],
                    solver='lbfgs',penalty='l2',max_iter=1000,tol=1e-4,random_state=cfg['seed'])
                start=time.perf_counter()
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always'); model.fit(X,y)
                fit=time.perf_counter()-start
                start=time.perf_counter(); pred=model.predict(V)
                predict=time.perf_counter()-start
                result=write_eval(name,vy,pred,labels,ids)
                result.update({'experiment':name,**setting,'feature_dim':X.shape[1],
                    'tfidf_fit_seconds':vector_fit,'tfidf_validation_transform_seconds':vector_validation,
                    'fit_seconds':fit,'validation_predict_seconds':predict,
                    'iterations':int(model.n_iter_.max()),'converged':not any(issubclass(w.category,ConvergenceWarning) for w in ws),
                    'warnings':[str(w.message) for w in ws],
                    'train_matrix_nnz':X.nnz,'validation_matrix_nnz':V.nnz})
                results.append(result)
                pipeline=Pipeline([('tfidf',vectorizer),('logistic_regression',model)])
                joblib.dump(pipeline,OUT/f'{name}.joblib',compress=3)
                if best is None or result['macro_f1']>best['macro_f1']:
                    best=dict(result); best_pipeline=pipeline
                names=vectorizer.get_feature_names_out()
                dfs=np.asarray((X>0).sum(axis=0)).ravel()
                for class_index,label in enumerate(model.classes_):
                    coefs=model.coef_[class_index]
                    ranked=sorted(np.flatnonzero(coefs>0),key=lambda j:(-coefs[j],names[j]))[:25]
                    for rank,j in enumerate(ranked,1):
                        in_class=np.asarray((X[y.to_numpy()==label,j]>0).sum()).item()
                        feature_rows.append({'experiment':name,'Issue':label,'rank':rank,
                            'feature':names[j],'coefficient':float(coefs[j]),
                            'train_document_frequency':int(dfs[j]),'train_class_document_frequency':int(in_class)})
                print(name,'macro_F1',round(result['macro_f1'],4),'dim',X.shape[1], 'fit_s',round(fit,2),flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!='warnings'} for r in results]).to_csv(OUT/'validation_results.csv',index=False)
    (OUT/'validation_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    feature_frame=pd.DataFrame(feature_rows)
    feature_frame.to_csv(OUT/'positive_features_all_models.csv',index=False)
    selected_features=feature_frame[feature_frame.experiment==best['experiment']]
    selected_features.to_csv(OUT/'selected_positive_features.csv',index=False)
    joblib.dump(best_pipeline,OUT/'selected_pipeline.joblib',compress=3)
    selected_pred=best_pipeline.predict(vtexts)
    # Reload validates persistence without adding any fit or data source.
    assert np.array_equal(joblib.load(OUT/'selected_pipeline.joblib').predict(vtexts),selected_pred)
    selected_cm=confusion_matrix(vy,selected_pred,labels=labels)
    plot_confusion(selected_cm,labels,best['experiment'])
    runtime={'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','pandas','joblib','matplotlib','threadpoolctl']},
        'threadpools':threadpool_info(),'input_sha256':input_hashes,
        'source_sha256':{p.name:sha(p) for p in [ROOT/'run_validation_benchmark.py',ROOT/'benchmark_text.py']},
        'elapsed_seconds':time.perf_counter()-total_start,'test_accessed':False,
        'selection':best,'train_label_counts':counts,'validation_label_counts':vy.value_counts().to_dict(),
        'checks':{'validation_does_not_change_vocabulary_or_idf':True,'no_overlapping_dev_ids':True,
                  'pipeline_reload_predictions_identical':True,'all_classes_present':True}}
    assert input_hashes=={str(p.relative_to(ROOT)):sha(p) for p in [train_path,val_path,ROOT/'cohort/manifest.json']}
    runtime['checks']['cohort_files_and_manifest_unchanged']=True
    runtime['all_environment_packages']={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    (OUT/'run_manifest.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    (OUT/'requirements-lock.txt').write_text('\n'.join(f'{p}=={v}' for p,v in sorted(runtime['all_environment_packages'].items()))+'\n',encoding='utf-8')
    build_report(results,best,selected_features,runtime,labels,selected_cm)
    print('Selected',best['experiment'],'Total seconds',round(runtime['elapsed_seconds'],2),flush=True)

def plot_confusion(cm,labels,name):
    short=['Marketing','Closing account','Fees / interest','Getting card','Report information',
        'Other features','Making payments','Investigation','Purchase / statement','Struggling to pay','Using card']
    fig,axes=plt.subplots(1,2,figsize=(19,9),layout='constrained')
    for ax,values,title in zip(axes,[cm,cm/cm.sum(axis=1,keepdims=True)],['Counts','Row-normalized recall']):
        image=ax.imshow(values,cmap='Blues',vmin=0)
        ax.set(xticks=range(11),yticks=range(11),xticklabels=short,yticklabels=short,
               xlabel='Predicted Issue',ylabel='True Issue',title=title)
        plt.setp(ax.get_xticklabels(),rotation=55,ha='right')
        for i in range(11):
            for j in range(11):
                value=values[i,j]; txt=str(int(value)) if title=='Counts' else f'{value:.2f}'
                ax.text(j,i,txt,ha='center',va='center',fontsize=8,
                        color='white' if value>values.max()*.6 else 'black')
        fig.colorbar(image,ax=ax,shrink=.7)
    fig.suptitle('December 2024 validation · '+name)
    fig.savefig(OUT/'validation_confusion_matrix.png',dpi=150)
    plt.close(fig)

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in df.values])

def build_report(results,best,features,runtime,labels,cm):
    frame=pd.DataFrame(results)
    selected=pd.read_csv(OUT/f'{best["experiment"]}_per_label.csv')
    cols=['experiment','macro_f1','weighted_f1','accuracy','balanced_accuracy','feature_dim','fit_seconds']
    summary=frame[cols].copy()
    for c in cols[1:]:
        if c!='feature_dim': summary[c]=summary[c].map(lambda x:f'{x:.4f}')
    detail=selected.copy()
    for c in ['precision','recall','f1']: detail[c]=detail[c].map(lambda x:f'{x:.4f}')
    report=['# First narrative-only validation benchmark',
        'The frozen cohort, labels, duplicate policy, and temporal splits are unchanged. Only cohort/train.csv (2,424 November 2024 records) and cohort/validation.csv (2,695 December 2024 records) were read for experiments. The 2025 test set was not opened, hashed, inspected, or scored. All learned vocabulary, IDF, coefficients, class weights, and fallback frequencies use training only. Models have not been refit on training plus validation.',
        '## Validation results',table(summary),
        f'Selected TF-IDF/logistic-regression configuration: **{best["experiment"]}**, using validation macro-F1. Feature dimensionality: **{best["feature_dim"]:,}**. Selection is among eight explicit configurations, not a random search or cross-validation search. All solver convergence warnings are recorded in validation_results.json.',
        '## Selected model: per-label validation metrics',table(detail),
        '## Validation confusion matrix','![Validation confusion matrix](validation_confusion_matrix.png)',
        'Rows are true labels; columns are predictions. Full-label count matrices and per-label reports are saved for every baseline/configuration. The companion panel normalizes within each true label.',
        '## Methods',
        'Majority predicts the most frequent November training label for every record. The fixed lexical baseline counts distinct phrase cues per class; ties and no-match cases fall back to training label frequencies. Cues were saved before evaluation, contain no company names, and were not revised using validation scores. This is a transparent heuristic, not a semantic representation; it can mishandle negation, multi-issue narratives, paraphrases, and overlapping cues. See experiment_config.json and lexical_validation_evidence.csv.',
        'TF-IDF uses word-level unigrams/bigrams with punctuation emitted as individual tokens, numbers including comma/decimal groups retained, and apostrophes retained within contractions. Lowercasing is the sole fixed text transformation. There is no stop-word list, stemming, lemmatization, truncation, redaction-token removal, or numeric masking. min_df=1 keeps every training token/ngram; min_df=2 excludes singleton-document features as an explicit validated option. No max_df filtering or feature cap is used. Sublinear term frequency, smoothed IDF, and L2 normalization are fixed. Unseen validation features contribute zero; validation never updates vocabulary or IDF.',
        'Logistic regression uses joint multinomial softmax with L2 penalty and lbfgs. The eight settings vary min_df∈{1,2}, C∈{1,4}, and class_weight∈{None,balanced}; balancing uses November frequencies. Maximum iterations=1000, tolerance=1e-4, random seed=20261008. No embeddings or transformers are used. [scikit-learn logistic-regression documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html), [TF-IDF documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).',
        '## Strongest positive lexical features',
        'These are the largest positive class coefficients, not causal effects or per-record contributions. Correlated features and class balancing affect their interpretation. Training document and class-document counts accompany the top 25 per class in selected_positive_features.csv; all eight models are in positive_features_all_models.csv. No suspicious feature has been removed.',
        table(pd.DataFrame([{'Issue':label,'Top 10 positive features':'; '.join(features[features.Issue==label].sort_values('rank').head(10).feature)} for label in labels])),
        '## Runtime and reproducibility',
        f'Wall time for the benchmark including evaluation, serialization, feature export, and plotting: {runtime["elapsed_seconds"]:.2f} seconds. Per-model fit and prediction times, vectorizer fit/transform times, solver iterations, matrix nonzeros, and dimensions are saved in validation_results.json. Shared vectorization times are reported on each configuration and should not be summed repeatedly. One BLAS/OpenMP thread was used. Fit time excludes TF-IDF fitting. Majority/lexical fit time is zero because only training frequency counting is needed; that common setup is included in total runtime.',
        'experiment_config.json saves every setting and phrase cue. run_manifest.json saves source/data hashes, package/runtime versions, class counts, and validation-only access declarations. All fitted pipelines are saved as joblib artifacts; selected_pipeline.joblib remains trained only on November. CSV predictions allow metrics to be recomputed without fitting. requirements-lock.txt pins numerical/runtime dependencies. Re-run with .venv-benchmark/Scripts/python.exe run_validation_benchmark.py. Only load joblib artifacts from trusted sources.',
        'Validation has been used to choose a configuration, so its scores are development estimates. Minority labels have small support; do not interpret small score differences as established gains. Any later feature-removal or cleaning experiment requires review. Test evaluation remains pending approval.']
    (OUT/'README.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')

if __name__=='__main__': main()
