"""Verify saved final evaluation outputs independently; no predictive inference."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support
R=Path(__file__).resolve().parent;O=R/'temporal_test_evaluation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((O/'test_evaluation_manifest.json').read_text())
    before=json.loads((O/'protected_before_sha256.json').read_text())
    assert all(sha(R/path)==expected for path,expected in before.items())
    assert all(sha(O/path)==expected for path,expected in manifest['evaluation_output_sha256'].items())
    labels=manifest['test_integrity']['labels'];keys=['tfidf','frozen','finetuned'];names=['TF-IDF','Frozen MiniLM','Fine-tuned MiniLM']
    metrics=pd.read_csv(O/'aggregate_metrics.csv');detail=pd.read_csv(O/'per_label_metrics.csv')
    changes=pd.read_csv(O/'validation_test_metric_changes.csv')
    cd=pd.read_csv(O/'validation_test_per_label_f1.csv')
    reference=pd.read_csv(R/'cohort/test_metadata.csv',dtype={'Complaint ID':str})[['Complaint ID','Issue']].rename(columns={'Issue':'true_Issue'})
    for key,name in zip(keys,names):
        pred=pd.read_csv(O/f'{key}_test_predictions.csv',dtype={'Complaint ID':str})
        assert pred[['Complaint ID','true_Issue']].equals(reference)
        a=metrics[(metrics.model==name)&(metrics.split=='test')].iloc[0]
        for metric,value in [('macro_f1',f1_score(pred.true_Issue,pred.predicted_Issue,labels=labels,average='macro',zero_division=0)),
            ('weighted_f1',f1_score(pred.true_Issue,pred.predicted_Issue,labels=labels,average='weighted',zero_division=0)),
            ('accuracy',accuracy_score(pred.true_Issue,pred.predicted_Issue)),
            ('balanced_accuracy',balanced_accuracy_score(pred.true_Issue,pred.predicted_Issue))]:assert abs(a[metric]-value)<1e-12
        cm=confusion_matrix(pred.true_Issue,pred.predicted_Issue,labels=labels)
        assert np.array_equal(cm,pd.read_csv(O/f'{key}_test_confusion.csv',index_col=0).values)
        p,r,f,s=precision_recall_fscore_support(pred.true_Issue,pred.predicted_Issue,labels=labels,zero_division=0)
        d=detail[(detail.model==name)&(detail.split=='test')].set_index('Issue').loc[labels]
        for col,value in [('precision',p),('recall',r),('f1',f),('support',s),('predicted_count',cm.sum(0)),
                          ('predicted_share_percent',100*cm.sum(0)/len(pred))]:assert np.allclose(d[col],value)
        assert d.support.sum()==6521 and d.predicted_count.sum()==6521
        v=metrics[(metrics.model==name)&(metrics.split=='validation')].iloc[0]
        for metric in ['macro_f1','weighted_f1','accuracy','balanced_accuracy']:
            assert abs(changes.set_index('model').loc[name,f'{metric}_change']-(a[metric]-v[metric]))<1e-12
        c=cd[cd.model==name].set_index('Issue').loc[labels]
        dv=detail[(detail.model==name)&(detail.split=='validation')].set_index('Issue').loc[labels]
        assert np.allclose(c.f1_change,d.f1-dv.f1)
    boot=pd.read_csv(O/'paired_bootstrap_replicates.csv');ci=pd.read_csv(O/'macro_f1_bootstrap_intervals.csv')
    assert len(boot)==10000
    for a,b in [(0,1),(0,2),(1,2)]:assert np.allclose(boot[f'{names[a]} minus {names[b]}'],boot[keys[a]]-boot[keys[b]])
    for i,row in ci.iterrows():
        column=keys[i] if i<3 else row.estimand
        assert np.allclose([row.lower_95,row.upper_95],np.quantile(boot[column],[.025,.975]))
    # Reproduce first paired replicate independently with sklearn and the declared seed.
    idx=np.random.default_rng(20261009).integers(0,6521,size=6521)
    for key in keys:
        p=pd.read_csv(O/f'{key}_test_predictions.csv')
        score=f1_score(p.true_Issue.iloc[idx],p.predicted_Issue.iloc[idx],labels=labels,average='macro',zero_division=0)
        assert abs(score-boot[key].iloc[0])<1e-12
    result={'all_test_predictions_align_with_frozen_metadata':True,'aggregate_per_label_metrics_and_confusions_verified':True,
        'metric_and_per_label_differences_verified':True,'paired_bootstrap_differences_intervals_and_first_replicate_verified':True,
        'all_protected_artifacts_unchanged':True,'evaluation_output_hashes_verified':True,
        'protected_files':len(before),'model_loading_or_inference_performed':False,'training_or_adaptation_performed':False,
        'verification_source_sha256':sha(Path(__file__))}
    (O/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
