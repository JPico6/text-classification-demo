"""Independently verify saved December reports/checkpoint; no test access."""
from pathlib import Path
import os,json,hashlib
R=Path(__file__).resolve().parent;O=R/'finetune_benchmark'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support
from finetune_model import NarrativeClassifier,balanced_weights

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    m=json.loads((O/'run_manifest.json').read_text());labels=m['protocol']['labels']
    assert m['test_accessed'] is False and all(m['checks'].values())
    for p,h in m['protected_sha256'].items():assert sha(R/p)==h
    assert sha(O/'best_model.safetensors')==m['checkpoint_sha256']
    val=pd.read_csv(R/'cohort/validation.csv',dtype=str,keep_default_na=False)
    train=pd.read_csv(R/'cohort/train.csv',dtype=str,keep_default_na=False)
    h=pd.read_csv(O/'epoch_history.csv')
    assert int(h.loc[h.macro_f1.idxmax(),'epoch'])==m['selection']['epoch']
    for _,row in h.iterrows():
        pred=pd.read_csv(O/f'epoch_{int(row.epoch)}_validation_predictions.csv',dtype=str,keep_default_na=False)
        assert pred['Complaint ID'].equals(val['Complaint ID']) and pred.true_Issue.equals(val.Issue)
        assert abs(f1_score(pred.true_Issue,pred.predicted_Issue,labels=labels,average='macro',zero_division=0)-row.macro_f1)<1e-12
    selected=pd.read_csv(O/'selected_validation_predictions.csv',dtype=str,keep_default_na=False)
    scores={'macro_f1':f1_score(selected.true_Issue,selected.predicted_Issue,labels=labels,average='macro',zero_division=0),
        'weighted_f1':f1_score(selected.true_Issue,selected.predicted_Issue,labels=labels,average='weighted',zero_division=0),
        'accuracy':accuracy_score(selected.true_Issue,selected.predicted_Issue),
        'balanced_accuracy':balanced_accuracy_score(selected.true_Issue,selected.predicted_Issue)}
    assert all(abs(v-m['selection'][k])<1e-12 for k,v in scores.items())
    pr,rc,f,s=precision_recall_fscore_support(selected.true_Issue,selected.predicted_Issue,labels=labels,zero_division=0)
    detail=pd.read_csv(O/'selected_per_label.csv')
    assert detail.Issue.tolist()==labels and np.array_equal(detail.support,s)
    assert all(np.allclose(detail[c],v) for c,v in [('precision',pr),('recall',rc),('f1',f)])
    cm=pd.read_csv(O/'selected_confusion.csv',index_col=0)
    assert np.array_equal(cm.values,confusion_matrix(selected.true_Issue,selected.predicted_Issue,labels=labels))
    audit=pd.read_csv(O/'narrative_retention_audit.csv')
    assert len(audit)==len(train)+len(val)
    assert np.array_equal(audit.retained_content_wordpieces,np.minimum(audit.original_content_wordpieces,510))
    assert np.array_equal(audit.dropped_content_wordpieces,audit.original_content_wordpieces-audit.retained_content_wordpieces)
    assert np.array_equal(audit.truncated,audit.original_content_wordpieces>510)
    weights=pd.read_csv(O/'loss_weights.csv')
    expected=balanced_weights(train.Issue.value_counts().reindex(labels).to_numpy()).numpy()
    assert weights.Issue.tolist()==labels and np.allclose(weights.loss_weight,expected)
    comp=pd.read_csv(O/'per_label_comparison.csv')
    assert np.allclose(comp.f1_difference_finetuned_minus_tfidf,comp.f1_finetuned-comp.f1_tfidf)
    assert np.allclose(comp.f1_difference_finetuned_minus_frozen,comp.f1_finetuned-comp.f1_embedding)
    # State file itself confirms the encoder shape and 11-class task head.
    state=load_file(str(O/'best_model.safetensors'))
    assert state['encoder.embeddings.position_embeddings.weight'].shape==(512,384)
    assert state['classifier.weight'].shape==(11,384)
    model=NarrativeClassifier(m['pretrained_model']['local_path'],11)
    model.load_state_dict(state,strict=True)
    summary={'all_epoch_macro_f1_recomputed':True,'highest_macro_f1_epoch_selected':True,
        'selected_metrics_per_label_and_confusion_recomputed':True,'loss_weights_use_training_counts':True,
        'right_tail_510_content_token_truncation_verified':True,'comparison_differences_verified':True,
        'checkpoint_hash_and_shapes_verified':True,'strict_checkpoint_reload_passed':True,
        'protected_hashes_unchanged':True,'test_accessed':False}
    (O/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
