"""Verify saved validation evidence; no test-data access or classifier fitting."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'embedding_benchmark'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import hashlib,json
import numpy as np
import pandas as pd
import joblib,torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,confusion_matrix
from frozen_embedding import encode_complete,state_hash

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    m=json.loads((OUT/'run_manifest.json').read_text()); labels=m['config']['labels']
    assert m['test_accessed'] is False and all(m['checks'].values())
    for path,expected in m['protected_cohort_and_tfidf_sha256'].items():assert sha(ROOT/path)==expected
    validation=pd.read_csv(ROOT/'cohort/validation.csv',dtype=str,keep_default_na=False)
    vectors=np.load(OUT/'validation_embeddings.npy',allow_pickle=False)
    rows=pd.read_csv(OUT/'validation_embedding_rows.csv',dtype=str,keep_default_na=False)
    assert vectors.shape==(2695,384) and np.isfinite(vectors).all()
    assert rows['Complaint ID'].equals(validation['Complaint ID']) and rows.Issue.equals(validation.Issue)
    audit=pd.read_csv(OUT/'narrative_length_audit.csv')
    assert (audit.content_wordpieces==audit.covered_content_wordpieces).all()
    assert (audit.chunks==np.maximum(1,np.ceil(audit.content_wordpieces/254))).all()
    for split,timing in m['encoding'].items():
        subset=audit[audit.split==split]
        assert len(subset)==timing['narratives']
        assert int(subset.exceeds_supported_length.sum())==timing['affected_narratives']
    results=json.loads((OUT/'validation_results.json').read_text())
    assert len(results)==4
    for r in results:
        preds=pd.read_csv(OUT/f'{r["id"]}_validation_predictions.csv',dtype=str,keep_default_na=False)
        assert preds['Complaint ID'].equals(validation['Complaint ID']) and preds.true_Issue.equals(validation.Issue)
        computed={'macro_f1':f1_score(preds.true_Issue,preds.predicted_Issue,labels=labels,average='macro',zero_division=0),
            'weighted_f1':f1_score(preds.true_Issue,preds.predicted_Issue,labels=labels,average='weighted',zero_division=0),
            'accuracy':accuracy_score(preds.true_Issue,preds.predicted_Issue),
            'balanced_accuracy':balanced_accuracy_score(preds.true_Issue,preds.predicted_Issue)}
        assert all(abs(v-r[k])<1e-12 for k,v in computed.items())
        clf=joblib.load(OUT/f'{r["id"]}.joblib')
        assert np.array_equal(clf.predict(vectors),preds.predicted_Issue)
        cm=pd.read_csv(OUT/f'{r["id"]}_confusion.csv',index_col=0)
        assert np.array_equal(cm.values,confusion_matrix(preds.true_Issue,preds.predicted_Issue,labels=labels))
    comparison=pd.read_csv(OUT/'per_label_comparison.csv')
    assert np.allclose(comparison.f1_difference_embedding_minus_tfidf,comparison.f1_embedding-comparison.f1_tfidf)
    assert all(comparison.support_tfidf==comparison.support_embedding)
    # Independent check against the native SentenceTransformer path for a short text.
    torch.set_num_threads(4)
    encoder=SentenceTransformer(m['model_files']['local_path'],local_files_only=True,device='cpu',trust_remote_code=False)
    for p in encoder.parameters():p.requires_grad_(False)
    encoder.eval(); original_state=state_hash(encoder)
    sample=['I cannot pay the interest charge, but I disputed the purchase.']
    native=encoder.encode(sample,normalize_embeddings=True,show_progress_bar=False)
    complete,_,_=encode_complete(encoder,sample,['synthetic-check'],'verification',1)
    assert np.allclose(native,complete,atol=1e-6)
    assert original_state==state_hash(encoder)==m['encoder_state_sha256']
    summary={'all_four_metric_reports_and_confusions_recomputed':True,'persisted_predictions_match':True,
        'length_counts_and_complete_token_coverage_verified':True,'f1_difference_direction_verified':True,
        'short_input_matches_native_sentence_transformer':True,'protected_hashes_unchanged':True,
        'encoder_state_unchanged':True,'test_accessed':False}
    (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
