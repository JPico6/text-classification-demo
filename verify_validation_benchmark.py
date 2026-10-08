"""Recompute saved validation metrics and check artifacts; never reads test."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

R=Path(__file__).resolve().parent; O=R/'benchmark'
manifest=json.loads((O/'run_manifest.json').read_text())
cfg=json.loads((O/'experiment_config.json').read_text()); labels=cfg['labels']
validation=pd.read_csv(R/'cohort/validation.csv',dtype=str,keep_default_na=False)
results=json.loads((O/'validation_results.json').read_text())
assert len(results)==10 and manifest['test_accessed'] is False
assert all(r['converged'] for r in results)
for path,expected in manifest['input_sha256'].items():
    assert hashlib.sha256((R/path).read_bytes()).hexdigest()==expected
for r in results:
    p=pd.read_csv(O/f'{r["experiment"]}_validation_predictions.csv',dtype=str,keep_default_na=False)
    assert p['Complaint ID'].equals(validation['Complaint ID'])
    assert p.true_Issue.equals(validation.Issue)
    computed={'macro_f1':f1_score(p.true_Issue,p.predicted_Issue,labels=labels,average='macro',zero_division=0),
        'weighted_f1':f1_score(p.true_Issue,p.predicted_Issue,labels=labels,average='weighted',zero_division=0),
        'accuracy':accuracy_score(p.true_Issue,p.predicted_Issue),
        'balanced_accuracy':balanced_accuracy_score(p.true_Issue,p.predicted_Issue)}
    assert all(abs(value-r[k])<1e-12 for k,value in computed.items())
    cm=pd.read_csv(O/f'{r["experiment"]}_confusion.csv',index_col=0)
    assert np.array_equal(cm.values,confusion_matrix(p.true_Issue,p.predicted_Issue,labels=labels))
    assert list(cm.index)==list(cm.columns)==labels
pipeline=joblib.load(O/'selected_pipeline.joblib')
assert len(pipeline.named_steps['tfidf'].vocabulary_)==manifest['selection']['feature_dim']
selected=pd.read_csv(O/f'{manifest["selection"]["experiment"]}_validation_predictions.csv',dtype=str,keep_default_na=False)
assert np.array_equal(pipeline.predict(validation['Consumer complaint narrative']),selected.predicted_Issue)
print('Verified all 10 validation reports and confusion matrices, frozen input hashes, solver convergence, selected feature dimension, and persisted predictions. No test file read.')
