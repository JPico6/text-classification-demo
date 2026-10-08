"""Lock integrity evidence, then one inference pass per saved selected model."""
from pathlib import Path
import json,hashlib,subprocess,sys,time
import pandas as pd
R=Path(__file__).resolve().parent;O=R/'temporal_test_evaluation'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not list(O.glob('*_test_predictions.csv')) and not (O/'pre_inference_integrity.json').exists(), 'Do not rerun predictive test access.'
    O.mkdir(exist_ok=True)
    config={'models':['tfidf_df2_C1.0_balanced','embedding_C4.0_balanced','fine-tuned best_epoch_8'],
        'expected_test_rows':6521,'input':'Consumer complaint narrative only',
        'no_fit_or_adaptation':True,'bootstrap':{'seed':20261009,'replicates':10000,
        'unit':'Complaint','sampling':'IID with replacement, N=6521 each replicate, paired indices across all models',
        'interval':'Percentile 2.5th and 97.5th; fixed 11-label macro-F1; zero_division=0',
        'stratified':False},'original_model_environments':True}
    (O/'evaluation_config.json').write_text(json.dumps(config,indent=2))
    protected=[]
    for directory in ['cohort','benchmark','embedding_benchmark','finetune_benchmark','.embedding-model/all-MiniLM-L6-v2']:
        protected.extend(p for p in (R/directory).rglob('*') if p.is_file())
    protected.extend(R/name for name in ['benchmark_text.py','frozen_embedding.py','finetune_model.py',
        'run_validation_benchmark.py','run_embedding_benchmark.py','run_finetune_benchmark.py'])
    hashes={str(p.relative_to(R)):sha(p) for p in sorted(protected)}
    (O/'protected_before_sha256.json').write_text(json.dumps(hashes,indent=2))
    labels=json.loads((R/'cohort/manifest.json').read_text())['labels']
    test=pd.read_csv(R/'cohort/test.csv',dtype=str,keep_default_na=False)
    metadata=pd.read_csv(R/'cohort/test_metadata.csv',dtype=str,keep_default_na=False)
    ledger=pd.read_csv(R/'cohort/disposition_ledger.csv',dtype=str,keep_default_na=False)
    kept=ledger[(ledger.split=='test')&(ledger.disposition=='kept')].set_index('Complaint ID')
    assert len(test)==6521 and len(metadata)==6521 and len(kept)==6521
    assert not test['Complaint ID'].duplicated().any()
    assert test[['Complaint ID','Issue']].equals(metadata[['Complaint ID','Issue']])
    assert set(test['Complaint ID'])==set(kept.index) and set(test.Issue)==set(labels)
    assert metadata['Date received'].str.startswith('2025-').all()
    for row in test.to_dict('records'):
        expected=kept.loc[row['Complaint ID']]
        assert row['Issue']==expected.Issue
        assert hashlib.sha256(row['Consumer complaint narrative'].strip().encode()).hexdigest()==expected.exact_hash
    existing=json.loads((R/'finetune_benchmark/run_manifest.json').read_text())
    assert sha(R/'finetune_benchmark/best_model.safetensors')==existing['checkpoint_sha256']
    assert existing['selection']['epoch']==8
    integrity={'test_sha256':hashes[str(Path('cohort')/'test.csv')],'prior_test_file_sha256_available':False,
        'hash_status':'First recorded test-file hash at final evaluation; not claimed as a pre-existing commitment',
        'labels':labels,'rows':6521,'metadata_id_label_order_matches':True,
        'kept_ledger_membership_and_exact_narrative_hashes_match':True,'receipt_year_2025':True,
        'unique_ids':True,'no_pre_inference_exploration':True,'fine_tuned_checkpoint_matches_development_hash':True,
        'pre_inference_integrity_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (O/'pre_inference_integrity.json').write_text(json.dumps(integrity,indent=2))
    print('Integrity verified; starting final inference, with no fitting.',flush=True)
    for mode,env in [('tfidf','.venv-benchmark'),('frozen','.venv-embedding'),('finetuned','.venv-finetune')]:
        subprocess.run([str(R/env/'Scripts/python.exe'),str(R/'final_test_worker.py'),mode],cwd=R,check=True)
    assert hashes=={str(p.relative_to(R)):sha(p) for p in sorted(protected)}
    (O/'post_inference_integrity.json').write_text(json.dumps({'all_protected_artifacts_unchanged':True,
        'protected_files':len(protected),'inference_completed':True,'retraining_or_adaptation':False},indent=2))
    print('All inference complete; all frozen artifacts unchanged.',flush=True)
if __name__=='__main__':main()
