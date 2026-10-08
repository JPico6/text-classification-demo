"""Frozen MiniLM + November-only classifiers; no test-data access."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'embedding_benchmark'
os.environ['HF_HOME']=str(ROOT/'.embedding-cache')
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['MPLCONFIGDIR']=str(OUT/'mpl-cache')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import hashlib,json,time,platform,sys,warnings,importlib.metadata
import numpy as np
import pandas as pd
import torch,joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score,accuracy_score,balanced_accuracy_score,
                             precision_recall_fscore_support,confusion_matrix)
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from frozen_embedding import state_hash,encode_complete

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def evaluate(y,pred,labels):
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    return {'macro_f1':f1_score(y,pred,labels=labels,average='macro',zero_division=0),
            'weighted_f1':f1_score(y,pred,labels=labels,average='weighted',zero_division=0),
            'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred)},pd.DataFrame(
                {'Issue':labels,'precision':p,'recall':r,'f1':f,'support':s})

def main():
    start=time.perf_counter(); OUT.mkdir(exist_ok=True)
    protected=[p for p in sorted((ROOT/'benchmark').rglob('*')) if p.is_file()]
    protected += [ROOT/'cohort/train.csv',ROOT/'cohort/validation.csv',ROOT/'cohort/manifest.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    labels=json.loads((ROOT/'cohort/manifest.json').read_text())['labels']
    model_meta=json.loads((OUT/'model_download.json').read_text())
    model_path=Path(model_meta['local_path'])
    for file,expected in model_meta['file_sha256'].items():assert sha(model_path/file)==expected
    settings=[{'id':f'embedding_C{c}_{w or "none"}','C':c,'class_weight':w}
              for c in [1.0,4.0] for w in [None,'balanced']]
    config={'pretrained_repository':model_meta['repository'],'revision':model_meta['revision'],
        'input':'Consumer complaint narrative','frozen_transformer':True,'device':'cpu','torch_threads':4,
        'batch_size':32,'dtype':'float32','embedding_dimension':384,'sequence_limit_including_special_tokens':256,
        'chunk_content_capacity':254,'overlap_tokens':0,
        'long_text_policy':'Tokenize entire narrative without truncation; contiguous nonoverlapping 254-wordpiece chunks; normalize each frozen chunk embedding; content-token-count weighted mean; L2-normalize narrative vector.',
        'learned_preprocessing':'None. Pretrained tokenizer and encoder are fixed; no scaler/PCA/feature selection.',
        'classifier':{'solver':'lbfgs','penalty':'l2','max_iter':1000,'tol':1e-4,'objective':'multinomial softmax',
                      'random_state':20261008,'threads':1},'experiments':settings,
        'selection':'highest December validation macro-F1; exact tie uses listed order',
        'test_accessed':False,'labels':labels}
    (OUT/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    train=pd.read_csv(ROOT/'cohort/train.csv',dtype=str,keep_default_na=False)
    val=pd.read_csv(ROOT/'cohort/validation.csv',dtype=str,keep_default_na=False)
    assert len(train)==2424 and len(val)==2695 and set(train.Issue)==set(val.Issue)==set(labels)
    assert set(train['Complaint ID']).isdisjoint(val['Complaint ID'])
    tfmanifest=json.loads((ROOT/'benchmark/run_manifest.json').read_text())
    for path,expected in tfmanifest['input_sha256'].items():assert sha(ROOT/path)==expected
    torch.manual_seed(20261008); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    load_start=time.perf_counter()
    encoder=SentenceTransformer(str(model_path),device='cpu',local_files_only=True,trust_remote_code=False)
    encoder.eval()
    for parameter in encoder.parameters():parameter.requires_grad_(False)
    load_seconds=time.perf_counter()-load_start; state_before=state_hash(encoder)
    embeddings={}; lengths=[]; timings={}
    for split,data in [('train',train),('validation',val)]:
        vectors,details,timing=encode_complete(encoder,data['Consumer complaint narrative'].tolist(),
                                               data['Complaint ID'].tolist(),split,config['batch_size'])
        embeddings[split]=vectors; timings[split]=timing; lengths+=details
        np.save(OUT/f'{split}_embeddings.npy',vectors,allow_pickle=False)
        data[['Complaint ID','Issue']].to_csv(OUT/f'{split}_embedding_rows.csv',index=False)
    assert state_before==state_hash(encoder)
    assert all(parameter.grad is None for parameter in encoder.parameters())
    pd.DataFrame(lengths).to_csv(OUT/'narrative_length_audit.csv',index=False)
    results=[]; best=None
    with threadpool_limits(limits=1):
        for setting in settings:
            clf=LogisticRegression(C=setting['C'],class_weight=setting['class_weight'],solver='lbfgs',
                penalty='l2',max_iter=1000,tol=1e-4,random_state=20261008)
            fit_start=time.perf_counter()
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always'); clf.fit(embeddings['train'],train.Issue)
            fit_seconds=time.perf_counter()-fit_start
            predict_start=time.perf_counter(); pred=clf.predict(embeddings['validation'])
            predict_seconds=time.perf_counter()-predict_start
            metrics,detail=evaluate(val.Issue,pred,labels)
            result={**setting,**metrics,'embedding_dimension':384,'fit_seconds':fit_seconds,
                'predict_seconds':predict_seconds,'iterations':int(clf.n_iter_.max()),
                'converged':not any(issubclass(w.category,ConvergenceWarning) for w in ws),
                'warnings':[str(w.message) for w in ws]}
            assert result['converged'],result
            results.append(result)
            name=setting['id']; detail.to_csv(OUT/f'{name}_per_label.csv',index=False)
            pd.DataFrame({'Complaint ID':val['Complaint ID'],'true_Issue':val.Issue,
                          'predicted_Issue':pred}).to_csv(OUT/f'{name}_validation_predictions.csv',index=False)
            pd.DataFrame(confusion_matrix(val.Issue,pred,labels=labels),index=labels,columns=labels).rename_axis(
                'true_Issue').to_csv(OUT/f'{name}_confusion.csv')
            joblib.dump(clf,OUT/f'{name}.joblib',compress=3)
            if best is None or result['macro_f1']>best['macro_f1']:
                best=dict(result); best_clf=clf
            print(name,'macro_F1',round(result['macro_f1'],4),'fit_seconds',round(fit_seconds,3),flush=True)
    joblib.dump(best_clf,OUT/'selected_classifier.joblib',compress=3)
    assert np.array_equal(best_clf.predict(embeddings['validation']),
                          joblib.load(OUT/'selected_classifier.joblib').predict(embeddings['validation']))
    (OUT/'validation_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    pd.DataFrame(results).drop(columns='warnings').to_csv(OUT/'validation_results.csv',index=False)
    tfbest=tfmanifest['selection']; tfname=tfbest['experiment']
    tfdetail=pd.read_csv(ROOT/f'benchmark/{tfname}_per_label.csv')
    edetail=pd.read_csv(OUT/f'{best["id"]}_per_label.csv')
    comparison=tfdetail.merge(edetail,on='Issue',suffixes=('_tfidf','_embedding'),validate='one_to_one')
    assert (comparison.support_tfidf==comparison.support_embedding).all()
    comparison['f1_difference_embedding_minus_tfidf']=comparison.f1_embedding-comparison.f1_tfidf
    comparison.to_csv(OUT/'per_label_comparison.csv',index=False)
    metric_comparison=pd.DataFrame([{'representation':'selected TF-IDF','configuration':tfname,
        **{key:tfbest[key] for key in ['macro_f1','weighted_f1','accuracy','balanced_accuracy']},
        'dimension':tfbest['feature_dim'],'classifier_fit_seconds':tfbest['fit_seconds']},
        {'representation':'frozen MiniLM chunk aggregate','configuration':best['id'],
         **{key:best[key] for key in ['macro_f1','weighted_f1','accuracy','balanced_accuracy']},
         'dimension':384,'classifier_fit_seconds':best['fit_seconds']}])
    metric_comparison.to_csv(OUT/'selected_model_comparison.csv',index=False)
    plot_differences(comparison)
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    manifest={'config':config,'selection':best,'encoder_state_sha256':state_before,
        'encoder_load_seconds':load_seconds,'encoding':timings,
        'total_runtime_seconds':time.perf_counter()-start,'python':sys.version,'platform':platform.platform(),
        'processor':platform.processor(),'packages':environment,'model_files':model_meta,
        'protected_cohort_and_tfidf_sha256':before,
        'source_sha256':{p.name:sha(p) for p in [Path(__file__),ROOT/'frozen_embedding.py',ROOT/'fetch_embedding_model.py']},
        'embedding_sha256':{split:sha(OUT/f'{split}_embeddings.npy') for split in ['train','validation']},
        'checks':{'transformer_state_unchanged':True,'no_transformer_gradients':True,
                  'all_content_wordpieces_covered':True,'no_oversized_model_inputs':True,
                  'protected_cohort_and_tfidf_unchanged':True,'persisted_classifier_predictions_identical':True},
        'test_accessed':False}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'requirements-lock.txt').write_text('\n'.join(f'{p}=={v}' for p,v in sorted(environment.items()))+'\n',encoding='utf-8')
    report(metric_comparison,comparison,results,manifest)
    print('Selected',best['id'],'Encoding',json.dumps(timings),'Total seconds',manifest['total_runtime_seconds'],flush=True)

def plot_differences(df):
    short=['Marketing','Closing account','Fees / interest','Getting card','Report information','Other features',
           'Making payments','Investigation','Purchase / statement','Struggling to pay','Using card']
    values=df.f1_difference_embedding_minus_tfidf.to_numpy()
    fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
    ax.barh(short,values,color=['#167b63' if v>=0 else '#bd4747' for v in values])
    ax.axvline(0,color='black',linewidth=.8); ax.invert_yaxis()
    ax.set(xlabel='Validation F1 difference: embedding minus TF-IDF',title='December 2024 · frozen MiniLM versus selected TF-IDF')
    for i,v in enumerate(values):ax.annotate(f'{v:+.3f}',(v,i),xytext=(5 if v>=0 else -5,0),
            textcoords='offset points',ha='left' if v>=0 else 'right',va='center')
    ax.margins(x=.2); fig.savefig(OUT/'per_label_f1_differences.png',dpi=150); plt.close(fig)

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in df.values])

def report(metrics,comparison,results,manifest):
    rounded=metrics.copy()
    for c in ['macro_f1','weighted_f1','accuracy','balanced_accuracy','classifier_fit_seconds']:
        rounded[c]=rounded[c].map(lambda v:f'{v:.4f}')
    per= comparison[['Issue','f1_tfidf','f1_embedding','f1_difference_embedding_minus_tfidf','support_embedding']].copy()
    for c in ['f1_tfidf','f1_embedding','f1_difference_embedding_minus_tfidf']:per[c]=per[c].map(lambda v:f'{v:+.4f}' if 'difference' in c else f'{v:.4f}')
    all_metrics=pd.DataFrame(results).drop(columns='warnings')
    timing=pd.DataFrame([{'split':split,**values} for split,values in manifest['encoding'].items()])
    text=['# Frozen sentence-embedding validation benchmark',
        'The approved cohort, 11 labels, duplicate handling, and temporal splits are unchanged. November 2024 is the sole classifier-fitting set; December 2024 chooses among four predeclared classifier configurations. No 2025 test data was opened, hashed, inspected, encoded, or scored. Existing TF-IDF artifacts and environment are unchanged.',
        '## Selected model comparison',table(rounded),'## Per-label F1 comparison',table(per),
        '![Per-label validation F1 differences](per_label_f1_differences.png)',
        'per_label_comparison.csv also contains both models’ precision, recall, F1, and support. Every embedding classifier has its own full per-label report, validation predictions, confusion matrix, and fitted artifact. Comparisons use the same 2,695 validation narratives, with no record removal.',
        '## All four frozen-embedding classifier choices',table(all_metrics),
        '## Encoding and long-narrative handling',table(timing),
        'The pretrained all-MiniLM-L6-v2 model yields 384-dimensional vectors and defaults to a 256-wordpiece input limit. We honor that limit including the two special tokens: each chunk has at most 254 original content wordpieces. Entire narratives are tokenized without truncation, split contiguously with zero overlap, and passed directly as token IDs to the pretrained SentenceTransformer modules. Every content token is covered exactly once; no decode/re-tokenize round trip is used.',
        'Each chunk uses the pretrained pooling/normalization; its vector is L2-normalized. Narrative vectors are the content-token-count-weighted mean of the chunk vectors, then L2-normalized. Even long narratives yield one fixed 384-dimensional vector. There is no learned pooling or validation-selected chunk policy. This preserves token coverage but loses cross-chunk context and mixes potentially distinct issues; it is a document-level extension of a short-paragraph model, not the encoder’s native long-context capability. narrative_length_audit.csv lists token lengths, chunk counts, and coverage by ID for training/validation only.',
        '[Official model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) documents the embedding size and default truncation limit.',
        '## Frozen training and reproducibility',
        f'Pretrained repository: {manifest["config"]["pretrained_repository"]}; immutable revision: {manifest["config"]["revision"]}. Encoder loading took {manifest["encoder_load_seconds"]:.2f}s. Whole benchmark runtime was {manifest["total_runtime_seconds"]:.2f}s, excluding dependency installation/model download. Encoding times include tokenization/chunk preparation, transformer inference, and aggregation; classifier times exclude encoding.',
        'CPU float32 encoder inference uses torch.inference_mode(), eval mode, requires_grad=False, four CPU threads, deterministic algorithms, and seed 20261008. State hashes before/after encoding match; no transformer parameter receives a gradient. No fine-tuning is performed. The model and tokenizer are loaded offline from locally downloaded pinned artifacts; narratives never leave the machine.',
        'Classifier inputs are only the frozen narrative vectors. No standardization, PCA, or additional learned preprocessing is used. Multinomial logistic regression uses lbfgs, L2 penalty, C∈{1,4}, class_weight∈{None,balanced}, max_iter=1000, tol=1e-4, and one BLAS thread. Balanced weights use November label counts only. Highest December macro-F1 selects the configuration, with listed order as an exact-tie fallback. No broad search, encoder comparison, or embedding-policy tuning was performed.',
        'run_manifest.json records runtime/package versions, input/source/model/embedding hashes, frozen-state checks, and unchanged hashes for existing benchmark files. experiment_config.json fixes every choice. Cached train/validation embeddings and row-ID/label CSVs allow classifier-only reproduction; all four classifiers and selected_classifier.joblib are saved. There is no refit on training plus validation.',
        'Use the separate .venv-embedding environment. Rebuild downloads with fetch_embedding_model.py (pins the recorded revision), then run run_embedding_benchmark.py offline. requirements-lock.txt pins the complete installed environment; the torch +cpu wheel uses the PyTorch CPU index. Existing TF-IDF’s environment was not modified.',
        'These are development comparisons after selecting both models on validation; differences are descriptive and not evidence of statistically established or future-year improvement. Small minority-label support limits confidence. Chunk aggregation, domain mismatch, and ambiguity can offset semantic benefits. Any new tuning or fine-tuning requires review. Test evaluation remains pending.']
    (OUT/'README.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__':main()
