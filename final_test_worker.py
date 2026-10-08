"""Inference only, using saved selected models in their original environments."""
from pathlib import Path
import os, sys, json, time, hashlib, importlib.metadata, platform
R=Path(__file__).resolve().parent; O=R/'temporal_test_evaluation'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HOME']=str(R/'.embedding-cache')
os.environ['MPLCONFIGDIR']=str(O/'mpl-cache')
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    mode=sys.argv[1];assert mode in ['tfidf','frozen','finetuned']
    assert not (O/f'{mode}_test_predictions.csv').exists(), 'Do not repeat predictive test access.'
    pre=json.loads((O/'pre_inference_integrity.json').read_text())
    assert sha(R/'cohort/test.csv')==pre['test_sha256']
    labels=pre['labels']; start=time.perf_counter()
    data=pd.read_csv(R/'cohort/test.csv',dtype=str,keep_default_na=False)
    texts=data['Consumer complaint narrative'].tolist()
    timings={}; details={}
    meta=json.loads((R/'embedding_benchmark/model_download.json').read_text())
    path=meta['local_path']
    if mode=='tfidf':
        t=time.perf_counter();model=joblib.load(R/'benchmark/selected_pipeline.joblib')
        timings['model_load_seconds']=time.perf_counter()-t
        assert model.named_steps['tfidf'].min_df==2
        assert model.named_steps['logistic_regression'].C==1 and model.named_steps['logistic_regression'].class_weight=='balanced'
        with threadpool_limits(limits=1):
            t=time.perf_counter(); features=model[:-1].transform(texts)
            timings['representation_transform_seconds']=time.perf_counter()-t
            t=time.perf_counter();pred=model[-1].predict(features)
            timings['classifier_predict_seconds']=time.perf_counter()-t
        details={'configuration':'tfidf_df2_C1.0_balanced','feature_dimension':features.shape[1],'device':'cpu'}
    elif mode=='frozen':
        import torch
        from sentence_transformers import SentenceTransformer
        from frozen_embedding import encode_complete,state_hash
        torch.manual_seed(20261008);torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        t=time.perf_counter();model=SentenceTransformer(path,device='cpu',local_files_only=True,trust_remote_code=False)
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        classifier=joblib.load(R/'embedding_benchmark/selected_classifier.joblib')
        assert classifier.C==4 and classifier.class_weight=='balanced'
        timings['model_load_seconds']=time.perf_counter()-t
        before=state_hash(model)
        vectors,audit,cost=encode_complete(model,texts,data['Complaint ID'].tolist(),'test',batch_size=32)
        timings.update(cost)
        with threadpool_limits(limits=1):
            t=time.perf_counter();pred=classifier.predict(vectors)
            timings['classifier_predict_seconds']=time.perf_counter()-t
        assert state_hash(model)==before
        pd.DataFrame(audit).to_csv(O/'frozen_test_token_audit.csv',index=False)
        details={'configuration':'embedding_C4.0_balanced','dimension':384,'device':'cpu','threads':4,
            'batch_size':32,'sequence_length':256,'content_chunk_capacity':254,
            'policy':'Complete contiguous nonoverlapping chunks; normalized embeddings; content-count weighted mean; L2 normalization',
            'encoder_state_sha256':before,'weights_unchanged':True}
    else:
        import torch
        from transformers import AutoTokenizer
        from safetensors.torch import load_file
        from torch.utils.data import DataLoader
        from finetune_model import NarrativeClassifier
        torch.manual_seed(20261008);torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available()
        t=time.perf_counter(); tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=NarrativeClassifier(path,11)
        model.load_state_dict(load_file(str(R/'finetune_benchmark/best_model.safetensors')),strict=True)
        model.to('cuda').eval()
        for p in model.parameters():p.requires_grad_(False)
        timings['model_load_seconds']=time.perf_counter()-t
        t=time.perf_counter();items=[];audit=[]
        for complaint,text in zip(data['Complaint ID'],texts):
            full=tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)['input_ids'];kept=full[:510]
            built=tokenizer.build_inputs_with_special_tokens(kept)
            items.append({'input_ids':built,'attention_mask':[1]*len(built),
                          'token_type_ids':tokenizer.create_token_type_ids_from_sequences(kept)})
            audit.append({'Complaint ID':complaint,'original_content_wordpieces':len(full),
                          'retained_content_wordpieces':len(kept),'truncated':len(full)>510})
        timings['tokenization_seconds']=time.perf_counter()-t
        loader=DataLoader(items,batch_size=16,shuffle=False,num_workers=0,
                          collate_fn=lambda x:tokenizer.pad(x,padding=True,return_tensors='pt'))
        pred=[];torch.cuda.synchronize();t=time.perf_counter()
        with torch.inference_mode():
            for i,batch in enumerate(loader):
                pred.extend(model(**{k:v.to('cuda') for k,v in batch.items()}).argmax(1).cpu().tolist())
                if i%100==0:print(f'fine-tuned: inferred {min((i+1)*16,len(data))}/{len(data)}',flush=True)
        torch.cuda.synchronize();timings['joint_encoder_head_inference_seconds']=time.perf_counter()-t
        pred=[labels[i] for i in pred]
        pd.DataFrame(audit).to_csv(O/'finetuned_test_token_audit.csv',index=False)
        details={'checkpoint':'best_epoch_8','sequence_length':512,'content_capacity':510,'batch_size':16,
            'policy':'Right-tail truncation','device':'cuda','gpu':torch.cuda.get_device_name(0),
            'cuda':torch.version.cuda,'dtype':'float32','head':'Existing masked-mean/L2-normalized 11-class head',
            'checkpoint_sha256':sha(R/'finetune_benchmark/best_model.safetensors')}
    assert len(pred)==6521 and set(pred)<=set(labels)
    pd.DataFrame({'Complaint ID':data['Complaint ID'],'true_Issue':data.Issue,'predicted_Issue':pred}).to_csv(
        O/f'{mode}_test_predictions.csv',index=False)
    timings['worker_total_seconds']=time.perf_counter()-start
    record={'mode':mode,'timings':timings,'details':details,'python':sys.version,'platform':platform.platform(),
        'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        'seed':20261008,'source_sha256':sha(Path(__file__)),'training_performed':False}
    (O/f'{mode}_runtime_manifest.json').write_text(json.dumps(record,indent=2))
    print(mode,json.dumps(timings),flush=True)

if __name__=='__main__':main()
