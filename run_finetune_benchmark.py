"""Single predefined November-only fine-tune. December selects best epoch; no test."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'finetune_benchmark'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['MPLCONFIGDIR']=str(OUT/'mpl-cache')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import hashlib,json,math,time,sys,platform,random,importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
from safetensors.torch import save_file,load_file
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from finetune_model import NarrativeClassifier,balanced_weights,weighted_loss_sum

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def parameter_hash(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def prepare(data,tokenizer,labels,limit,split):
    items=[];audit=[];capacity=limit-tokenizer.num_special_tokens_to_add(pair=False)
    for r in data.to_dict('records'):
        ids=tokenizer(r['Consumer complaint narrative'],add_special_tokens=False,truncation=False,verbose=False)['input_ids']
        kept=ids[:capacity]; built=tokenizer.build_inputs_with_special_tokens(kept)
        assert len(built)<=limit
        item={'input_ids':built,'attention_mask':[1]*len(built),
              'token_type_ids':tokenizer.create_token_type_ids_from_sequences(kept),
              'label':labels.index(r['Issue'])}
        items.append(item)
        audit.append({'Complaint ID':r['Complaint ID'],'split':split,'Issue':r['Issue'],
            'original_content_wordpieces':len(ids),'retained_content_wordpieces':len(kept),
            'dropped_content_wordpieces':len(ids)-len(kept),'retained_fraction':len(kept)/len(ids) if ids else 1.0,
            'truncated':len(ids)>capacity})
    return items,audit

def evaluate(model,loader,device,labels):
    model.eval();pred=[];start=time.perf_counter()
    if device=='cuda':torch.cuda.synchronize()
    with torch.inference_mode():
        for batch,target in loader:
            logits=model(**{k:v.to(device) for k,v in batch.items()})
            pred+=logits.argmax(1).cpu().tolist()
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    return [labels[i] for i in pred],elapsed

def metrics(y,pred,labels):
    return {'macro_f1':f1_score(y,pred,labels=labels,average='macro',zero_division=0),
        'weighted_f1':f1_score(y,pred,labels=labels,average='weighted',zero_division=0),
        'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred)}

def retention_tables(audit):
    result=[]
    for keys,g in audit.groupby(['split','Issue']):
        result.append(retention_row(g,keys[0],keys[1]))
    by_issue=pd.DataFrame(result);by_issue.to_csv(OUT/'retention_by_issue.csv',index=False)
    overall=pd.DataFrame([retention_row(g,s,'ALL') for s,g in audit.groupby('split')])
    overall.to_csv(OUT/'retention_overall.csv',index=False)
    return overall

def retention_row(g,split,issue):
    original=g.original_content_wordpieces.to_numpy();retained=g.retained_content_wordpieces.to_numpy()
    return {'split':split,'Issue':issue,'narratives':len(g),'truncated_count':int(g.truncated.sum()),
        'truncated_percent':float(g.truncated.mean()*100),
        'mean_narrative_retained_percent':float(g.retained_fraction.mean()*100),
        'corpus_tokens_retained_percent':float(retained.sum()/original.sum()*100),
        **{f'retain_at_least_{percent}_percent_narratives':float((retained*100>=original*percent).mean()*100) for percent in [50,75,90,100]},
        **{f'retention_percentile_{q}':float(g.retained_fraction.quantile(q/100)*100) for q in [0,10,25,50,75,90,100]}}

def main():
    run_start=time.perf_counter();OUT.mkdir(exist_ok=True)
    protected=[p for directory in ['benchmark','embedding_benchmark'] for p in sorted((ROOT/directory).rglob('*')) if p.is_file()]
    protected += [ROOT/'cohort/train.csv',ROOT/'cohort/validation.csv',ROOT/'cohort/manifest.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    labels=json.loads((ROOT/'cohort/manifest.json').read_text())['labels']
    meta=json.loads((ROOT/'embedding_benchmark/model_download.json').read_text());path=Path(meta['local_path'])
    for file,expected in meta['file_sha256'].items():assert sha(path/file)==expected
    tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,use_fast=True)
    tokenizer.truncation_side='right';tokenizer.padding_side='right'
    cfg_native=json.loads((path/'config.json').read_text());wrapper=json.loads((path/'sentence_bert_config.json').read_text())
    seed=20261008;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=NarrativeClassifier(str(path),len(labels)).to(device)
    positions=model.encoder.embeddings.position_embeddings.weight
    limit=min(int(tokenizer.model_max_length),model.encoder.config.max_position_embeddings,positions.shape[0])
    assert positions.shape==(512,384) and limit==512 and wrapper['max_seq_length']==256
    initial_encoder_hash=parameter_hash(model.encoder)
    model.eval()
    with torch.inference_mode():
        probe=torch.tensor([[tokenizer.cls_token_id]+[2009]*(limit-2)+[tokenizer.sep_token_id]],device=device)
        native_out=model.encoder(input_ids=probe,attention_mask=torch.ones_like(probe)).last_hidden_state
        assert native_out.shape==(1,512,384)
    native={'checkpoint':meta['repository'],'revision':meta['revision'],
        'sentence_transformer_default_max_seq_length':wrapper['max_seq_length'],
        'tokenizer_model_max_length':tokenizer.model_max_length,
        'config_max_position_embeddings':cfg_native['max_position_embeddings'],
        'loaded_positional_embedding_shape':list(positions.shape),'verified_forward_sequence_length':512,
        'selected_sequence_length':limit,'content_capacity':limit-2,'architecture_resized':False,
        'interpretation':'256 is the sentence-embedding wrapper default, not the 512-position encoder architecture limit. No position resizing or extrapolation.'}
    (OUT/'native_length_verification.json').write_text(json.dumps(native,indent=2),encoding='utf-8')
    config={'seed':seed,'model':meta['repository'],'revision':meta['revision'],'labels':labels,
        'sequence_length':limit,'truncation':'right tail; first 510 original wordpieces + CLS/SEP; no chunking',
        'max_epochs':8,'early_stopping_patience':2,'early_stopping_min_delta':0.0,
        'selection_metric':'December validation macro-F1; strictly higher wins, earliest epoch wins exact ties',
        'batch_size':8,'gradient_accumulation_steps':4,'effective_batch_size':32,'eval_batch_size':16,
        'encoder_learning_rate':2e-5,'head_learning_rate':1e-3,'optimizer':'AdamW','weight_decay':.01,
        'betas':[.9,.999],'epsilon':1e-8,'schedule':'linear decay with 10% warmup over maximum 8-epoch steps',
        'gradient_clip_norm':1.0,'loss':'weighted cross-entropy; per-example weighted NLL summed / accumulation-window sample count',
        'loss_weight_formula':'N_train/(11 * n_train_class)','dtype':'float32; no mixed precision',
        'pooling':'pretrained-style attention-mask mean pooling then L2 normalization',
        'classification_head':'Dropout(0.1) then Linear(384,11), randomly initialized normal std=0.02, zero bias',
        'encoder_architecture':'Original 6-layer BERT/MiniLM and 512 positional rows; no resizing or other encoder architecture changes',
        'trainable':'encoder plus task head; original pooler remains unused as in sentence mean-pooling',
        'attention_implementation':'eager','device':device,'torch_cpu_threads':4,'deterministic_algorithms':True,
        'TF32':False,'test_accessed':False,'protocol_count':1}
    # Written before reading December content or performing training; no search.
    (OUT/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    train=pd.read_csv(ROOT/'cohort/train.csv',dtype=str,keep_default_na=False)
    val=pd.read_csv(ROOT/'cohort/validation.csv',dtype=str,keep_default_na=False)
    assert len(train)==2424 and len(val)==2695 and set(train.Issue)==set(val.Issue)==set(labels)
    assert set(train['Complaint ID']).isdisjoint(val['Complaint ID'])
    counts=train.Issue.value_counts().reindex(labels).to_numpy()
    weights=balanced_weights(counts).to(device)
    pd.DataFrame({'Issue':labels,'train_support':counts,'loss_weight':weights.cpu().numpy()}).to_csv(OUT/'loss_weights.csv',index=False)
    prepare_start=time.perf_counter()
    train_items,train_audit=prepare(train,tokenizer,labels,limit,'train')
    val_items,val_audit=prepare(val,tokenizer,labels,limit,'validation')
    tokenization_seconds=time.perf_counter()-prepare_start
    audit=pd.DataFrame(train_audit+val_audit);audit.to_csv(OUT/'narrative_retention_audit.csv',index=False)
    overall=retention_tables(audit)
    prior=pd.read_csv(ROOT/'embedding_benchmark/narrative_length_audit.csv',dtype={'Complaint ID':str})
    assert audit.set_index('Complaint ID').original_content_wordpieces.sort_index().equals(
        prior.set_index('Complaint ID').content_wordpieces.sort_index())
    def collate(items):
        targets=torch.tensor([item['label'] for item in items],dtype=torch.long)
        inputs=tokenizer.pad([{k:v for k,v in item.items() if k!='label'} for item in items],
                             padding=True,return_tensors='pt')
        assert inputs['input_ids'].shape[1]<=limit
        return dict(inputs),targets
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(train_items,batch_size=8,shuffle=True,generator=gen,collate_fn=collate,num_workers=0)
    validation_loader=DataLoader(val_items,batch_size=16,shuffle=False,collate_fn=collate,num_workers=0)
    no_decay=['bias','LayerNorm.weight']
    groups=[]
    for is_head,lr in [(False,2e-5),(True,1e-3)]:
        for decay in [True,False]:
            params=[p for n,p in model.named_parameters() if n.startswith('classifier.')==is_head
                    and (not any(part in n for part in no_decay))==decay]
            if params:groups.append({'params':params,'lr':lr,'weight_decay':.01 if decay else 0.0})
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)
    steps_per_epoch=math.ceil(len(loader)/4);total_steps=steps_per_epoch*8;warmup=math.ceil(total_steps*.1)
    scheduler=get_linear_schedule_with_warmup(optimizer,warmup,total_steps)
    history=[];best=None;bad_epochs=0;global_step=0;training_start=time.perf_counter()
    for epoch in range(1,9):
        model.train();optimizer.zero_grad(set_to_none=True)
        if device=='cuda':torch.cuda.synchronize()
        epoch_start=time.perf_counter();loss_total=0.0
        for step,(batch,target) in enumerate(loader):
            inputs={k:v.to(device) for k,v in batch.items()};target=target.to(device)
            logits=model(**inputs);loss_sum=weighted_loss_sum(logits,target,weights)
            window_start=(step//4)*4*8;window_n=min(32,len(train)-window_start)
            (loss_sum/window_n).backward();loss_total+=loss_sum.detach().item()
            if (step+1)%4==0 or step+1==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);global_step+=1
            if step%75==0:print(f'Epoch {epoch}: training batch {step+1}/{len(loader)}',flush=True)
        if device=='cuda':torch.cuda.synchronize()
        train_seconds=time.perf_counter()-epoch_start
        predictions,inference_seconds=evaluate(model,validation_loader,device,labels)
        scores=metrics(val.Issue,predictions,labels)
        row={'epoch':epoch,'global_optimizer_step':global_step,'training_weighted_loss':loss_total/len(train),
             'training_seconds':train_seconds,'validation_inference_seconds':inference_seconds,
             'encoder_lr_end':optimizer.param_groups[0]['lr'],**scores}
        history.append(row)
        pd.DataFrame(history).to_csv(OUT/'epoch_history.csv',index=False)
        pd.DataFrame({'Complaint ID':val['Complaint ID'],'true_Issue':val.Issue,'predicted_Issue':predictions}).to_csv(
            OUT/f'epoch_{epoch}_validation_predictions.csv',index=False)
        improved=best is None or row['macro_f1']>best['macro_f1']
        if improved:
            best=dict(row);bad_epochs=0
            save_file({n:t.detach().cpu().contiguous().clone() for n,t in model.state_dict().items()},
                      str(OUT/'best_model.safetensors'),metadata={'epoch':str(epoch),'revision':meta['revision']})
            (OUT/'best_checkpoint.json').write_text(json.dumps(best,indent=2),encoding='utf-8')
        else:bad_epochs+=1
        print('Epoch',epoch,'macro_F1',round(row['macro_f1'],4),'best_epoch',best['epoch'],
              'train_seconds',round(train_seconds,2),flush=True)
        if bad_epochs>=2:break
    fit_and_selection_seconds=time.perf_counter()-training_start
    model.load_state_dict(load_file(str(OUT/'best_model.safetensors'),device=device),strict=True)
    assert model.encoder.embeddings.position_embeddings.weight.shape==(512,384)
    assert initial_encoder_hash!=parameter_hash(model.encoder)
    final_pred,final_inference_seconds=evaluate(model,validation_loader,device,labels)
    assert np.array_equal(final_pred,pd.read_csv(OUT/f'epoch_{best["epoch"]}_validation_predictions.csv').predicted_Issue)
    selected_metrics=metrics(val.Issue,final_pred,labels)
    assert all(abs(selected_metrics[k]-best[k])<1e-12 for k in selected_metrics)
    p,r,f,s=precision_recall_fscore_support(val.Issue,final_pred,labels=labels,zero_division=0)
    detail=pd.DataFrame({'Issue':labels,'precision':p,'recall':r,'f1':f,'support':s})
    detail.to_csv(OUT/'selected_per_label.csv',index=False)
    pd.DataFrame({'Complaint ID':val['Complaint ID'],'true_Issue':val.Issue,'predicted_Issue':final_pred}).to_csv(
        OUT/'selected_validation_predictions.csv',index=False)
    pd.DataFrame(confusion_matrix(val.Issue,final_pred,labels=labels),index=labels,columns=labels).rename_axis(
        'true_Issue').to_csv(OUT/'selected_confusion.csv')
    previous=pd.read_csv(ROOT/'embedding_benchmark/per_label_comparison.csv')
    comparison=previous.merge(detail,on='Issue',validate='one_to_one').rename(columns={
        'precision':'precision_finetuned','recall':'recall_finetuned','f1':'f1_finetuned','support':'support_finetuned'})
    comparison['f1_difference_finetuned_minus_tfidf']=comparison.f1_finetuned-comparison.f1_tfidf
    comparison['f1_difference_finetuned_minus_frozen']=comparison.f1_finetuned-comparison.f1_embedding
    assert (comparison.support_finetuned==comparison.support_tfidf).all()
    comparison.to_csv(OUT/'per_label_comparison.csv',index=False)
    prior_metrics=pd.read_csv(ROOT/'embedding_benchmark/selected_model_comparison.csv')
    all_metrics=pd.concat([prior_metrics,pd.DataFrame([{'representation':'fine-tuned MiniLM, tail truncation',
        'configuration':f'best_epoch_{best["epoch"]}',**selected_metrics,'dimension':384,
        'classifier_fit_seconds':np.nan}])],ignore_index=True)
    all_metrics.to_csv(OUT/'selected_model_comparison.csv',index=False)
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    manifest={'protocol':config,'native_length':native,'selection':best,'epochs_completed':len(history),
        'early_stop_triggered':bad_epochs>=2,'initial_encoder_state_sha256':initial_encoder_hash,
        'selected_encoder_state_sha256':parameter_hash(model.encoder),'checkpoint_sha256':sha(OUT/'best_model.safetensors'),
        'checkpoint_path':str(OUT/'best_model.safetensors'),'tokenization_seconds':tokenization_seconds,
        'sum_epoch_training_seconds':sum(h['training_seconds'] for h in history),
        'sum_epoch_validation_inference_seconds':sum(h['validation_inference_seconds'] for h in history),
        'training_and_selection_wall_seconds':fit_and_selection_seconds,'selected_validation_inference_seconds':final_inference_seconds,
        'total_runtime_seconds':time.perf_counter()-run_start,'scheduler_max_steps':total_steps,'warmup_steps':warmup,
        'packages':environment,'python':sys.version,'platform':platform.platform(),
        'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
        'protected_sha256':hashes,'pretrained_model':meta,
        'source_sha256':{p.name:sha(p) for p in [Path(__file__),ROOT/'finetune_model.py']},
        'checks':{'native_position_shape_unchanged':True,'no_protected_artifact_changes':True,
                  'token_counts_match_frozen_audit':True,'restored_best_checkpoint_predictions_match':True,
                  'encoder_weights_updated_by_training':True},'test_accessed':False}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'requirements-lock.txt').write_text('\n'.join(f'{p}=={v}' for p,v in sorted(environment.items()))+'\n',encoding='utf-8')
    print('Selected epoch',best['epoch'],'Metrics',json.dumps(selected_metrics),'Runtime',manifest['total_runtime_seconds'],flush=True)

if __name__=='__main__':main()
