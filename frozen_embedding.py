"""Complete token coverage with fixed chunk aggregation; no training code."""
import hashlib
import time
import numpy as np
import torch

def state_hash(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def partition_tokens(ids,capacity):
    if capacity<1: raise ValueError('Nonpositive chunk capacity')
    chunks=[ids[start:start+capacity] for start in range(0,len(ids),capacity)]
    return chunks or [[]]

def aggregate_chunks(vectors,weights):
    mean=np.average(vectors.astype(np.float64),axis=0,weights=weights)
    norm=np.linalg.norm(mean)
    if not np.isfinite(norm) or norm==0: raise ValueError('Invalid aggregate')
    return (mean/norm).astype(np.float32)

def encode_complete(model,texts,ids,split,batch_size=32):
    start=time.perf_counter(); tokenizer=model.tokenizer
    limit=model.max_seq_length
    special=tokenizer.num_special_tokens_to_add(pair=False)
    capacity=limit-special
    assert limit==256 and special==2
    all_chunks=[]; owners=[]; weights=[]; details=[]
    for i,(text,complaint_id) in enumerate(zip(texts,ids)):
        full=tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)['input_ids']
        chunks=partition_tokens(full,capacity)
        assert [token for chunk in chunks for token in chunk]==full
        details.append({'Complaint ID':complaint_id,'split':split,'content_wordpieces':len(full),
            'chunks':len(chunks),'exceeds_supported_length':len(full)+special>limit,
            'covered_content_wordpieces':sum(map(len,chunks))})
        for chunk in chunks:
            built=tokenizer.build_inputs_with_special_tokens(chunk)
            assert len(built)<=limit
            item={'input_ids':built,'attention_mask':[1]*len(built)}
            if 'token_type_ids' in tokenizer.model_input_names:
                item['token_type_ids']=tokenizer.create_token_type_ids_from_sequences(chunk)
            all_chunks.append(item); owners.append(i); weights.append(max(len(chunk),1))
    tokenize_seconds=time.perf_counter()-start
    outputs=[]; inference_start=time.perf_counter()
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    with torch.inference_mode():
        for offset in range(0,len(all_chunks),batch_size):
            features=tokenizer.pad(all_chunks[offset:offset+batch_size],padding=True,return_tensors='pt')
            assert features['input_ids'].shape[1]<=limit
            encoded=model(dict(features))['sentence_embedding'].cpu().numpy()
            encoded=encoded/np.linalg.norm(encoded,axis=1,keepdims=True)
            outputs.append(encoded)
            if offset%(batch_size*20)==0:
                print(f'{split}: encoded {min(offset+batch_size,len(all_chunks))}/{len(all_chunks)} chunks',flush=True)
    chunk_vectors=np.concatenate(outputs)
    inference_seconds=time.perf_counter()-inference_start
    combine_start=time.perf_counter()
    embeddings=[]; offset=0
    for d in details:
        count=d['chunks']
        embeddings.append(aggregate_chunks(chunk_vectors[offset:offset+count],weights[offset:offset+count]))
        offset+=count
    embeddings=np.stack(embeddings)
    assert embeddings.shape==(len(texts),384) and np.isfinite(embeddings).all()
    assert np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-6)
    timing={'tokenize_and_chunk_seconds':tokenize_seconds,'transformer_inference_seconds':inference_seconds,
        'aggregate_seconds':time.perf_counter()-combine_start,'encode_total_seconds':time.perf_counter()-start,
        'narratives':len(texts),'affected_narratives':sum(d['exceeds_supported_length'] for d in details),
        'chunks':len(all_chunks),'content_wordpieces':sum(d['content_wordpieces'] for d in details),
        'max_content_wordpieces':max(d['content_wordpieces'] for d in details),
        'max_chunks_per_narrative':max(d['chunks'] for d in details)}
    return embeddings,details,timing
