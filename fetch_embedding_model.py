"""Download only public pretrained artifacts; never read complaint data."""
from pathlib import Path
import os, json, hashlib, time, subprocess
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT/'.embedding-cache')
os.environ['HF_HUB_DISABLE_XET']='1'
import requests

def get(url, **kwargs):
    for attempt in range(3):
        try:
            response=requests.get(url,timeout=120,**kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt==2: raise
            time.sleep(1)

def main():
    repo='sentence-transformers/all-MiniLM-L6-v2'
    target=ROOT/'.embedding-model/all-MiniLM-L6-v2'
    meta=ROOT/'embedding_benchmark/model_download.json'
    meta.parent.mkdir(exist_ok=True)
    # Reuse the recorded immutable revision on later downloads.
    revision=json.loads(meta.read_text())['revision'] if meta.exists() else get(f'https://huggingface.co/api/models/{repo}').json()['sha']
    cached_info=meta.parent/'hub_file_metadata.json'
    if cached_info.exists():
        info=json.loads(cached_info.read_text())
    else:
        info=get(f'https://huggingface.co/api/models/{repo}').json()
        cached_info.write_text(json.dumps(info),encoding='utf-8')
    assert info['sha']==revision, 'Upstream revision changed; retrieve pinned metadata before continuing.'
    target.mkdir(parents=True,exist_ok=True)
    meta.write_text(json.dumps({'repository':repo,'revision':revision,'local_path':str(target)},indent=2))
    for entry in info['siblings']:
        name=entry['rfilename']
        if name not in {'1_Pooling/config.json','config.json','config_sentence_transformers.json',
                        'model.safetensors','modules.json','sentence_bert_config.json',
                        'special_tokens_map.json','tokenizer.json','tokenizer_config.json'}: continue
        path=target/name; path.parent.mkdir(parents=True,exist_ok=True)
        expected=entry.get('lfs',{}).get('sha256')
        if path.exists() and (not expected or hashlib.sha256(path.read_bytes()).hexdigest()==expected): continue
        print('Fetching',name,flush=True)
        temp=path.with_suffix(path.suffix+'.partial')
        url=(f'https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true' if name=='model.safetensors'
             else f'https://huggingface.co/api/resolve-cache/models/{repo}/{revision}/{name}')
        subprocess.run(['curl.exe','--http1.1','--tls-max','1.2','--retry','2','--fail','--location',
            '--silent','--show-error',url,
            '--output',str(temp)],check=True)
        if expected: assert hashlib.sha256(temp.read_bytes()).hexdigest()==expected
        temp.replace(path)
        print('Downloaded',name,flush=True)
    files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(target.rglob('*')) if p.is_file() and '.cache' not in p.parts}
    assert 'model.safetensors' in files
    meta.write_text(json.dumps({'repository':repo,'revision':revision,
        'local_path':str(target),'file_sha256':files},indent=2),encoding='utf-8')
    print(repo,revision,len(files),'files downloaded; no complaint data read.')

if __name__=='__main__': main()
