"""Inspect exact downloaded checkpoint architecture without complaint data."""
from pathlib import Path
import os,json
R=Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from transformers import AutoTokenizer,AutoModel

def main():
    meta=json.loads((R/'embedding_benchmark/model_download.json').read_text())
    p=Path(meta['local_path'])
    tokenizer=AutoTokenizer.from_pretrained(str(p),local_files_only=True)
    model=AutoModel.from_pretrained(str(p),local_files_only=True,use_safetensors=True,attn_implementation='eager')
    torch.set_num_threads(4);model.eval()
    shape=list(model.embeddings.position_embeddings.weight.shape)
    assert shape==[512,384] and model.config.max_position_embeddings==tokenizer.model_max_length==512
    with torch.inference_mode():
        ids=torch.tensor([[tokenizer.cls_token_id]+[2009]*510+[tokenizer.sep_token_id]])
        output=model(input_ids=ids,attention_mask=torch.ones_like(ids)).last_hidden_state
    assert list(output.shape)==[1,512,384]
    result={'repository':meta['repository'],'revision':meta['revision'],
        'wrapper_default_limit':json.loads((p/'sentence_bert_config.json').read_text())['max_seq_length'],
        'tokenizer_limit':tokenizer.model_max_length,'encoder_position_limit':model.config.max_position_embeddings,
        'actual_position_tensor_shape':shape,'actual_512_forward_output_shape':list(output.shape),
        'positions_resized':False,'complaint_data_accessed':False}
    out=R/'finetune_benchmark';out.mkdir(exist_ok=True)
    (out/'native_length_check_prior_to_training.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
