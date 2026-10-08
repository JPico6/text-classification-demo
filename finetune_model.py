"""Unmodified pretrained BERT encoder/pooling with a required 11-way task head."""
import torch
from transformers import AutoModel

class NarrativeClassifier(torch.nn.Module):
    def __init__(self,model_path,num_labels=11,dropout=.1):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(model_path,local_files_only=True,
            trust_remote_code=False,attn_implementation='eager',use_safetensors=True)
        self.dropout=torch.nn.Dropout(dropout)
        self.classifier=torch.nn.Linear(self.encoder.config.hidden_size,num_labels)
        torch.nn.init.normal_(self.classifier.weight,std=self.encoder.config.initializer_range)
        torch.nn.init.zeros_(self.classifier.bias)

    def forward(self,input_ids,attention_mask,token_type_ids=None):
        args={'input_ids':input_ids,'attention_mask':attention_mask}
        if token_type_ids is not None:args['token_type_ids']=token_type_ids
        tokens=self.encoder(**args).last_hidden_state
        mask=attention_mask.unsqueeze(-1).to(tokens.dtype)
        mean=(tokens*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
        sentence=torch.nn.functional.normalize(mean,p=2,dim=1)
        return self.classifier(self.dropout(sentence))

def balanced_weights(counts):
    counts=torch.as_tensor(counts,dtype=torch.float32)
    if torch.any(counts<=0):raise ValueError('Missing training label')
    return counts.sum()/(len(counts)*counts)

def weighted_loss_sum(logits,target,weights):
    return (torch.nn.functional.cross_entropy(logits,target,reduction='none')*weights[target]).sum()
