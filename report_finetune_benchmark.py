"""Create reviewable reports from saved results; no fitting or test access."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parent;O=R/'finetune_benchmark'
import os
os.environ['MPLCONFIGDIR']=str(O/'mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in df.values])

def main():
    m=json.loads((O/'run_manifest.json').read_text());cfg=m['protocol']
    metrics=pd.read_csv(O/'selected_model_comparison.csv')
    detail=pd.read_csv(O/'selected_per_label.csv')
    comparison=pd.read_csv(O/'per_label_comparison.csv')
    overall=pd.read_csv(O/'retention_overall.csv');by=pd.read_csv(O/'retention_by_issue.csv')
    history=pd.read_csv(O/'epoch_history.csv')
    short=['Marketing','Closing account','Fees / interest','Getting card','Report information','Other features',
        'Making payments','Investigation','Purchase / statement','Struggling to pay','Using card']
    fig,ax=plt.subplots(figsize=(11,8),layout='constrained')
    yy=np.arange(11)
    for offset,col,label,color in [(-.18,'f1_difference_finetuned_minus_tfidf','Fine-tuned minus TF-IDF','#2166ac'),
                                    (.18,'f1_difference_finetuned_minus_frozen','Fine-tuned minus frozen MiniLM','#1b9e77')]:
        ax.barh(yy+offset,comparison[col],height=.34,label=label,color=color)
    ax.axvline(0,color='black',linewidth=.8);ax.set_yticks(yy,short);ax.invert_yaxis();ax.legend()
    ax.set(xlabel='December validation F1 difference',title='Selected fine-tuned checkpoint versus existing benchmarks')
    fig.savefig(O/'per_label_f1_differences.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.plot(history.epoch,history.macro_f1,'o-',label='Validation macro-F1')
    ax.axvline(m['selection']['epoch'],color='gray',linestyle='--',label='Selected checkpoint')
    ax.set(xlabel='Epoch',ylabel='Macro-F1',title='Predefined fine-tuning run: December validation');ax.legend()
    fig.savefig(O/'validation_macro_f1_by_epoch.png',dpi=150);plt.close(fig)
    formatted=metrics[['representation','configuration','macro_f1','weighted_f1','accuracy','balanced_accuracy']].copy()
    for c in ['macro_f1','weighted_f1','accuracy','balanced_accuracy']:formatted[c]=formatted[c].map(lambda v:f'{v:.4f}')
    detail_fmt=detail.copy()
    for c in ['precision','recall','f1']:detail_fmt[c]=detail_fmt[c].map(lambda v:f'{v:.4f}')
    dif=comparison[['Issue','f1_tfidf','f1_embedding','f1_finetuned',
        'f1_difference_finetuned_minus_tfidf','f1_difference_finetuned_minus_frozen','support_finetuned']].copy()
    for c in dif.columns:
        if c.startswith('f1'):dif[c]=dif[c].map(lambda v:f'{v:+.4f}' if 'difference' in c else f'{v:.4f}')
    retained_columns=['split','narratives','truncated_count','truncated_percent','mean_narrative_retained_percent',
        'corpus_tokens_retained_percent']+[f'retain_at_least_{q}_percent_narratives' for q in [50,75,90,100]]
    retention=overall[retained_columns].copy()
    for c in retained_columns[3:]:retention[c]=retention[c].map(lambda v:f'{v:.2f}')
    history_display=history[['epoch','macro_f1','weighted_f1','accuracy','balanced_accuracy','training_weighted_loss',
        'training_seconds','validation_inference_seconds']].copy()
    for c in history_display.columns[1:]:history_display[c]=history_display[c].map(lambda v:f'{v:.4f}')
    text=['# Fine-tuned MiniLM: frozen-cohort validation benchmark',
        'Single predefined protocol. All encoder/task-head updates and loss weights use November training data (2,424 narratives). December validation (2,695 narratives) selects the best epoch and stops training. No 2025 test file was read, hashed, inspected, tokenized, encoded, or scored. No cohort, duplicate, split, label, or validation-driven text-cleaning changes occurred. Earlier benchmark artifacts are protected by unchanged hashes.',
        '## Native input-length verification',
        f'Exact pretrained checkpoint: {cfg["model"]}, revision `{cfg["revision"]}`. The saved SentenceTransformer wrapper default is **256**, tokenizer model_max_length is **512**, encoder max_position_embeddings is **512**, and the loaded position tensor is **512 × 384**. An actual 512-position forward pass succeeded before training (and independently on CPU). The largest native encoder input is therefore **512 total positions**, including CLS/SEP, leaving **510 content wordpieces**. Nothing was resized or extrapolated.',
        'This distinguishes the sentence-embedding wrapper’s default policy from the encoder’s architectural limit. A longer supported input does not establish that the sentence encoder was pretrained effectively on all those lengths. Input handling uses the first 510 original wordpieces with simple right-tail truncation; the frozen comparator continues to use its original full-coverage chunk aggregation.',
        '[Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) documents the wrapper default; local checkpoint configuration/tensor inspection and the successful forward pass establish the encoder limit. Evidence is in native_length_verification.json and native_length_check_prior_to_training.json.',
        '## Selected-checkpoint validation metrics',table(formatted),
        f'Selected epoch: **{m["selection"]["epoch"]}**. Completed epochs: {m["epochs_completed"]}; patience-based early stop triggered: {m["early_stop_triggered"]}. Selection uses macro-F1 only, with the earlier epoch retained on exact ties. best_checkpoint.json and best_model.safetensors identify the selected checkpoint.',
        '## Validation metrics after every epoch',table(history_display),
        '![Epoch selection](validation_macro_f1_by_epoch.png)',
        '## Selected fine-tuned model: per-class validation metrics',table(detail_fmt),
        '## Direct per-class F1 comparisons',table(dif),
        '![Per-label F1 differences](per_label_f1_differences.png)',
        'per_label_comparison.csv includes all three models’ precision, recall, F1, support, and signed F1 differences. Existing comparator settings and scores were imported unchanged. Small-support labels and development-set selection limit interpretation of differences; there is no test-year generalization estimate.',
        '## Token retention: overall',table(retention),
        'Retention is retained content wordpieces / original content wordpieces, excluding special tokens and padding. Threshold columns are percentages of narratives retaining at least 50%, 75%, 90%, or 100%. Mean narrative retention weights each narrative equally; corpus-token retention weights longer narratives more. Narrative-retention quantiles (min/p10/p25/median/p75/p90/max) are in the CSVs. Token counts match the earlier frozen-embedding audit exactly.',
        '## Token retention by Issue']
    for split in ['train','validation']:
        g=by[by.split==split][['Issue','narratives','truncated_count','truncated_percent','mean_narrative_retained_percent']+
                [f'retain_at_least_{q}_percent_narratives' for q in [50,75,90,100]]].copy()
        for c in g.columns[3:]:g[c]=g[c].map(lambda v:f'{v:.2f}')
        text += [f'### {split}',table(g)]
    text += ['All truncation is explicit in narrative_retention_audit.csv and retention_by_issue.csv. No record is removed for length. The classifier cannot see discarded tail tokens; this policy differs from the frozen benchmark’s complete-token chunking, so score differences combine training and input-policy effects.',
        '## Fixed protocol and runtime',
        f'AdamW: encoder LR={cfg["encoder_learning_rate"]}, new task-head LR={cfg["head_learning_rate"]}, weight decay 0.01 except biases/LayerNorm weights; betas=(0.9,0.999), epsilon=1e-8. Eight-epoch maximum, patience=2, batch size=8, accumulation=4 (effective 32 except the final 24-example window), evaluation batch=16. Linear schedule with {m["warmup_steps"]} warmup steps out of {m["scheduler_max_steps"]} maximum optimizer steps. Float32, gradient clipping=1, no mixed precision, seed={cfg["seed"]}. Only one protocol was run; no LR/epoch grid or feature cleaning was selected on validation.',
        'Loss weights are N_train/(11*n_train_class), matching the inverse-frequency rationale of balanced logistic regression. Loss is the weighted per-example negative log likelihood averaged over each accumulation window’s actual number of examples, not renormalized independently in each tiny batch. weights are saved in loss_weights.csv. No class resampling is applied.',
        'The original encoder structure, positional table size, pretrained-style masked mean pooling, and L2 normalization are preserved. A required 11-class linear task head with dropout 0.1 is added; it is initialized with normal std=0.02 and zero bias. All encoder parameters and the task head are fine-tuned, although the original unused BERT pooler receives no task gradient. Position values may learn during fine-tuning but the table is never extended.',
        f'Device: {cfg["device"]}; GPU: {m["gpu"]}; CUDA runtime: {m["cuda_runtime"]}. Deterministic algorithms enabled; TF32 disabled; eager attention; four CPU threads. Summed epoch training time: {m["sum_epoch_training_seconds"]:.2f}s. Summed per-epoch validation inference: {m["sum_epoch_validation_inference_seconds"]:.2f}s. Training-plus-selection wall time: {m["training_and_selection_wall_seconds"]:.2f}s. Selected-checkpoint validation inference: {m["selected_validation_inference_seconds"]:.2f}s (includes dynamic padding, host-to-device transfer, model forward, and prediction collection; excludes tokenization). Tokenization: {m["tokenization_seconds"]:.2f}s. Total benchmark runtime: {m["total_runtime_seconds"]:.2f}s, excluding installation and later verification/reporting.',
        f'Best-checkpoint SHA-256: `{m["checkpoint_sha256"]}`. Initial and selected encoder-state hashes differ, verifying actual fine-tuning; the positional tensor remains 512×384. Reloaded best-checkpoint predictions match its saved epoch predictions.',
        'experiment_config.json is saved before training. epoch_history.csv persists validation macro-F1 after every epoch, with losses, runtime, optimizer step and LR. All epoch validation predictions are retained. Selected per-class metrics and confusion matrix are saved. run_manifest.json and requirements-lock.txt record exact Python/package versions, source hashes, pretrained files/revision, GPU/runtime, input/protected-artifact hashes, and integrity checks. Previous environments were not altered. The separate .venv-finetune environment uses PyTorch CUDA 12.1 wheels. Rebuild with run_finetune_benchmark.py, then report_finetune_benchmark.py. No training-plus-validation refit or test inference was performed.']
    (O/'README.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    (O/'report_manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'test_accessed':False,'model_changed':False},indent=2),encoding='utf-8')

if __name__=='__main__':main()
