"""Report saved validation comparisons without fitting or encoding anything."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'embedding_benchmark'

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in df.values])

def main():
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    metrics=pd.read_csv(OUT/'selected_model_comparison.csv')
    comparison=pd.read_csv(OUT/'per_label_comparison.csv')
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['test_accessed'] is False
    text=['# Embedding comparison findings',
        'Selected frozen MiniLM uses C=4 and balanced training-derived class weights. Validation macro-F1 is 0.4983 versus 0.5111 for the unchanged selected TF-IDF model (difference −0.0129). TF-IDF remains stronger on the primary metric. Embedding weighted-F1 is 0.6101, accuracy 0.6037, and balanced accuracy 0.5337. Balanced accuracy improves by 0.0220, reflecting a different recall/precision tradeoff rather than an across-the-board gain.',
        'Three of eleven labels improve in F1: hardship (+0.0717), account closure (+0.0094), and investigation (+0.0088). The largest losses are incorrect-report information (−0.0543), marketing (−0.0533), and payments (−0.0501). These are descriptive December differences after model selection, not established semantic or causal effects.',
        'For hardship, recall rises from 0.2286 to 0.5143 while precision falls from 0.3333 to 0.2571 (35 validation examples). For investigation, recall rises from 0.1304 to 0.3261 while precision falls from 0.4000 to 0.1500 (46 examples); F1 therefore gains only slightly. The broad Other label stays weak for both models. No cohort, cleaning, or label decisions were changed.',
        '## Long narratives and runtime']
    timing=[]
    for split,t in manifest['encoding'].items():
        timing.append({'Split':split,'Narratives':t['narratives'],'Chunked':t['affected_narratives'],
            'Chunked share':f'{t["affected_narratives"]/t["narratives"]:.2%}',
            'Chunks':t['chunks'],'Encoding seconds':f'{t["encode_total_seconds"]:.2f}'})
    text += [table(pd.DataFrame(timing)),
        'Each narrative becomes one 384-dimensional vector. The fixed policy uses all original wordpieces in contiguous chunks of at most 254 content tokens plus two special tokens, a content-length-weighted mean of normalized chunk vectors, and final L2 normalization. No silent truncation occurred. The longest training narrative uses 31 chunks; the longest validation narrative uses 16. Coverage is complete at the pretrained tokenizer level, with ordinary pretrained normalization/unknown-token behavior still applying.',
        f'Training encoding took {manifest["encoding"]["train"]["encode_total_seconds"]:.2f}s and validation encoding took {manifest["encoding"]["validation"]["encode_total_seconds"]:.2f}s. Selected classifier fitting took {manifest["selection"]["fit_seconds"]:.3f}s. Encoder load time was {manifest["encoder_load_seconds"]:.2f}s; benchmark runtime was {manifest["total_runtime_seconds"]:.2f}s. Download/setup and separate verification are excluded. Lower classifier cost comes with an encoder inference cost that the lexical benchmark does not incur.',
        '## Per-label metrics for both selected models']
    for suffix,name in [('tfidf','TF-IDF'),('embedding','Frozen embedding')]:
        detail=comparison[['Issue',f'precision_{suffix}',f'recall_{suffix}',f'f1_{suffix}',f'support_{suffix}']].copy()
        detail.columns=['Issue','Precision','Recall','F1','Support']
        for col in ['Precision','Recall','F1']:detail[col]=detail[col].map(lambda v:f'{v:.4f}')
        text += [f'### {name}',table(detail)]
    text += ['All eleven F1 differences and a bar chart are in README.md, per_label_comparison.csv, and per_label_f1_differences.png. Full configurations, revision, hashes, cached embeddings, timings, classifier artifacts, predictions, and confusion matrices are saved in this directory.',
        'Three chunk-policy tests passed. Saved metrics/confusions were independently recomputed for all four classifiers, native SentenceTransformer output matches the custom path on a short synthetic input, pretrained state hashes match, and protected cohort/TF-IDF hashes are unchanged. No transformer fine-tuning and no 2025 test access occurred.']
    (OUT/'comparison_findings.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    readme=OUT/'README.md'; contents=readme.read_text(encoding='utf-8')
    note='\n\n## Interpretation and full per-label metrics\n\nSee [comparison_findings.md](comparison_findings.md) for the selected-model interpretation, affected-narrative percentages, and both models’ full per-label precision/recall/F1/support tables. Verification is saved in verification.json. Rebuild this interpretation using summarize_embedding_comparison.py after verify_embedding_benchmark.py.\n'
    if '## Interpretation and full per-label metrics' not in contents:readme.write_text(contents+note,encoding='utf-8')
    (OUT/'report_manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'test_accessed':False,'models_changed':False},indent=2),encoding='utf-8')

if __name__=='__main__':main()
