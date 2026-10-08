"""Read-only documentation checks; no model loading or narrative-data access."""
from pathlib import Path
import csv,hashlib,json,re,subprocess
R=Path(__file__).resolve().parent
def main():
    docs=[R/'README.md',R/'docs/REPRODUCING.md',R/'docs/PUBLICATION_REVIEW.md']
    checked=0
    for path in docs:
        text=path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https://','http://','#')):continue
            assert (path.parent/target.split('#')[0]).exists(),f'Broken link: {path.name}: {target}'
            checked+=1
    readme=docs[0].read_text(encoding='utf-8')
    with (R/'temporal_test_evaluation/aggregate_metrics.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            assert f"{float(row['macro_f1']):.4f}" in readme
    manifest=json.loads((R/'temporal_test_evaluation/test_evaluation_manifest.json').read_text())
    for name,expected in manifest['evaluation_output_sha256'].items():
        assert hashlib.sha256((R/'temporal_test_evaluation'/name).read_bytes()).hexdigest()==expected,name
    probes=['CCDB_Export_example.csv','cohort/train.csv','cohort/manual_review.html',
        'benchmark/shortcut_training_contexts.csv','example.joblib','example.safetensors','example.npy',
        'temporal_test_evaluation/mpl-cache/example.json','finetune_benchmark/epoch_class_diagnostics/mpl-cache/example.json']
    cmd=['git','-c',f'safe.directory={R.as_posix()}']
    for probe in probes:
        result=subprocess.run(cmd+['check-ignore','--no-index','-q',probe],cwd=R)
        assert result.returncode==0,f'Not ignored: {probe}'
    for probe in ['cohort/counts_overall.csv','cohort/counts_by_issue.csv','cohort/stage_counts.csv',
                  'temporal_test_evaluation/aggregate_metrics.csv']:
        assert subprocess.run(cmd+['check-ignore','--no-index','-q',probe],cwd=R).returncode==1
    tracked=subprocess.check_output(cmd+['ls-files'],cwd=R,text=True).splitlines()
    forbidden=[p for p in tracked if p.endswith(('.joblib','.safetensors','.npy')) or p.startswith('CCDB_Export_')]
    assert not forbidden,forbidden
    review=[p for p in tracked if p in ['cohort/manual_review.html','benchmark/shortcut_training_contexts.csv'] or '/mpl-cache/' in p]
    print(json.dumps({'documentation_files':len(docs),'local_links_checked':checked,
        'readme_macro_f1_matches_saved_results':True,'all_final_evaluation_outputs_unchanged':True,
        'ignore_rules_checked':True,'tracked_raw_exports_or_large_models':forbidden,
        'tracked_files_requiring_publication_cleanup':review},indent=2))
if __name__=='__main__':main()
