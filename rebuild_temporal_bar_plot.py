"""Rebuild only the presentation figure from saved metrics and intervals."""
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from report_final_temporal_test import O,NAMES,COLORS,savefig

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    inputs=[O/'aggregate_metrics.csv',O/'macro_f1_bootstrap_intervals.csv']
    before={p.name:sha(p) for p in inputs}
    old={ext:sha(O/f'model_macro_f1_validation_test.{ext}') for ext in ['png','svg','pdf']}
    metrics=pd.read_csv(inputs[0]);ci=pd.read_csv(inputs[1])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(3);width=.34
    val=metrics[metrics.split=='validation'].set_index('model_key').loc[list(NAMES)].macro_f1.to_numpy()
    test=metrics[metrics.split=='test'].set_index('model_key').loc[list(NAMES)].macro_f1.to_numpy()
    ax.bar(x-width/2,val,width,color=COLORS,alpha=.4,label='December 2024 validation')
    ax.bar(x+width/2,test,width,color=COLORS,label='2025 temporal test')
    modelci=ci.iloc[:3];err=np.vstack([test-modelci.lower_95.to_numpy(),modelci.upper_95.to_numpy()-test])
    ax.errorbar(x+width/2,test,yerr=err,fmt='none',ecolor='#222222',capsize=4)
    for j in range(3):
        ax.text(x[j]-width/2,val[j]+.009,f'{val[j]:.3f}',ha='center',fontsize=9)
        ax.text(x[j]+width/2,test[j]+.029,f'{test[j]:.3f}',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=list(NAMES.values()),ylabel='Macro-F1',ylim=(0,max(val.max(),test.max())+.11),
           title='Frozen models on a later complaint cohort')
    ax.legend(loc='upper right',fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    note='Whiskers show 95% bootstrap intervals for 2025 test performance'
    fig.text(.5,.01,note,ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1));savefig(fig,'model_macro_f1_validation_test')
    assert before=={p.name:sha(p) for p in inputs}
    (O/'bar_plot_presentation_revision.json').write_text(json.dumps({'change':'User-requested caption replacement',
        'note':note,'input_sha256':before,'previous_figure_sha256':old,
        'figure_sha256':{ext:sha(O/f'model_macro_f1_validation_test.{ext}') for ext in old},
        'metrics_and_intervals_unchanged':True,'inference_or_bootstrap_performed':False},indent=2))

if __name__=='__main__':main()
