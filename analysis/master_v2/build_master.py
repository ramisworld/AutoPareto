#!/usr/bin/env python3
"""Regenerate master-v2 tables and figures from frozen artifacts only."""
import json, os, statistics as st
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN=os.path.join(ROOT,'paper/generated')
OUT=os.path.join(ROOT,'paper/master_v2/generated')
FIG=os.path.join(ROOT,'paper/master_v2/figures')
os.makedirs(OUT,exist_ok=True); os.makedirs(FIG,exist_ok=True)
read=lambda p: json.load(open(p))
runs=read(os.path.join(GEN,'equal_token_runs.json'))
hist=read(os.path.join(GEN,'historical_experiments.json'))

ORDER=['baseline','autoresearch_compatible','tpe_quality','motpe_quality','autopareto_292','autopareto_299','autopareto_294']
LABELS={'baseline':'Baseline','autoresearch_compatible':'AutoResearch','tpe_quality':'TPE','motpe_quality':'MOTPE',
        'autopareto_292':'AP292','autopareto_299':'AP299','autopareto_294':'AP294'}
COLORS={'baseline':'#65748a','autoresearch_compatible':'#2467a8','tpe_quality':'#75649b','motpe_quality':'#88803b',
        'autopareto_292':'#b84e2f','autopareto_299':'#df9638','autopareto_294':'#9b673e'}

g=defaultdict(list)
for r in runs: g[r['model']].append(r)

# Discovery-stage archive row for each frozen recipe, resolved through its source path.
disc={}
for m in ORDER:
    p=g[m][0]['source']; camp=p.split('/')[2]; num=int(os.path.basename(p).split('-')[0])
    e=next(x for x in hist if x['campaign_id']==camp and x['experiment_number']==num)
    disc[m]=e

def agg(m,key):
    v=[r[key] for r in g[m]]
    return st.mean(v), (st.stdev(v) if len(v)>1 else 0.0)

def ds(m,task,metric):
    v=[next(x['metric_value'] for x in r['downstream'] if x['task_name']==task and x['metric_name']==metric) for r in g[m]]
    return st.mean(v), st.stdev(v)

def table(name,head,rows,align):
    body='\\begin{tabular}{%s}\n\\toprule\n'%align+' & '.join(head)+' \\\\\n\\midrule\n'
    body+=''.join(' & '.join(str(c) for c in r)+' \\\\\n' for r in rows)+'\\bottomrule\n\\end{tabular}\n'
    open(os.path.join(OUT,name),'w').write(body)

# ---- Table: discovery exposure and matched-exposure verification -------------
rows=[]
for m in ORDER:
    e=disc[m]
    params,_=agg(m,'parameters'); bpb,bsd=agg(m,'bpb'); sp,spsd=agg(m,'speed')
    mem,_=agg(m,'memory'); kv,_=agg(m,'incremental_kv_memory'); ss,_=agg(m,'steady_seconds')
    rows.append([LABELS[m], '%.2f'%(params/1e6),
                 '%.1f'%(e['total_training_tokens']/1e6), '%.4f'%e['val_bpb'],
                 '%.4f'%bpb, '%.4f'%bsd, '%.1f'%sp, '%.1f'%mem, '%.3f'%kv, '%.0f'%ss])
table('master_results_table.tex',
      ['Recipe','M par.','Disc.\\ Mtok','Disc.\\ val BPB $\\downarrow$','Held-out BPB $\\downarrow$','SD','Decode $\\uparrow$','MiB $\\downarrow$','KV MiB $\\downarrow$','Steady s'],
      rows,'lrrrrrrrrr')

# ---- Table: full downstream suite -------------------------------------------
KEYS=[('arc_easy','accuracy','ARC-E'),('arc_easy','length_normalized_accuracy','ARC-E$_n$'),
      ('piqa','accuracy','PIQA'),('piqa','length_normalized_accuracy','PIQA$_n$'),
      ('lambada_openai','exact_match','LMB acc'),('lambada_openai','perplexity','LMB PPL'),
      ('blimp','macro_accuracy','BLiMP')]
drows=[]
for m in ORDER:
    cells=[LABELS[m]]
    for task,metric,_ in KEYS:
        mu,sd=ds(m,task,metric)
        if metric=='perplexity': cells.append('%.2f\\,\\tiny$\\pm$%.2f'%(mu,sd))
        else: cells.append('%.2f\\,\\tiny$\\pm$%.2f'%(100*mu,100*sd))
    drows.append(cells)
table('master_downstream_table.tex',
      ['Recipe']+['%s %s'%(lab,'$\\downarrow$' if m=='perplexity' else '$\\uparrow$') for _,m,lab in KEYS],
      drows,'lrrrrrrr')

# ---- Figure: discovery versus verification ----------------------------------
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'axes.labelsize':8.5,'legend.fontsize':7.5})
fig,axes=plt.subplots(1,2,figsize=(6.6,2.85),layout='constrained',sharex=True)
tok=[disc[m]['total_training_tokens']/1e6 for m in ORDER]
tmin,tmax=min(tok),max(tok)
for m in ORDER:
    x,_=agg(m,'speed'); e=disc[m]
    size=45+165*((e['total_training_tokens']/1e6-tmin)/(tmax-tmin))
    head=m in ('autopareto_292','autoresearch_compatible')
    for ax,y,ysd in [(axes[0],e['val_bpb'],None),(axes[1],agg(m,'bpb')[0],agg(m,'bpb')[1])]:
        if ysd is not None: ax.errorbar(x,y,yerr=ysd,fmt='none',ecolor=COLORS[m],capsize=2,lw=1,zorder=2)
        ax.scatter(x,y,s=size,facecolor=COLORS[m],edgecolor='#12202c' if head else 'none',
                   lw=1.4 if head else 0,alpha=.92,zorder=4 if head else 3,label=LABELS[m])
    off={'tpe_quality':(0,-15,'center'),'motpe_quality':(0,10,'center'),'baseline':(0,-13,'center'),
         'autoresearch_compatible':(0,-14,'center'),'autopareto_292':(0,-15,'center'),
         'autopareto_299':(0,-15,'center'),'autopareto_294':(0,-17,'center')}[m]
    axes[0].annotate('%.0fM'%(e['total_training_tokens']/1e6),(x,e['val_bpb']),xytext=off[:2],
                     textcoords='offset points',ha=off[2],fontsize=6.3,color='#4a5560')
for ax,ti,yl in [(axes[0],'A. Fixed-time discovery (unequal exposure)','Discovery validation BPB'),
                 (axes[1],'B. Matched exposure (201.3M tokens, 3 new seeds)','Held-out BPB')]:
    ax.set_title(ti,fontsize=8.2,fontweight='bold',pad=6)
    ax.set_xlabel('Cached-decode tok/s'); ax.set_ylabel(yl); ax.grid(alpha=.15); ax.margins(x=.15,y=.20)
for ax,m,dy in [(axes[0],'autopareto_292',11),(axes[0],'autoresearch_compatible',11),
                (axes[1],'autopareto_292',12),(axes[1],'autoresearch_compatible',12)]:
    x,_=agg(m,'speed'); y=disc[m]['val_bpb'] if ax is axes[0] else agg(m,'bpb')[0]
    ax.annotate(LABELS[m],(x,y),xytext=(0,dy),textcoords='offset points',ha='center',fontsize=7,
                fontweight='bold',color='#12202c')
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,1.14),ncol=7,frameon=False,columnspacing=.7,handletextpad=.2)
fig.savefig(os.path.join(FIG,'discovery_vs_verification.pdf'),bbox_inches='tight',metadata={'Creator':'AutoPareto analysis','Author':''})
fig.savefig(os.path.join(FIG,'discovery_vs_verification.png'),dpi=190,bbox_inches='tight'); plt.close(fig)

json.dump({m:{'disc_tokens':disc[m]['total_training_tokens'],'disc_val_bpb':disc[m]['val_bpb'],
              'disc_exp_id':disc[m]['id'],'heldout_bpb':agg(m,'bpb')[0],'heldout_sd':agg(m,'bpb')[1],
              'speed':agg(m,'speed')[0],'speed_sd':agg(m,'speed')[1],'memory':agg(m,'memory')[0],
              'kv':agg(m,'incremental_kv_memory')[0],'params':agg(m,'parameters')[0],
              'steps':g[m][0]['steps'],'batch':g[m][0]['constants']['TOTAL_BATCH_SIZE'],
              'source':g[m][0]['source'],'sha256':g[m][0]['source_sha256']} for m in ORDER},
          open(os.path.join(OUT,'master_facts.json'),'w'),indent=1)
print('master v2 tables and figure rebuilt')
