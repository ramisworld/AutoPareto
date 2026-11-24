#!/usr/bin/env python3
from common import *
from lineage_labels import load_labels
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
FIG=ROOT/'paper/master_v2/figures' if os.environ.get('MASTER_V2') else ROOT/'paper/figures'
COLORS=['#65748a','#2467a8','#75649b','#88803b','#b84e2f','#df9638','#9b673e']
def save(fig,name,tight=True):
 bb='tight' if tight else None
 fig.savefig(FIG/(name+'.pdf'),bbox_inches=bb,metadata={'Creator':'AutoPareto analysis','Author':''});fig.savefig(FIG/(name+'.png'),dpi=int(os.environ.get('AUTOPARETO_PNG_DPI','190')),bbox_inches=bb);plt.close(fig)
def main():
 FIG.mkdir(exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.labelsize':8,'legend.fontsize':7})
 hist=read(OUT/'historical_experiments.json')
 budget=read(OUT/'budget_analysis.json')['rows'];fig,axes=plt.subplots(2,2,figsize=(5.5,3.1),layout='constrained')
 for arm,color in [('T','#2467a8'),('N','#b84e2f')]:
  rs=sorted([x for x in budget if x['condition']==arm and x['quality_eligible']],key=lambda x:x['attempt'])
  for ax,(key,div,label) in zip(axes.flat,[('parameters',1e6,'Parameters (M)'),('tokens',1e6,'Training tokens (M)'),('bpb',1,'Validation BPB'),('steps',1,'Optimizer updates')]):
   ax.plot([x['attempt'] for x in rs],[x[key]/div for x in rs],'-o',color=color,ms=4,lw=1.3,label={'T':'Fixed time','N':'Fixed tokens'}[arm]);ax.set_xlabel('Attempt within arm');ax.set_ylabel(label);ax.set_xticks(range(1,max(x['attempt'] for x in budget)+1));ax.grid(alpha=.15)
 axes[0,0].legend(frameon=False);save(fig,'budget_trajectory')

 # The complete seed-0 lineage, aligned by proposal order from top to bottom.
 # Each row shows the source edit and its two measured discovery objectives.
 lineage=sorted([x for x in hist if x['campaign_id']=='autopareto-seed0-20260731'],key=lambda x:x['experiment_number'])
 baseline=next(x for x in hist if x['id']==8)
 byid={x['id']:x for x in [baseline]+lineage};children={x['id']:[] for x in [baseline]+lineage}
 for x in lineage:
  if x['parent_experiment_id'] in children:children[x['parent_experiment_id']].append(x['id'])
 leaves=[]
 def collect(node):
  if not children[node]:leaves.append(node)
  else:
   for child in children[node]:collect(child)
 collect(baseline['id'])
 ypos={leaf:i for i,leaf in enumerate(leaves)}
 def place(node):
  if node not in ypos:ypos[node]=sum(place(c) for c in children[node])/len(children[node])
  return ypos[node]
 place(baseline['id'])
 branch_x={node:(pos/(max(1,len(leaves)-1))) for node,pos in ypos.items()}
 # Edit labels are derived from the frozen sources (width rounded to a multiple of HEAD_DIM
 # by each source's own build_model_config); see analysis/lineage_labels.py.  The
 # hand-typed dictionary used for the submitted PDF mislabelled A07-A10 (discrepancy D9).
 edit=load_labels()
 main_path={8,113,116,118,119,120,130,131,132,134}
 sec_path={134,136,137}
 allrows=[baseline]+lineage
 order={baseline['id']:0,**{x['id']:x['experiment_number'] for x in lineage}}
 green='#15935f';amber='#c07a1e';gray='#aab3bc';dark='#263746';bad='#d8dde2'
 fig=plt.figure(figsize=(6.6,4.6),layout='constrained')
 gs=fig.add_gridspec(1,3,width_ratios=[3.1,1.0,1.0],wspace=.05)
 ax=fig.add_subplot(gs[0,0]);axb=fig.add_subplot(gs[0,1],sharey=ax);axs=fig.add_subplot(gs[0,2],sharey=ax)
 for x in lineage:
  p=byid[x['parent_experiment_id']];ispath=x['id'] in main_path and p['id'] in main_path
  issec=(not ispath) and x['id'] in sec_path and p['id'] in sec_path
  ax.annotate('',xy=(branch_x[x['id']],order[x['id']]),xytext=(branch_x[p['id']],order[p['id']]),arrowprops={'arrowstyle':'-|>','mutation_scale':5,'lw':1.25 if ispath else (1.0 if issec else .55),'color':green if ispath else (amber if issec else gray),'linestyle':'-' if ispath else ((0,(3,1.6)) if issec else '-'),'shrinkA':2,'shrinkB':2})
  for metric_ax,key in [(axb,'val_bpb'),(axs,'inference_tokens_per_second')]:
   if x.get(key) is not None and p.get(key) is not None:
    metric_ax.plot([p[key],x[key]],[order[p['id']],order[x['id']]],color=green if ispath else (amber if issec else '#d4d9de'),lw=1.15 if ispath else (.9 if issec else .45),linestyle='-' if ispath else ((0,(3,1.6)) if issec else '-'),zorder=1)
 for x in allrows:
  y=order[x['id']];onpath=x['id'] in main_path;onsec=(not onpath) and x['id'] in sec_path
  failed=x.get('status')!='completed';dom=x.get('decision')=='dominated'
  if failed:
   ax.scatter(branch_x[x['id']],y,marker='x',s=20,color='#9b3a32',lw=1,zorder=4)
  else:
   ax.scatter(branch_x[x['id']],y,s=26,facecolor=green if onpath else ('white' if not dom else bad),edgecolor=green if onpath else (amber if onsec else dark),lw=.9 if onsec else .75,zorder=4)
  label='B  registered baseline' if x['id']==baseline['id'] else f"A{x['experiment_number']:02d}  {edit[x['experiment_number']]}"
  if x['id']==134:label+='   [AP-Q]'
  ax.text(branch_x[x['id']]+.045,y,label,va='center',fontsize=7.0,fontweight='bold' if x['id']==134 else 'normal',color=(green if x['id']==134 else dark) if not failed else '#7f302a')
  for metric_ax,key,fmt in [(axb,'val_bpb',lambda z:f'{z:.4f}'),(axs,'inference_tokens_per_second',lambda z:f'{z:.0f}')]:
   value=x.get(key)
   if value is None:
    metric_ax.text(metric_ax.get_xlim()[0] if metric_ax.has_data() else 0,y,'crash',va='center',fontsize=6.6,color='#9b3a32')
   else:
    metric_ax.scatter(value,y,s=18,facecolor=green if onpath else ('white' if not dom else bad),edgecolor=green if onpath else (amber if onsec else dark),lw=.8 if onsec else .65,zorder=3)
    metric_ax.annotate(fmt(value),(value,y),xytext=(3,0),textcoords='offset points',va='center',fontsize=6.2,color=dark)
 for a,title in [(ax,'Parent-linked source edits'),(axb,'Validation BPB $\\downarrow$'),(axs,'Decode tok/s $\\uparrow$')]:
  a.set_title(title,fontsize=8.4,fontweight='bold',pad=5);a.set_ylim(25.8,-.8);a.grid(axis='x',alpha=.12,lw=.5);a.spines['left'].set_visible(False);a.spines['right'].set_visible(False);a.spines['top'].set_visible(False);a.tick_params(labelsize=7.0,length=2)
 ax.set_xlim(-.05,2.05);ax.set_xticks([]);ax.set_yticks(range(0,26,5));ax.set_yticklabels(['baseline','A05','A10','A15','A20','A25'],fontsize=6.4);ax.tick_params(axis='y',length=0,pad=2)
 axb.set_xlim(1.085,1.225);axb.set_xticks([1.10,1.15,1.20]);axb.tick_params(axis='y',left=False,labelleft=False)
 axs.set_xlim(140,445);axs.set_xticks([200,300,400]);axs.tick_params(axis='y',left=False,labelleft=False)
 # Legend content lives in the manuscript caption for the master build.
 save(fig,'autopareto_lineage',tight=False)
 models=read(OUT/'pareto_analysis.json')['models'];fig,ax=plt.subplots(1,1,figsize=(4.4,2.25),layout='constrained')
 HEAD={'autopareto_292','autoresearch_compatible'}
 for m,color in zip(models,COLORS):
  head=m['model'] in HEAD
  ax.errorbar(m['speed']['mean'],m['bpb']['mean'],xerr=m['speed']['sd'],yerr=m['bpb']['sd'],fmt='o' if m['nondominated'] else 's',color=color,ms=7 if head else 4.5,mew=1.5 if head else 1,markeredgecolor='#12202c' if head else color,capsize=2,zorder=4 if head else 2,label=LABELS[m['model']])
  if head:
   dy,ha_=(10,'left') if m['model']=='autoresearch_compatible' else (-13,'center')
   ax.annotate(LABELS[m['model']],(m['speed']['mean'],m['bpb']['mean']),xytext=(-2 if ha_=='left' else 0,dy),textcoords='offset points',ha=ha_,fontsize=6.6,fontweight='bold',color='#12202c')
 ax.set_xlabel('Cached decode tok/s');ax.set_ylabel('Held-out BPB');ax.grid(alpha=.15);ax.margins(x=.14,y=.16)
 handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.17),ncol=4,frameon=False,columnspacing=.7,fontsize=6.4);save(fig,'pareto_front')
 rows=read(OUT/'equal_token_runs.json');fig,axes=plt.subplots(1,2,figsize=(5.5,1.85),layout='constrained')
 for model,color in zip(ORDER,COLORS):
  r=next(x for x in rows if x['model']==model and x['seed']==3)
  pts=r['milestones'];axes[0].plot([p['token_count']/1e6 for p in pts],[p['validation_bpb'] for p in pts],'-o',ms=3,lw=1,color=color)
  axes[1].scatter(r['parameters']/1e6,r['train_speed']/1e3,s=22,color=color)
 axes[0].set(xlabel='Counted training tokens (M)',ylabel='Validation BPB',title='Frozen recipes, seed 3');axes[1].set(xlabel='Parameters (M)',ylabel='Steady training ktok/s',title='Exposure mechanism diagnostic')
 for ax in axes:ax.grid(alpha=.15)
 save(fig,'mechanism')
 import textwrap
 # Drawn at print size (full text width, 5.5 in) so that box titles print at 7 pt and box text at 6 pt.
 FIGW,FIGH=5.6,1.98
 fig=plt.figure(figsize=(FIGW,FIGH));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,FIGW),ylim=(0,FIGH));ax.axis('off')
 blue='#dbe7f0';green='#dee9e0';orange='#f6e3c6';ink='#25333f';body_col='#3d4952';line='#5c6a75';boundary_col='#8b98a4'
 TS,BS=7.0,6.0
 def wrap_body(text,w):
  chars=max(10,int(w*72/(0.60*BS)))
  return '\n'.join(l for seg in text.split('|') for l in textwrap.wrap(seg,chars))
 def box(x,y,w,h,title,body,color):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.05',facecolor=color,edgecolor=ink,lw=.7,zorder=2))
  ax.text(x+w/2,y+h-.06,title,ha='center',va='top',fontsize=TS,fontweight='bold',color=ink)
  ax.text(x+w/2,y+h-.205,wrap_body(body,w-.06),ha='center',va='top',fontsize=BS,color=body_col,linespacing=1.15)
  return {'top':(x+w/2,y+h),'bottom':(x+w/2,y),'left':(x,y+h/2),'right':(x+w,y+h/2),'x':x,'w':w,'y':y,'h':h}
 def arrow(a,b,color=line,lw=.9,shrinkA=1,shrinkB=1):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','mutation_scale':7,'lw':lw,'color':color,'shrinkA':shrinkA,'shrinkB':shrinkB},zorder=3)
 def elbow(pts,color=line,lw=.9):
  ax.plot([p[0] for p in pts[:-1]],[p[1] for p in pts[:-1]],color=color,lw=lw,zorder=1,solid_capstyle='round',solid_joinstyle='round')
  arrow(pts[-2],pts[-1],color,lw,shrinkA=0,shrinkB=1)
 ax.text(.05,1.90,'REGISTERED STUDY CONTRACT (HUMAN)',fontsize=6.0,fontweight='bold',color='#53616e',va='center')
 b1=box(.05,1.22,1.55,.58,'1  Human specification','objectives, budget, seed, hardware, editable surface, acceptance rules',blue)
 b2=box(1.82,1.22,1.55,.58,'2  Freeze inputs','hash protected files, parent and source; Pair-A: manifest, per-arm histories',blue)
 b3=box(3.62,1.22,1.93,.58,'3  LLM coding agent','hypothesis, parent, patch|gpt-5.6-sol: historical, Codex arm|claude-opus-5-5: Opus arm',green)
 arrow(b1['right'],b2['left']);arrow(b2['right'],b3['left'])
 ax.add_patch(FancyBboxPatch((.02,.04),FIGW-.04,1.01,boxstyle='round,pad=0,rounding_size=.04',facecolor='none',edgecolor=boundary_col,lw=.8,linestyle=(0,(3,2)),zorder=0))
 W,G,X0,Y,H=1.0,.11,.09,.25,.72
 bodies=[('4  Validate','parse source; check parent, source and manifest hashes'),
         ('5  Train (A40)','300 s (Pair-A N arm: N* tokens) after a discarded warm-up'),
         ('6  Evaluate','validation BPB, cache correctness, decode speed, memory'),
         ('7  Record','hypothesis, patch, hashes, metrics; failures kept'),
         ('8  Next view','this campaign\'s (Pair-A: this arm\'s) records and eligible parents')]
 bb=[box(X0+i*(W+G),Y,W,H,t,b,orange if i<4 else blue) for i,(t,b) in enumerate(bodies)]
 for a_,b_ in zip(bb,bb[1:]):arrow(a_['right'],b_['left'])
 xo=b3['x']+.45
 elbow([(xo,b3['y']),(xo,1.13),(bb[0]['top'][0],1.13),bb[0]['top']])
 xi=b3['x']+b3['w']-.30
 elbow([(xi,bb[4]['y']+bb[4]['h']),(xi,b3['y'])])
 ax.text(FIGW/2,.13,'PROTECTED BOUNDARY: FILES HASH-CHECKED EVERY ATTEMPT; PAIR-A ALSO ISOLATES THE AGENT',ha='center',va='center',fontsize=5.6,fontweight='bold',color='#53616e')
 save(fig,'research_loop')
 # Full historical trials, separated from the current paired contract.
 fig,axes=plt.subplots(1,2,figsize=(6.5,2.6),layout='constrained')
 for method,color in zip(['autopareto','autoresearch','random_search','tpe','motpe','nsgaii','cmaes'],COLORS):
  rs=[x for x in hist if x['method']==method and x['valid_result'] and not x['exclude_from_analysis'] and x['total_training_tokens'] and x['parameter_count'] and x['val_bpb']]
  axes[0].scatter([x['parameter_count']/1e6 for x in rs],[x['total_training_tokens']/1e6 for x in rs],s=12,alpha=.6,color=color,label=method)
  axes[1].scatter([x['total_training_tokens']/1e6 for x in rs],[x['val_bpb'] for x in rs],s=12,alpha=.6,color=color)
 axes[0].set(xlabel='Parameters (M)',ylabel='Historical training tokens (M)');axes[1].set(xlabel='Historical training tokens (M)',ylabel='Validation BPB');axes[0].legend(fontsize=6,frameon=False);save(fig,'historical_exposure')
if __name__=='__main__':main()
