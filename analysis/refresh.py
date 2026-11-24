#!/usr/bin/env python3
from common import *
import argparse,subprocess,sys,shutil

def main():
 # CAMERA-READY GUARD (2026-11): see paper/camera_ready/BUILD.md.  This legacy refresh rewrites
 # paper/generated/abstract.tex and primary_result.tex (submitted-era wording) and recompiles main.tex.
 import os
 if os.environ.get('AUTOPARETO_ALLOW_LEGACY_REFRESH')!='1':
  raise SystemExit('analysis/refresh.py is disabled for the camera-ready; use paper/build_camera_ready.sh (set AUTOPARETO_ALLOW_LEGACY_REFRESH=1 to override).')
 p=argparse.ArgumentParser();p.add_argument('--cutoff',default=DEFAULT_CUTOFF);p.add_argument('--inventory',action='store_true');p.add_argument('--no-compile',action='store_true');args=p.parse_args()
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 old=read(OUT/'headline_values.json') if (OUT/'headline_values.json').exists() else {}
 oldfile=OUT/'headline_values.previous.json';write(oldfile,old)
 cmd=[sys.executable,str(ROOT/'analysis/build_dataset.py'),'--cutoff',args.cutoff]
 if args.inventory or not (OUT/'artifact_inventory.json').exists():cmd.append('--inventory')
 subprocess.run(cmd,check=True,cwd=ROOT)
 for name in ['analyze_budget_effect','analyze_pareto','analyze_downstream','verify_evidence','generate_tables','generate_reports','generate_figures']:
  subprocess.run([sys.executable,str(ROOT/'analysis'/f'{name}.py')],check=True,cwd=ROOT)
 subprocess.run([sys.executable,'-m','unittest','discover','-s','analysis/tests'],check=True,cwd=ROOT)
 new=read(OUT/'headline_values.json');changes=[f'- {k}: {old.get(k,"not previously built")} -> {v}' for k,v in new.items() if old.get(k)!=v]
 (ROOT/'paper/headline_changes.md').write_text('# Headline changes\n\nBuild '+datetime.now(timezone.utc).isoformat()+'\nCutoff '+args.cutoff+'\n\n'+('\n'.join(changes) if changes else 'No numerical headline changes.')+'\n\nInterpretation is not automatically strengthened when new results arrive. The completed-prefix discussion remains scoped to the rows present at this build. Inspect protocol changes and newly eligible speed evidence before updating conclusions.\n')
 if not args.no_compile:
  if shutil.which('latexmk'):cmd=['latexmk','-pdf','-interaction=nonstopmode','main.tex']
  elif shutil.which('tectonic'):cmd=['tectonic','--keep-logs','--keep-intermediates','main.tex']
  else:raise SystemExit('Install latexmk or Tectonic; no PDF compiler found.')
  with (ROOT/'paper/audit/compile.log').open('w') as log:subprocess.run(cmd,cwd=ROOT/'paper',stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([sys.executable,str(ROOT/'analysis/pdf_qa.py')],check=True,cwd=ROOT)
  subprocess.run([sys.executable,str(ROOT/'analysis/package_paper.py')],check=True,cwd=ROOT)
 archive=ROOT/'paper/audit/refreshes'/stamp;archive.mkdir(parents=True,exist_ok=True)
 for name in ['headline_values.json','validation_report.json','budget_analysis.json','equal_token_runs.json','pdf_qa.json']:
  if (OUT/name).exists():shutil.copy2(OUT/name,archive/name)
 write(archive/'manifest.json',{'built_at_utc':stamp,'cutoff':args.cutoff,'hypothesis_sha256':sha(ROOT/'paper/research_question_freeze.md'),'pdf_sha256':sha(ROOT/'paper/main.pdf') if not args.no_compile else None})
 print('Refreshed paper. Human Part 2 rewrite and visual review are required.')
if __name__=='__main__':main()
