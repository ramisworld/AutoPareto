#!/usr/bin/env python3
from common import *
from pypdf import PdfReader
import re

def main():
 p=ROOT/'paper/main.pdf';r=PdfReader(p);texts=[x.extract_text() for x in r.pages];aux=(ROOT/'paper/main.aux').read_text()
 def page(label):
  z=re.search(r'\\newlabel\{'+label+r'\}\{\{.*?\}\{(\d+)\}',aux);return int(z.group(1)) if z else None
 bounds={s:page(s) for s in ['partone','partoneend','parttwo','parttwoend','partthree','partthreeend']}
 counts={'part1':bounds['parttwo']-1, 'part2':bounds['partthree']-bounds['parttwo'],'part3':bounds['partthreeend']-bounds['partthree']+1}
 failures=[]
 if any(counts[k]>v for k,v in [('part1',4),('part2',4),('part3',1)]) or sum(counts.values())>9:failures.append('page limits exceeded')
 s='\n'.join(texts)+'\n'+str(r.metadata); forbidden=['/Users/','/workspace/','Rami','autopareto_trusted','runpod.io','GPU-0188884d']
 leaks=[x for x in forbidden if x.lower() in s.lower()]
 if leaks:failures.append('anonymity strings: '+str(leaks))
 log=(ROOT/'paper/main.log').read_text(errors='replace');bad=[l for l in log.splitlines() if any(t in l for t in ['Overfull \\hbox','Overfull \\vbox','Citation','undefined references','Missing character'])]
 biblog=ROOT/'paper/main.blg'
 if biblog.exists():bad += [line for line in biblog.read_text(errors='replace').splitlines() if 'Warning--' in line or 'error message' in line]
 if bad:failures.append('LaTeX diagnostics require review')
 (ROOT/'paper/generated/pdf_text.txt').write_text(s)
 report={'pdf_sha256':sha(p),'total_pages':len(r.pages),'section_bounds':bounds,'content_page_counts':counts,'content_pages_total':sum(counts.values()),'metadata':{str(k):str(v) for k,v in (r.metadata or {}).items()},'latex_diagnostics':bad,'failures':failures,'automated_pass':not failures,'visual_review':'pending; separate rendered-page inspection required'}
 write(OUT/'pdf_qa.json',report);print(json.dumps(report,indent=2))
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
