#!/usr/bin/env python3
"""Package only manuscript inputs and PDF, never evidence dossiers or raw records."""
from common import *
import re,zipfile

def main():
 paper=ROOT/'paper';source=(paper/'main.tex').read_text()
 names=['main.tex','main.pdf','references.bib','neurips_2026.sty']
 names += [x if Path(x).suffix else x+'.tex' for x in re.findall(r'\\input\{([^}]+)\}',source)]
 names += re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',source)
 names=sorted(set(names));failures=[]
 for name in names:
  p=paper/name
  if p.suffix in ['.tex','.bib','.sty']:
   text=p.read_text()
   for forbidden in ['/Users/','/workspace/','runpod.io','Rami','id_ed25519']:
    if forbidden in text:failures.append({'file':name,'match':forbidden})
 if failures:raise ValueError(failures)
 manifest={name:sha(paper/name) for name in names}
 with zipfile.ZipFile(paper/'working_source_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in names:z.write(paper/name,name)
  z.writestr('README.txt','Working manuscript. Part 2 requires human rewriting and verification. Disclosure boxes are intentionally unchecked. No submission has been performed. Compile main.tex with Tectonic or latexmk. This source bundle contains rendered evidence, not the private raw data or analysis pipeline.\n')
  z.writestr('SHA256.json',json.dumps(manifest,indent=2)+'\n')
 write(paper/'audit/submission_asset_check.json',{'files':manifest,'source_scan_failures':failures,'scope':'PDF text and metadata checked by pdf_qa.py; source bundle excludes internal audit','bundle_sha256':sha(paper/'working_source_bundle.zip')})
 print('Packaged',len(names),'manuscript assets; human obligations remain.')
if __name__=='__main__':main()
