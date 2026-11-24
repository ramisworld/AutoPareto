#!/usr/bin/env python3
"""Retrieve original metadata; cache source pages. No generated author lists."""
from common import *
import urllib.request,re,html,time
from html.parser import HTMLParser
class Meta(HTMLParser):
 def __init__(self):super().__init__();self.meta={}
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='meta' and a.get('name','').startswith('citation_'):self.meta.setdefault(a['name'],[]).append(a.get('content',''))
def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'AutoPareto bibliography verification (research)'})
 return urllib.request.urlopen(req,timeout=40).read().decode()
SOURCES={
'ferreira': 'https://arxiv.org/abs/2603.24647v5',
'primer':'https://papers.nips.cc/paper_files/paper/2021/hash/2f3c6a4cd8af177f6456e7e51a916ff3-Abstract.html',
'mote':'https://proceedings.neurips.cc/paper_files/paper/2024/hash/b6e118c759c16f2424997bbb6a1ffd61-Abstract-Conference.html',
'einspace':'https://papers.nips.cc/paper_files/paper/2024/hash/0382cb76309820f71c6eacd47b36ce71-Abstract-Conference.html',
'autodiscovery':'https://papers.neurips.cc/paper_files/paper/2025/hash/23b127521af7ca7a42f5cdb7507be4f2-Abstract-Conference.html',
'mlestar':'https://papers.nips.cc/paper_files/paper/2025/hash/a9619dd0f0d54a5cf7734add1dc38cd1-Abstract-Conference.html',
'airesearcher':'https://arxiv.org/abs/2505.18705',
'mledojo':'https://arxiv.org/abs/2505.07782',
'monaq':'https://aclanthology.org/2025.findings-emnlp.918/',
'hyperband':'https://jmlr.org/papers/v18/16-558.html',
'bohb':'https://proceedings.mlr.press/v80/falkner18a.html'}
def main():
 bib=[]; records=[]
 for key,url in SOURCES.items():
  p=ROOT/'paper/audit/literature'/f'{key}.html';p.parent.mkdir(parents=True,exist_ok=True)
  try:
   if not p.exists():p.write_text(get(url))
   parser=Meta();parser.feed(p.read_text());m=parser.meta
   title=m['citation_title'][0];authors=m['citation_author'];year=(m.get('citation_publication_date') or m.get('citation_date'))[0][:4]
   if '/paper/2025/' in url: year='2025'  # conference year; publisher ingestion metadata is April 2026
   venue=(m.get('citation_conference_title') or m.get('citation_journal_title') or ['arXiv preprint'])[0]
   if not authors or not year.isdigit():raise ValueError('incomplete metadata')
   # PDF title includes the Primer prefix omitted by its HTML metadata.
   if key=='primer': title='Primer: Searching for Efficient Transformers for Language Modeling'
   authors=[a.replace('ń',r"{\'n}") for a in authors]
   # Preserve verified author order and conference year.
   bib.append('@misc{'+key+',\n title={{'+title+'}},\n author={'+ ' and '.join(authors)+'},\n year={'+year+'},\n howpublished={'+venue+'},\n url={'+url+'}\n}')
   records.append({'key':key,'url':url,'title':title,'authors':authors,'year':year,'venue':venue,'source_sha256':sha(p),'metadata':m})
   print(key,year,title,authors[:2])
  except Exception as e:print('REFERENCE ERROR',key,str(e));records.append({'key':key,'url':url,'error':str(e)})
 p=ROOT/'paper/audit/literature/autoresearch_README.md';p.write_text(get('https://raw.githubusercontent.com/karpathy/autoresearch/master/README.md'))
 bib.append('@misc{autoresearch, title={{autoresearch}}, author={Andrej Karpathy}, year={2026}, howpublished={Software repository}, url={https://github.com/karpathy/autoresearch}}')
 (ROOT/'paper/references.bib').write_text('\n\n'.join(bib)+'\n');write(ROOT/'paper/audit/reference_metadata.json',records)
if __name__=='__main__':main()
