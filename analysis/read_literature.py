from common import *
import urllib.request,concurrent.futures
records=read(ROOT/'paper/audit/reference_metadata.json')
def fetch(r):
 try:
  url=(r.get('metadata',{}).get('citation_pdf_url') or [None])[0]
  if not url:return r['key']+' no pdf URL'
  p=ROOT/'paper/audit/literature'/(r['key']+'.pdf')
  if not p.exists():p.write_bytes(urllib.request.urlopen(url.replace('http:','https:'),timeout=30).read())
  return r['key']+' '+str(p.stat().st_size)
 except Exception as e:return r['key']+' '+str(e)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for x in ex.map(fetch,records):print(x)
