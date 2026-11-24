"""CPU-only evidence utilities. Never import candidate programs."""
from pathlib import Path
import ast, hashlib, json, math, statistics
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'paper/generated'
SNAP = ROOT/'results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted'
PHASE = SNAP/'results/phase1'
DB = PHASE/'downstream_200m.db'
CURRENT = ROOT/'runs/pair-a-official-20260906-fixed-warmup-v2'
DEFAULT_CUTOFF = '2026-09-06T23:59:00+12:00'
ORDER = ['baseline','autoresearch_compatible','tpe_quality','motpe_quality','autopareto_292','autopareto_299','autopareto_294']
# Camera-ready names (one name per object; old IDs AP292/AP299/AP294 kept once in the glossary).
LABELS = dict(zip(ORDER,['Baseline','AR-ref','TPE','MOTPE','AP-Q','AP-F','AP-S']))
def read(p): return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
def rel(p): return str(Path(p).relative_to(ROOT))
def parse_time(x):
 if not x:return None
 t=datetime.fromisoformat(str(x).replace('Z','+00:00'))
 return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t

def before_cutoff(x,cutoff):
 t=parse_time(x);return t is not None and t<=parse_time(cutoff)
def summary(xs):
 xs=[x for x in xs if x is not None and math.isfinite(x)]
 return dict(n=len(xs),mean=statistics.mean(xs) if xs else None,sd=statistics.stdev(xs) if len(xs)>1 else None,min=min(xs) if xs else None,max=max(xs) if xs else None)
def safe_constants(source):
 """Evaluate only literal arithmetic, no candidate execution."""
 env={}
 def val(n):
  if isinstance(n,ast.Constant):return n.value
  if isinstance(n,ast.Name):return env[n.id]
  if isinstance(n,ast.BinOp):
   a,b=val(n.left),val(n.right)
   if isinstance(n.op,ast.Pow) and abs(b)<20:return a**b
   if isinstance(n.op,ast.Mult):return a*b
   if isinstance(n.op,ast.Add):return a+b
   if isinstance(n.op,ast.Div):return a/b
   if isinstance(n.op,ast.FloorDiv):return a//b
  if isinstance(n,ast.Tuple):return tuple(val(x) for x in n.elts)
  raise ValueError()
 try: tree=ast.parse(source)
 except (SyntaxError,ValueError):return env
 for n in tree.body:
  if isinstance(n,ast.Assign):
   try:v=val(n.value)
   except (KeyError,ValueError,TypeError,ZeroDivisionError,OverflowError):continue
   for t in n.targets:
    if isinstance(t,ast.Name):env[t.id]=v
 return json.loads(json.dumps({k:v for k,v in env.items() if k.isupper()}))
def tex(s):
 return str(s).translate(str.maketrans({'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$'}))
def dominates(a,b,keys=('bpb','speed','memory')):
 if not a.get('protocol') or a.get('protocol')!=b.get('protocol'):return False
 vals=[a.get(k) for k in keys]+[b.get(k) for k in keys]
 if any(x is None or not math.isfinite(x) for x in vals):return False
 return a[keys[0]]<=b[keys[0]] and a[keys[1]]>=b[keys[1]] and a[keys[2]]<=b[keys[2]] and any(a[k]!=b[k] for k in keys)

def positive_finite(x):
 return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0

def budget_result_checks(d,source_hash,observed,cutoff):
 """A completion label alone is insufficient. Missing counts never equal valid exposure."""
 r=d.get('result') or {};inf=r.get('inference_result') or {};contract=d.get('contract') or {}
 tokens=r.get('total_tokens');steps=r.get('optimizer_updates');seconds=r.get('protected_measured_training_seconds')
 target=contract.get('n_star')
 exposure=(positive_finite(target) and tokens==target) if d.get('contract_id')=='N' else (positive_finite(seconds) and 300<=seconds<=315)
 return {'completed':d.get('status')=='completed','tokens':bool(exposure),
  'finite_outcomes':all(positive_finite(x) for x in [tokens,steps,seconds,r.get('validation_bpb'),r.get('num_params')]),
  'integer_counts':all(isinstance(x,int) and not isinstance(x,bool) for x in [tokens,steps]),
  'source_hash':bool(source_hash and source_hash==d.get('candidate_source_after_sha256')),
  'correctness':inf.get('correctness',{}).get('passed') is True,
  'cutoff_bound':before_cutoff(observed,cutoff)}

def assert_unique_ids(rows):
 seen=set()
 for row in rows:
  if row['id'] in seen:raise ValueError('Duplicate experiment identity: '+row['id'])
  seen.add(row['id'])
