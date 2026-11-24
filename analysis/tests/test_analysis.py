import unittest,sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from analyze_budget_effect import classify
class AnalysisTests(unittest.TestCase):
 def test_dominance_directions_and_ties(self):
  a=dict(protocol='p',bpb=1,speed=100,memory=50);b=dict(protocol='p',bpb=1.1,speed=90,memory=60)
  self.assertTrue(dominates(a,b));self.assertFalse(dominates(b,a));self.assertFalse(dominates(a,a))
 def test_protocols_never_pool(self):
  a=dict(protocol='old',bpb=1,speed=100,memory=50);b=dict(protocol='new',bpb=2,speed=90,memory=60);self.assertFalse(dominates(a,b))
 def test_missing_and_nonfinite_metrics(self):
  a=dict(protocol='p',bpb=1,speed=100,memory=None);b=dict(protocol='p',bpb=2,speed=90,memory=60);self.assertFalse(dominates(a,b));a['memory']=float('nan');self.assertFalse(dominates(a,b))
 def test_cutoff_timezone_boundary(self):
  self.assertTrue(before_cutoff('2026-09-06T11:59:00Z',DEFAULT_CUTOFF));self.assertFalse(before_cutoff('2026-09-06T11:59:01Z',DEFAULT_CUTOFF));self.assertFalse(before_cutoff(None,DEFAULT_CUTOFF))
 def test_no_candidate_execution(self):
  x=safe_constants('import os\nos.system("false")\nDEPTH=8\nTOTAL_BATCH_SIZE=2**17\nBAD=__import__("sys").exit()');self.assertEqual(x['TOTAL_BATCH_SIZE'],131072);self.assertNotIn('BAD',x)
 def test_categories_survive_json_roundtrip(self):
  a=safe_constants('DEPTH=8\nADAM_BETAS=(0.8,0.95)\nTOTAL_BATCH_SIZE=2**17');b=json.loads(json.dumps(a));self.assertNotIn('optimizer',classify({'constants':a},b));b['DEPTH']=10;self.assertIn('capacity/depth/width',classify({'constants':a},b))
 def test_real_data_unique_and_complete(self):
  rows=read(OUT/'equal_token_runs.json');self.assertEqual(len({(x['model'],x['seed']) for x in rows}),len(rows));self.assertEqual(len(rows),21)
  for r in rows:self.assertTrue(r['eligible']);self.assertEqual(r['tokens'],r['steps']*r['constants']['TOTAL_BATCH_SIZE']);self.assertEqual(len(r['downstream']),9)
 def test_namespace_avoids_duplicate_ids(self):
  rows=read(OUT/'budget_attempts.json');ids=[x['id'] for x in rows];self.assertEqual(len(ids),len(set(ids)))
 def test_all_current_lineage_and_uncertain_speeds(self):
  rows=read(OUT/'budget_analysis.json')['rows']
  for r in rows:
   self.assertTrue(r['lineage_verified'])
   if r['speed_status']=='unstable':self.assertFalse(r['pareto_eligible'])
 def test_malformed_registration_retained(self):
  errors=read(OUT/'parse_errors.json');self.assertTrue(any(x['path'].endswith('pair_registration.json') for x in errors))
 def test_corrupt_source_is_not_executed(self):
  self.assertEqual(safe_constants('DEPTH=8\x00'),{})
 def test_duplicate_detection(self):
  with self.assertRaises(ValueError):assert_unique_ids([{'id':'same'},{'id':'same'}])
 def test_incomplete_and_missing_tokens_rejected(self):
  d={'status':'completed','contract_id':'N','contract':{},'result':{}}
  checks=budget_result_checks(d,None,'2026-09-06T06:00:00Z',DEFAULT_CUTOFF)
  self.assertFalse(checks['tokens']);self.assertFalse(checks['finite_outcomes']);self.assertFalse(all(checks.values()))
  d['status']='running';self.assertFalse(budget_result_checks(d,None,None,DEFAULT_CUTOFF)['completed'])
 def test_corrupt_nonfinite_result_rejected(self):
  for value in [None,float('nan'),float('inf'),-1,True,'100']:self.assertFalse(positive_finite(value))
 def test_missing_protocol_not_comparable(self):
  a=dict(bpb=1,speed=100,memory=50);b=dict(bpb=2,speed=90,memory=60);self.assertFalse(dominates(a,b))
if __name__=='__main__':unittest.main()
