import json, hashlib
from pathlib import Path
R = Path('/Users/Rami/Documents/dev/autopareto/runs/pair-a-official-20260906-fixed-warmup-v2')
for arm in ['pair-a-fixed-time-seed3','pair-a-fixed-tokens-seed3']:
    print('=====', arm)
    lines = (R/arm/'history.jsonl').read_text().splitlines()
    print('history lines', len(lines))
    for i,l in enumerate(lines,1):
        d = json.loads(l)
        r = d.get('result') or {}
        inf = r.get('inference_result') or {}
        dec = r.get('pareto_decision') or {}
        print(i, 'contract', d.get('contract_id'), 'attempt', d.get('campaign_attempt'), 'status', d.get('status'),
              '| agent', d.get('agent_model_identifier'), d.get('reasoning_setting'), 'seed', d.get('training_seed'),
              '| params', r.get('num_params'), 'tokens', r.get('total_tokens'), 'steps', r.get('optimizer_updates'),
              'secs', r.get('protected_measured_training_seconds'), 'bpb', r.get('validation_bpb'),
              '| exp', r.get('experiment_id'), 'parent', d.get('parent_experiment_id'),
              '| src_before', (d.get('candidate_source_before_sha256') or '')[:10], 'src_after', (d.get('candidate_source_after_sha256') or '')[:10],
              '| inf_status', inf.get('normalized_inference_status', inf.get('inference_status')), 'speed', inf.get('absolute_candidate_tok_s', inf.get('cached_decode_tokens_per_second_median')),
              '| decision', dec.get('decision'), dec.get('pareto_eligible'), (dec.get('reason') or '')[:80],
              '| contract', json.dumps(d.get('contract'))[:300])
        print('   keys:', sorted(d.keys())[:60])
        print('   hypothesis:', (d.get('hypothesis') or '')[:200].replace('\n',' '))
        print('   change:', d.get('change_category'), '|', (d.get('intended_change') or '')[:200].replace('\n',' '))
        print('   manifest_sha', (d.get('manifest_sha256') or '')[:12], 'program_sha', (d.get('program_sha256') or '')[:12], 'start_src', (d.get('starting_source_sha256') or '')[:12], 'runtime', (d.get('study_runtime_sha256') or '')[:12])
