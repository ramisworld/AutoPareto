import json, sys, re
def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d: return default
        d = d[p]
    return d
def summarize(path, verbose=False):
    out = []
    for ln, l in enumerate(open(path), 1):
        d = json.loads(l)
        r = d.get('result') or {}
        inf = r.get('inference_result') or {}
        sci = d.get('scientist') or {}
        row = dict(
            line=ln,
            att=d.get('campaign_attempt'), pos=d.get('global_position'), status=d.get('status'),
            parent=d.get('parent_experiment_id'), exp=r.get('experiment_id'),
            qstat=r.get('quality_status'), dec=g(d,'pareto_decision','decision'),
            pelig=g(d,'pareto_decision','parent_eligible'), paretoelig=g(d,'pareto_decision','pareto_eligible'),
            bpb=r.get('validation_bpb'), pbpb=r.get('protected_validation_bpb'),
            tbs=r.get('total_batch_size'), tok=r.get('total_tokens'), upd=r.get('optimizer_updates'),
            params=r.get('num_params'), secs=r.get('protected_measured_training_seconds') or r.get('measured_training_seconds'),
            cat=d.get('change_category'), model=d.get('agent_model_identifier'), reas=d.get('reasoning_setting'),
            src=(d.get('candidate_source_after_sha256') or '')[:12], prog=(d.get('program_sha256') or '')[:12],
            manifest=(d.get('manifest_sha256') or '')[:12],
        )
        out.append((row, d))
    return out
if __name__ == '__main__':
    for p in sys.argv[1:]:
        print('##', p)
        for row, d in summarize(p):
            print({k: (round(v,6) if isinstance(v,float) else v) for k,v in row.items()})
