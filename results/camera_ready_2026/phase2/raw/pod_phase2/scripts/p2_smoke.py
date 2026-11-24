import sys, time
sys.path.insert(0, "/workspace/autopareto-camera-ready/results/camera_ready_2026/phase2/scripts")
import p2c_context_position as p
import downstream_evaluator as de
name = sys.argv[1]
_, cand, ck = next(m for m in p.MODELS if m[0] == name)
t = time.time()
s = de.ModelScorer(candidate=cand, checkpoint=ck, max_length=64, batch_size=1)
r = s.score([("The sky appears", " blue")])[0]
print(f"{name}: load+score ok in {time.time()-t:.0f}s, params {sum(x.numel() for x in s.model.parameters())/1e6:.1f}M, ll {r['log_likelihood']:.3f}", flush=True)
