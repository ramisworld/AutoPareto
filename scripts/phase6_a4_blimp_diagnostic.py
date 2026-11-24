"""Diagnostic only (incident 7): reproduce both BLiMP numbers for gpt_a22-seed6 from one scoring pass.

Frozen evaluator rule: score ("", sentence); correct iff LL(good) > LL(bad); macro over 67 configs.
lm-eval 0.4.13 rule: score ("", " " + sentence) (default target_delimiter " "); correct iff
argmax([LL(good), LL(bad)]) == 0, so a tie counts as correct; macro over 67 configs.
Writes nothing into the results tree; prints one JSON line.
"""
import json, statistics, sys
from pathlib import Path

ROOT = Path("/workspace/autopareto-camera-ready")
sys.path.insert(0, str(ROOT))
import datasets
from downstream_evaluator import ModelScorer

suite = json.loads((ROOT / "evaluation/downstream_suite.json").read_text())
runtime = suite["runtime"]
task = next(t for t in suite["tasks"] if t["name"] == "blimp")
reused = json.loads((ROOT / "results/camera_ready_2026/phase5/A/downstream/downstream_models.json").read_text())
source = (ROOT / "results/camera_ready_2026/amendment1/E1/runs/e1-apq-tokens201326592-seed6/final.pt").resolve()
model = next(m for m in reused["models"] if Path(m["checkpoint"]).resolve() == source)
staged = Path("/root/phase6_stage/f604859e1982163fd4d2c9b36e2954172bf45dd4124b4b56035d055a912b2ab6.pt")
scorer = ModelScorer(candidate=Path(model["candidate"]), checkpoint=staged,
                     max_length=int(runtime["max_sequence_tokens"]), batch_size=int(runtime["scoring_batch_size"]))

names = sorted(d.name for d in Path("/root/phase6_lmeval_cache/hf/datasets/nyu-mll___blimp").iterdir() if d.is_dir())
assert len(names) == 67, len(names)
rules = {"frozen_nospace_strict": [], "lmeval_space_argmax": [], "space_strict": [], "nospace_argmax": []}
ties = {"nospace": 0, "space": 0}
pairs = 0
for name in names:
    rows = datasets.load_dataset(task["dataset"], name, revision=task["dataset_revision"], split=task["split"])
    reqs = []
    for r in rows:
        reqs += [("", r["sentence_good"]), ("", r["sentence_bad"]), ("", " " + r["sentence_good"]), ("", " " + r["sentence_bad"])]
    s = [x["log_likelihood"] for x in scorer.score(reqs)]
    acc = {k: [] for k in rules}
    for i in range(0, len(s), 4):
        g0, b0, g1, b1 = s[i:i + 4]
        acc["frozen_nospace_strict"].append(float(g0 > b0))
        acc["nospace_argmax"].append(float(g0 >= b0))
        acc["space_strict"].append(float(g1 > b1))
        acc["lmeval_space_argmax"].append(float(g1 >= b1))
        ties["nospace"] += g0 == b0
        ties["space"] += g1 == b1
        pairs += 1
    for k in rules:
        rules[k].append(statistics.mean(acc[k]))
out = {k: statistics.mean(v) for k, v in rules.items()}
out.update({"configs": len(names), "pairs": pairs, "ties": ties,
            "frozen_reported": 0.7254776119402985, "lmeval_reported": 0.7368805970149256})
print(json.dumps(out))
