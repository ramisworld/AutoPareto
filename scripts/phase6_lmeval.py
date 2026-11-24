#!/usr/bin/env python3
"""Phase 6 expanded downstream suite: lm-evaluation-harness tasks on one checkpoint.

The task definitions (datasets, splits, prompts, answer choices, metrics) are
lm-evaluation-harness's own, pinned by its installed version. Model access goes
through a thin ``lm_eval.api.model.LM`` subclass that scores with the frozen
project evaluator's ``ModelScorer`` (``downstream_evaluator.py``): the same
checkpoint loader, tokenizer, bfloat16 forward pass, fp32 log-softmax and
continuation tokenisation (context and continuation encoded together; BOS only
for an empty context; left truncation to 2,048 tokens). Rolling log-likelihood
(WikiText bits per byte) uses lm-eval's own disjoint rolling windows over the
same forward pass, with BOS as the prefix token.

    python scripts/phase6_lmeval.py --candidate <train.py> --checkpoint <ck> --output <out.json> \
        --tasks arc_easy,arc_challenge,piqa,hellaswag,winogrande,openbookqa,sciq,boolq,lambada_openai,blimp,wikitext
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TASKS = ["arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande", "openbookqa", "sciq",
                 "boolq", "lambada_openai", "blimp", "wikitext"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_lm(candidate: Path, checkpoint: Path, batch_size: int, max_length: int):
    import torch
    from lm_eval.api.model import LM
    from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

    from downstream_evaluator import ModelScorer

    class AutoParetoLM(LM):
        def __init__(self):
            super().__init__()
            self.scorer = ModelScorer(candidate=candidate, checkpoint=checkpoint,
                                      max_length=max_length, batch_size=batch_size)
            self.max_length = max_length

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            pairs = [request.args for request in requests]
            scored = self.scorer.score([(context, continuation) for context, continuation in pairs])
            return [(row["log_likelihood"], row["greedy_match"]) for row in scored]

        def _score_windows(self, windows):
            scorer = self.scorer
            out = []
            autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            with torch.inference_mode(), autocast:
                for context, target in windows:
                    full = list(context) + list(target)
                    inputs = torch.tensor([full[:-1]], dtype=torch.long, device=scorer.device)
                    logits = scorer.model(inputs)[0, -len(target):].float()
                    labels = torch.tensor(target, dtype=torch.long, device=scorer.device)
                    out.append(float(torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None]).sum().item()))
            return out

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            results = []
            for request in requests:
                (text,) = request.args
                tokens = self.scorer.encode(text)
                windows = list(map(make_disjoint_window, get_rolling_token_windows(
                    token_list=tokens, prefix_token=self.scorer.bos, max_seq_len=self.max_length, context_len=1)))
                results.append(sum(self._score_windows(windows)))
            return results

        def generate_until(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError("Phase 6 uses log-likelihood tasks only")

    return AutoParetoLM()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--suite", type=Path, default=ROOT / "evaluation/downstream_suite.json",
                        help="frozen suite whose runtime (scoring batch size, maximum sequence tokens) is reused")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    import lm_eval
    import torch
    from lm_eval import evaluator

    started = time.time()
    runtime = json.loads(args.suite.read_text())["runtime"]
    args.batch_size, args.max_length = int(runtime["scoring_batch_size"]), int(runtime["max_sequence_tokens"])
    lm = build_lm(args.candidate.resolve(), args.checkpoint.resolve(), args.batch_size, args.max_length)
    tasks = [task for task in args.tasks.split(",") if task]
    results = evaluator.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, limit=args.limit,
                                       bootstrap_iters=1000, log_samples=False, random_seed=0,
                                       numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234)
    payload = {
        "format": "autopareto_phase6_lmeval_v1",
        "label": args.label,
        "candidate": str(args.candidate), "candidate_sha256": sha256(args.candidate),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "script_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(ROOT / "downstream_evaluator.py"),
        "lm_eval_version": getattr(lm_eval, "__version__", None),
        "torch": torch.__version__, "python": platform.python_version(),
        "tasks": tasks, "num_fewshot": 0, "limit": args.limit, "max_length": args.max_length,
        "scoring_batch_size": args.batch_size, "suite_sha256": sha256(args.suite),
        "results": results.get("results"), "groups": results.get("groups"), "n_samples": results.get("n-samples"),
        "configs": results.get("configs"), "versions": results.get("versions"),
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".partial")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    tmp.rename(args.output)
    print(json.dumps({"label": args.label, "wall_s": round(payload["wall_seconds"]),
                      "tasks": {k: {m: v for m, v in r.items() if m.startswith(("acc", "bits", "word"))}
                                for k, r in (payload["results"] or {}).items() if not k.startswith("blimp_")}}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
