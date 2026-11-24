"""Build the AutoPareto master experiment registry (read-only over the archive).

Usage: python3 -I build_registry.py <repo_root> <out_dir>
Writes <out_dir>/02_EXPERIMENT_REGISTRY.csv and <out_dir>/supporting/registry_build_report.json.
"""
import ast
import csv
import glob
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict

REPO = sys.argv[1]
OUT = sys.argv[2]
SNAP = os.path.join(REPO, "results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db")
MAIN = os.path.join(REPO, "results/results.db")
PHASE1_JSON = os.path.join(REPO, "results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/phase1_equal_token_results.json")

CONST_NAMES = ["DEPTH", "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN", "TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE",
               "MAX_SEQ_LEN", "MATRIX_LR", "EMBEDDING_LR", "UNEMBEDDING_LR", "SCALAR_LR", "WEIGHT_DECAY",
               "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC", "ADAM_BETAS", "TIME_BUDGET"]

PANEL_SOURCES = {4: "baseline", 68: "autoresearch_compatible", 183: "tpe_quality", 234: "motpe_quality",
                 134: "autopareto_292", 137: "autopareto_299", 136: "autopareto_294"}
FIXED_TIME_CONFIRMED = {65: "lowest_observed_quality (AutoResearch)", 68: "best_compatible (AutoResearch)",
                        124: "lowest_memory_within_1pct_quality", 134: "best_quality",
                        136: "largest_exclusive_hypervolume_contribution", 137: "fastest_within_1pct_quality"}


def ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def safe_eval(expr):
    try:
        node = ast.parse(expr.strip(), mode="eval").body
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, str)):
            return n.value
        if isinstance(n, ast.Tuple):
            return tuple(ev(e) for e in n.elts)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            ops = {ast.Add: lambda x, y: x + y, ast.Sub: lambda x, y: x - y, ast.Mult: lambda x, y: x * y,
                   ast.Div: lambda x, y: x / y, ast.FloorDiv: lambda x, y: x // y, ast.Pow: lambda x, y: x ** y}
            return ops[type(n.op)](a, b)
        raise ValueError(type(n).__name__)

    try:
        return ev(node)
    except Exception:
        return None


def parse_source(path):
    out = {"candidate_source_path": "", "candidate_sha256": ""}
    if not path or not os.path.exists(path):
        return out
    data = open(path, "rb").read()
    out["candidate_source_path"] = os.path.relpath(path, REPO)
    out["candidate_sha256"] = hashlib.sha256(data).hexdigest()
    text = data.decode("utf-8", "replace")
    for name in CONST_NAMES:
        m = re.search(rf"^{name}\s*=\s*(.+?)(?:\s+#.*)?$", text, re.M)
        if m:
            v = safe_eval(m.group(1))
            out[name] = v if v is not None else m.group(1).strip()
    m = re.search(r"n_kv_head\s*=\s*([^,\)\n]+)", text[text.find("def build_model_config"):] if "def build_model_config" in text else text)
    out["n_kv_head_expr"] = m.group(1).strip() if m else ""
    m = re.search(r"self\.c_fc\s*=\s*nn\.Linear\(\s*config\.n_embd\s*,\s*([^,]+?)\s*,", text)
    out["mlp_hidden_expr"] = m.group(1).strip() if m else ""
    m = re.search(r"def has_ve\([^)]*\)[^:]*:\s*\n(?:\s*\"\"\".*?\"\"\"\s*\n)?\s*return\s+(.+)", text, re.S)
    out["has_ve_expr"] = m.group(1).strip().splitlines()[0] if m else ""
    return out


def short(s, n=300):
    if s is None:
        return ""
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def main():
    report = {"sources": {}, "anomalies": []}
    con = ro(SNAP)
    con.row_factory = sqlite3.Row
    exps = [dict(r) for r in con.execute("select * from experiments order by id")]
    camps = {r["campaign_id"]: dict(r) for r in con.execute("select * from campaigns")}
    truns = defaultdict(list)
    for r in con.execute("select * from training_runs order by id"):
        truns[r["experiment_id"]].append(dict(r))
    conf = {r["training_run_id"]: dict(r) for r in con.execute("select * from confirmation_runs")}
    abl = {r["training_run_id"]: dict(r) for r in con.execute("select * from ablation_runs")}
    report["sources"]["snapshot_db"] = {"path": os.path.relpath(SNAP, REPO), "experiments": len(exps)}

    # Consistency with the main database (ids 1..300).
    mcon = ro(MAIN)
    mexp = {r[0]: r[1:] for r in mcon.execute("select id, campaign_id, val_bpb, status, total_training_tokens from experiments")}
    mism = [e["id"] for e in exps if e["id"] in mexp and mexp[e["id"]] != (e["campaign_id"], e["val_bpb"], e["status"], e["total_training_tokens"])]
    report["sources"]["main_db"] = {"path": "results/results.db", "experiments": len(mexp), "mismatching_overlap_ids": mism,
                                    "ids_only_in_snapshot": sorted(set(e["id"] for e in exps) - set(mexp))}
    missing_ids = sorted(set(range(1, max(e["id"] for e in exps) + 1)) - set(e["id"] for e in exps))
    report["anomalies"].append({"type": "missing_experiment_ids", "ids": missing_ids,
                                "note": "absent from every retained DB snapshot (results/archive, results/snapshots); gap sits inside autopareto-seed42-20260727 id range 84-112"})

    phase1 = json.load(open(PHASE1_JSON))
    p1 = {r["run_id"]: r for r in phase1["runs"]}
    p1_models = {m["model_id"]: m for m in phase1["models"]}

    rows = []
    dup_tr = []
    for e in exps:
        camp = camps.get(e["campaign_id"], {})
        trs = truns.get(e["id"], [])
        # Detect duplicated training-run rows that describe the same run (identical val_bpb and tokens).
        if len(trs) > 1:
            sig = {(t["val_bpb"], t["total_training_tokens"]) for t in trs if t["val_bpb"] is not None}
            kinds = ",".join(f"{t['run_kind']}:{t['status']}" for t in trs)
            dup_tr.append({"experiment_id": e["id"], "training_runs": [t["id"] for t in trs], "kinds": kinds,
                           "distinct_val_bpb_token_pairs": len(sig)})
        roles = []
        for t in trs:
            if t["id"] in conf:
                roles.append(f"confirmation:{conf[t['id']]['selection_role']}:seed{conf[t['id']]['seed']}:{conf[t['id']]['status']}")
            if t["id"] in abl:
                roles.append(f"ablation:{abl[t['id']]['ablation_name']}:src{abl[t['id']]['source_experiment_id']}")
        # Candidate source.
        src_path = os.path.join(REPO, "results/candidates", e["campaign_id"], f"{int(e['experiment_number']):04d}-train.py") if e["experiment_number"] is not None else ""
        heldout = ""
        run_id = ""
        if e["campaign_id"] == "phase1-200m-downstream":
            m = re.search(r"for (phase1-\S+)", e["description"] or "")
            run_id = m.group(1) if m else ""
            pr = p1.get(run_id)
            if pr:
                heldout = pr["heldout_bpb"]
                model = p1_models.get(pr["model_id"], {})
                srcexp = model.get("source_experiment_id")
                srcrow = next((x for x in exps if x["id"] == srcexp), None)
                if srcrow:
                    src_path = os.path.join(REPO, "results/candidates", srcrow["campaign_id"], f"{int(srcrow['experiment_number']):04d}-train.py")
        src = parse_source(src_path)
        tb = src.get("TOTAL_BATCH_SIZE")
        tokens = e["total_training_tokens"]
        updates = ""
        upd_src = ""
        if isinstance(tb, int) and tokens:
            updates = round(tokens / tb, 2)
            upd_src = "DERIVED tokens/TOTAL_BATCH_SIZE (includes no warm-up)"
        if run_id and run_id in p1:
            updates = p1[run_id]["optimizer_steps"]
            upd_src = "OBSERVED phase1_equal_token_results.json optimizer_steps"
        params = e["parameter_count"]
        sel = []
        if e["id"] in FIXED_TIME_CONFIRMED:
            sel.append("selected_for_fixed_time_confirmation(seeds0-2):" + FIXED_TIME_CONFIRMED[e["id"]])
        if e["id"] in PANEL_SOURCES:
            sel.append("source_of_201M_panel_recipe:" + PANEL_SOURCES[e["id"]])
        if e["campaign_id"] == "phase1-200m-downstream":
            sel.append("201M_matched_token_panel_run")
        if e["id"] in (134, 68) or e["campaign_id"] == "phase1-200m-downstream":
            sel.append("published_in_paper")
        era = {"setup": "setup", "baseline": "baseline", "campaign": "discovery", "confirmation": "fixed_time_confirmation",
               "ablation": "fixed_time_ablation", "phase1_downstream": "matched_token_panel"}.get(e["phase"], e["phase"])
        if e["campaign_id"].startswith("downstream-recovery"):
            era = "downstream_checkpoint_recreation"
        rows.append({
            "registry_key": f"db:{e['id']}", "era": era, "source_store": os.path.relpath(SNAP, REPO),
            "source_locator": f"experiments.id={e['id']}; training_runs.id={'|'.join(str(t['id']) for t in trs)}",
            "campaign_id": e["campaign_id"], "campaign_method": camp.get("method", e["method"]), "pair_a_group": "",
            "namespace_list": "", "arm": "", "experiment_id": e["id"], "attempt": e["experiment_number"],
            "run_kind": "|".join(sorted({t["run_kind"] for t in trs})), "chrono_start": e["started_at"],
            "chrono_end": e["finished_at"], "parent_experiment_id": e["parent_experiment_id"],
            "agent_model": e["coding_model_name"] or camp.get("coding_model_name") or "",
            "reasoning_level": e["reasoning_level"] or camp.get("reasoning_level") or "",
            "program_sha256": e["program_sha256"] or camp.get("program_sha256") or "",
            "contract": "fixed_time_300s" if era in ("discovery", "baseline", "fixed_time_confirmation", "fixed_time_ablation", "downstream_checkpoint_recreation") else ("fixed_tokens_201326592" if era == "matched_token_panel" else "UNKNOWN"),
            "change_category": e["change_category"] or "", "hypothesis_or_description": short(e["hypothesis"] or e["intended_change"] or e["description"]),
            "candidate_source_path": src.get("candidate_source_path", ""), "candidate_sha256": src.get("candidate_sha256", ""),
            "git_commit": e["git_commit"] or "", "depth": src.get("DEPTH", ""), "aspect_ratio": src.get("ASPECT_RATIO", ""),
            "head_dim": src.get("HEAD_DIM", ""), "n_kv_head_expr": src.get("n_kv_head_expr", ""),
            "window_pattern": src.get("WINDOW_PATTERN", ""), "mlp_hidden_expr": src.get("mlp_hidden_expr", ""),
            "has_ve_expr": src.get("has_ve_expr", ""), "total_batch_size": tb if tb is not None else "",
            "device_batch_size": src.get("DEVICE_BATCH_SIZE", ""), "seq_len": src.get("MAX_SEQ_LEN", 2048 if src.get("candidate_sha256") else ""),
            "matrix_lr": src.get("MATRIX_LR", ""), "embedding_lr": src.get("EMBEDDING_LR", ""),
            "unembedding_lr": src.get("UNEMBEDDING_LR", ""), "scalar_lr": src.get("SCALAR_LR", ""),
            "weight_decay": src.get("WEIGHT_DECAY", ""), "warmup_ratio": src.get("WARMUP_RATIO", ""),
            "warmdown_ratio": src.get("WARMDOWN_RATIO", ""), "final_lr_frac": src.get("FINAL_LR_FRAC", ""),
            "parameter_count": params if params is not None else "", "training_seed": e["training_seed"] if e["training_seed"] is not None else "",
            "dataset_ref": "karpathy/climbmix-400b-shuffle (revision not stored per run; manifest-level only)",
            "training_seconds": e["training_seconds"] if e["training_seconds"] is not None else "",
            "total_training_tokens": tokens if tokens is not None else "", "optimizer_updates": updates,
            "optimizer_updates_source": upd_src,
            "approx_6ND_flops": f"{6 * params * tokens:.3e}" if params and tokens else "",
            "harness_flops_estimate": "", "val_bpb": e["val_bpb"] if e["val_bpb"] is not None else "",
            "heldout_bpb": heldout, "decode_tok_s_median": e["median_cached_decode_tokens_per_second"] if e["median_cached_decode_tokens_per_second"] is not None else (e["inference_tokens_per_second"] or ""),
            "peak_infer_mem_mb": e["peak_inference_cuda_memory_mb"] if e["peak_inference_cuda_memory_mb"] is not None else "",
            "incr_kv_mb": e["incremental_kv_cache_memory_mb"] if e["incremental_kv_cache_memory_mb"] is not None else "",
            "status": e["status"], "training_status": e["training_status"] or "", "inference_status": e["inference_status"] or "",
            "failure_reason_short": short(e["crash_or_failure_reason"], 160), "decision": e["decision"] or "",
            "valid_result": e["valid_result"], "exclude_from_analysis": e["exclude_from_analysis"],
            "selection_status": "; ".join(sel + roles), "published_in_paper": "yes" if "published_in_paper" in sel else "no",
            "notes": "",
        })
    report["anomalies"].append({"type": "experiments_with_multiple_training_run_rows", "items": dup_tr})
    stuck = [dict(r) for r in con.execute("select id, experiment_id, run_kind, seed, started_at from training_runs where status='running'")]
    report["anomalies"].append({"type": "training_runs_left_in_running_state", "items": stuck})

    # Pair-A study histories (every namespace).
    files = sorted(set(glob.glob(os.path.join(REPO, "runs/*/pair-a-fixed-*/history.jsonl"))))
    by_hash = {}
    pair_rows = []
    group_of = lambda ns: (
        "nonstudy" if ns.startswith("nonstudy") else
        "G0_pair-a_original" if ns == "pair-a" else
        "G1_20260906_fresh" if ns == "pair-a-official-20260906-fresh" else
        "G2_20260906_fixed_warmup_v2(+0907 continuations)" if ns.startswith("pair-a-official-20260906-fixed-warmup-v2") else
        "G3_20260909_new_a40" if ns == "pair-a-official-20260909-new-a40" else
        "G4_20260914_new_a40(+0916 continuations)" if ns.startswith("pair-a-official-20260914-new-a40") else "UNGROUPED")
    for f in files:
        ns = f.split(os.sep)[-3]
        arm = "T" if "fixed-time" in f else "N"
        for line_no, line in enumerate(open(f), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            res = r.get("result") or {}
            h = r.get("candidate_source_after_sha256") or ""
            key = (group_of(ns), arm, r.get("campaign_attempt"), h or f"{ns}:{line_no}")
            if key in by_hash:
                by_hash[key]["namespace_list"] += "|" + ns
                continue
            inf = res.get("inference_result") or {}
            row = {k: "" for k in rows[0].keys()}
            row.update({
                "registry_key": f"pairA:{group_of(ns)}:{arm}:{r.get('campaign_attempt')}", "era": "pair_a_contract_study",
                "source_store": os.path.relpath(f, REPO), "source_locator": f"line {line_no}", "campaign_id": r.get("campaign_id", ""),
                "campaign_method": "autopareto_pair_a", "pair_a_group": group_of(ns), "namespace_list": ns, "arm": arm,
                "experiment_id": res.get("experiment_id", ""), "attempt": r.get("campaign_attempt"), "run_kind": "pair_a_attempt",
                "parent_experiment_id": r.get("parent_experiment_id", ""), "agent_model": r.get("agent_model_identifier", ""),
                "reasoning_level": r.get("reasoning_setting", ""), "program_sha256": r.get("program_sha256", ""),
                "contract": r.get("contract_id", ""), "change_category": r.get("change_category") or "",
                "hypothesis_or_description": short(r.get("intended_change") or r.get("hypothesis")),
                "candidate_sha256": h, "total_batch_size": res.get("total_batch_size", ""),
                "device_batch_size": res.get("device_batch_size", ""), "seq_len": res.get("sequence_length", ""),
                "parameter_count": res.get("num_params", ""), "training_seed": r.get("training_seed", ""),
                "training_seconds": res.get("measured_training_seconds", ""), "total_training_tokens": res.get("total_tokens", ""),
                "optimizer_updates": res.get("optimizer_updates", ""),
                "optimizer_updates_source": "OBSERVED history.jsonl result.optimizer_updates" if res.get("optimizer_updates") else "",
                "approx_6ND_flops": f"{6 * res['num_params'] * res['total_tokens']:.3e}" if res.get("num_params") and res.get("total_tokens") else "",
                "harness_flops_estimate": res.get("training_flops_estimate", ""), "val_bpb": res.get("validation_bpb", ""),
                "decode_tok_s_median": inf.get("cached_decode_tokens_per_second_median", ""), "status": r.get("status", ""),
                "inference_status": inf.get("inference_status", ""), "decision": json.dumps(r.get("pareto_decision")) if r.get("pareto_decision") else "",
                "selection_status": "paper_reported_prefix" if group_of(ns).startswith("G2") and (r.get("campaign_attempt") or 99) <= 4 else "not_reported_in_paper",
                "published_in_paper": "yes" if group_of(ns).startswith("G2") and (r.get("campaign_attempt") or 99) <= 4 else "no",
                "dataset_ref": "karpathy/climbmix-400b-shuffle (prepare_sha256=" + str(r.get("prepare_sha256", ""))[:12] + ")",
                "notes": f"manifest_sha256={str(r.get('manifest_sha256',''))[:12]}",
            })
            by_hash[key] = row
            pair_rows.append(row)
    report["sources"]["pair_a_histories"] = {"files": [os.path.relpath(x, REPO) for x in files], "unique_attempt_records": len(pair_rows)}

    all_rows = rows + pair_rows
    os.makedirs(os.path.join(OUT, "supporting"), exist_ok=True)
    with open(os.path.join(OUT, "02_EXPERIMENT_REGISTRY.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in all_rows:
            w.writerow({k: ("UNKNOWN" if v in ("", None) and k in ("parameter_count", "total_training_tokens", "val_bpb", "training_seed", "candidate_sha256", "parent_experiment_id", "optimizer_updates") else v) for k, v in row.items()})
    report["row_counts"] = {"db_experiments": len(rows), "pair_a_unique_attempts": len(pair_rows), "total": len(all_rows)}
    json.dump(report, open(os.path.join(OUT, "supporting", "registry_build_report.json"), "w"), indent=2, default=str)
    print(json.dumps(report["row_counts"]), "missing ids", missing_ids, "mismatch", mism)


if __name__ == "__main__":
    main()
