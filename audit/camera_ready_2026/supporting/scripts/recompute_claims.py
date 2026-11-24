"""Recompute the AutoPareto paper's numerical claims from raw records (read-only).

Usage: python3 -I recompute_claims.py <repo_root> <out_json>
"""
import json
import math
import os
import sqlite3
import statistics as st
import sys
from collections import defaultdict, Counter

REPO = sys.argv[1]
OUT = sys.argv[2]
SNAP = os.path.join(REPO, "results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db")
P1 = os.path.join(REPO, "results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/phase1_equal_token_results.json")


def ro(p):
    c = sqlite3.connect(f"file:{p}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    return c


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    return pearson(ranks(x), ranks(y))


con = ro(SNAP)
out = {}
p1 = json.load(open(P1))
runs = p1["runs"]
models = {m["model_id"]: m for m in p1["models"]}

# --- Table 1 and headline ratios -------------------------------------------------------------
tab = {}
for mid, m in models.items():
    rs = [r for r in runs if r["model_id"] == mid]
    tab[mid] = {
        "n": len(rs),
        "source_experiment_id": m["source_experiment_id"],
        "source_sha256_prefix": m["source_sha256"][:8],
        "heldout_mean": st.mean(r["heldout_bpb"] for r in rs),
        "heldout_sd": st.stdev(r["heldout_bpb"] for r in rs),
        "final_val_mean": st.mean(r["final_validation_bpb"] for r in rs),
        "final_val_sd": st.stdev(r["final_validation_bpb"] for r in rs),
        "decode_mean": st.mean(r["stabilized_decode_tokens_per_second"] for r in rs),
        "decode_sd": st.stdev(r["stabilized_decode_tokens_per_second"] for r in rs),
        "peak_mem_values": sorted({r["peak_inference_cuda_memory_mb"] for r in rs}),
        "steady_s_mean": st.mean(r["steady_state_equivalent_seconds"] for r in rs),
        "wall_s_mean": st.mean(r["total_scientific_wall_seconds"] for r in rs),
        "optimizer_steps": sorted({r["optimizer_steps"] for r in rs}),
        "tokens": sorted({r["tokens"] for r in rs}),
        "per_seed": {r["seed"]: {"heldout": r["heldout_bpb"], "val": r["final_validation_bpb"],
                                 "decode": r["stabilized_decode_tokens_per_second"]} for r in rs},
    }
out["table1"] = tab
ap, ar = tab["autopareto_292"], tab["autoresearch_compatible"]
params = {mid: con.execute("select parameter_count from experiments where id=?", (m["source_experiment_id"],)).fetchone()[0] for mid, m in models.items()}
out["params"] = params
out["headline"] = {
    "decode_ratio": ap["decode_mean"] / ar["decode_mean"],
    "param_reduction_pct": 100 * (1 - params["autopareto_292"] / params["autoresearch_compatible"]),
    "memory_reduction_pct": 100 * (1 - ap["peak_mem_values"][0] / ar["peak_mem_values"][0]),
    "heldout_cost": ap["heldout_mean"] - ar["heldout_mean"],
    "val_cost_at_201M": ap["final_val_mean"] - ar["final_val_mean"],
    "paired_heldout_diff_by_seed": {s: ap["per_seed"][s]["heldout"] - ar["per_seed"][s]["heldout"] for s in (3, 4, 5)},
    "paired_val_diff_by_seed": {s: ap["per_seed"][s]["val"] - ar["per_seed"][s]["val"] for s in (3, 4, 5)},
}
# DB cross-check of the panel validation values.
dbval = {r["id"]: r["val_bpb"] for r in con.execute("select id, val_bpb, description from experiments where id between 301 and 321")}
xcheck = []
for r in con.execute("select id, description, val_bpb from experiments where id between 301 and 321"):
    rid = r["description"].split(" for ")[-1]
    jr = next((x for x in runs if x["run_id"] == rid), None)
    if jr:
        xcheck.append({"experiment_id": r["id"], "run_id": rid, "db_val_bpb": r["val_bpb"], "json_final_validation_bpb": jr["final_validation_bpb"],
                       "abs_diff": abs(r["val_bpb"] - jr["final_validation_bpb"])})
out["panel_val_db_vs_json"] = {"max_abs_diff": max(x["abs_diff"] for x in xcheck), "items": xcheck}
ho = {}
for r in con.execute("select d.experiment_id, t.metric_value from downstream_task_results t join downstream_evaluation_runs d on d.id=t.downstream_run_id where t.task_name='heldout_climbmix_bpb'"):
    ho[r[0]] = r[1]
out["heldout_db_vs_json_max_abs_diff"] = max(
    abs(ho[x["experiment_id"]] - next(j for j in runs if j["run_id"] == x["run_id"])["heldout_bpb"]) for x in xcheck if x["experiment_id"] in ho)
out["fixed_time_checkpoint_heldout"] = {str(k): v for k, v in ho.items() if k < 301}

# --- Archive bests (fixed-time discovery, val_bpb) -----------------------------------------
best = {}
for camp, label in [("autopareto-seed0-20260731", "AP seed0"), ("autopareto-seed1-20260731", "AP seed1"), ("autopareto-seed42-20260727", "AP seed42"),
                    ("autoresearch-seed42-20260723", "AR main50"), ("autoresearch-diagnostic5-seed42-20260724", "AR diag5"),
                    ("tpe-seed42-20260805", "TPE"), ("motpe-seed42-20260805", "MOTPE"), ("random-search-seed42-20260805", "RS"),
                    ("nsgaii-seed42-20260805", "NSGA-II"), ("cmaes-seed42-20260805", "CMA-ES")]:
    rows = con.execute("select id, val_bpb, status, inference_status, training_seed from experiments where campaign_id=? and val_bpb is not null order by val_bpb", (camp,)).fetchall()
    comp = [r for r in rows if r["inference_status"] in ("compatible",) or (r["inference_status"] is None and r["status"] == "completed")]
    best[label] = {"best_any": dict(rows[0]) if rows else None, "best_compatible_or_completed": dict(comp[0]) if comp else None,
                   "n_with_bpb": len(rows), "attempts": con.execute("select count(*) from experiments where campaign_id=?", (camp,)).fetchone()[0]}
out["archive_bests"] = best

# --- Fixed-time confirmations (seeds 0-2) --------------------------------------------------
conf = defaultdict(list)
for r in con.execute("select c.candidate_experiment_id cid, c.seed, c.status, t.val_bpb, t.total_training_tokens tok, t.parameter_count p from confirmation_runs c join training_runs t on t.id=c.training_run_id"):
    conf[r["cid"]].append(dict(r))
disc = {r["id"]: dict(r) for r in con.execute("select id, val_bpb, total_training_tokens, training_seed, campaign_id from experiments where id in (65,68,124,134,136,137)")}
ft = {}
for cid, rs in conf.items():
    vals = [r["val_bpb"] for r in rs]
    ft[cid] = {"discovery_val": disc[cid]["val_bpb"], "discovery_seed": disc[cid]["training_seed"], "discovery_tokens": disc[cid]["total_training_tokens"],
               "confirm_vals_by_seed": {r["seed"]: r["val_bpb"] for r in rs}, "confirm_tokens_by_seed": {r["seed"]: r["tok"] for r in rs},
               "confirm_status_by_seed": {r["seed"]: r["status"] for r in rs},
               "confirm_mean": st.mean(vals), "confirm_sd": st.stdev(vals),
               "optimism_discovery_minus_confirm_mean": disc[cid]["val_bpb"] - st.mean(vals)}
out["fixed_time_confirmations"] = {str(k): v for k, v in ft.items()}
pair = {s: ft[134]["confirm_vals_by_seed"][s] - ft[68]["confirm_vals_by_seed"][s] for s in (0, 1, 2)}
out["fixed_time_AP292_minus_AR_by_seed"] = pair
out["fixed_time_AP292_minus_AR_mean"] = st.mean(pair.values())

# Baseline seed effects (fixed time, cached-fresh baseline ids 8-12 plus seed 42 reference id 13 and early ids 3-7).
base = {r["training_seed"]: r["val_bpb"] for r in con.execute("select training_seed, val_bpb from experiments where campaign_id='baseline-a40-cached-fresh-20260722'")}
base_early = {r["training_seed"]: r["val_bpb"] for r in con.execute("select training_seed, val_bpb from experiments where campaign_id='baseline-a40-seed0'")}
ref42 = con.execute("select val_bpb from experiments where id=13").fetchone()[0]
out["baseline_seed_effects"] = {"cached_fresh_seeds0_4": base, "early_seeds0_4": base_early, "seed42_reference": ref42,
                                "cached_fresh_mean": st.mean(base.values()), "cached_fresh_sd": st.stdev(base.values()),
                                "seed42_minus_cached_fresh_mean": ref42 - st.mean(base.values()),
                                "seed0_minus_seed42": base[0] - ref42}

# Ablations.
abl = defaultdict(dict)
for r in con.execute("select a.ablation_name n, t.seed s, t.val_bpb v, t.total_training_tokens tok from ablation_runs a join training_runs t on t.id=a.training_run_id"):
    abl[r["n"]][r["s"]] = (r["v"], r["tok"])
base012 = st.mean(base[s] for s in (0, 1, 2))
ar012 = ft[68]["confirm_mean"]
bo = st.mean(v for v, _ in abl["batch_only"].values())
nbo = st.mean(v for v, _ in abl["nonbatch_only"].values())
out["ablation_decomposition_seeds0_2"] = {"baseline_mean": base012, "autoresearch_exp68_mean": ar012, "batch_only_mean": bo, "nonbatch_only_mean": nbo,
                                          "total_gain": base012 - ar012, "batch_only_gain": base012 - bo, "nonbatch_only_gain": base012 - nbo,
                                          "batch_share_of_total": (base012 - bo) / (base012 - ar012),
                                          "raw": {k: {str(s): v for s, v in d.items()} for k, d in abl.items()}}

# Rank correlation: discovery val (fixed time) vs 201M held-out, across the 7 panel recipes.
disc_val = {mid: con.execute("select val_bpb, training_seed from experiments where id=?", (m["source_experiment_id"],)).fetchone() for mid, m in models.items()}
mids = list(models)
dv = [disc_val[m][0] for m in mids]
hv = [tab[m]["heldout_mean"] for m in mids]
vv = [tab[m]["final_val_mean"] for m in mids]
out["discovery_vs_panel"] = {"recipes": {m: {"discovery_val": disc_val[m][0], "discovery_seed": disc_val[m][1], "panel_heldout": tab[m]["heldout_mean"], "panel_val": tab[m]["final_val_mean"]} for m in mids},
                             "spearman_discovery_vs_heldout": spearman(dv, hv), "pearson_discovery_vs_heldout": pearson(dv, hv),
                             "spearman_panel_val_vs_heldout": spearman(vv, hv), "pearson_panel_val_vs_heldout": pearson(vv, hv)}
fixed = [134, 136, 137, 124, 68, 65]
out["discovery_vs_fixed_time_confirm"] = {"spearman": spearman([ft[i]["discovery_val"] for i in fixed], [ft[i]["confirm_mean"] for i in fixed]),
                                          "pearson": pearson([ft[i]["discovery_val"] for i in fixed], [ft[i]["confirm_mean"] for i in fixed])}

# --- Downstream (201M runs) ------------------------------------------------------------------
ds = defaultdict(lambda: defaultdict(list))
blimp_cfg = defaultdict(lambda: defaultdict(list))
for r in con.execute("""select d.experiment_id eid, t.task_name tn, t.metric_name mn, t.metric_value mv, t.raw_json rj
                        from downstream_task_results t join downstream_evaluation_runs d on d.id=t.downstream_run_id where d.experiment_id between 301 and 321"""):
    desc = con.execute("select description from experiments where id=?", (r["eid"],)).fetchone()[0]
    rid = desc.split(" for ")[-1]
    mid = next(x["model_id"] for x in runs if x["run_id"] == rid)
    ds[mid][(r["tn"], r["mn"])].append(r["mv"])
    if r["tn"] == "blimp" and r["mn"] == "macro_accuracy" and r["rj"]:
        try:
            raw = json.loads(r["rj"])
        except Exception:
            raw = {}
        cfgs = raw.get("configs") or raw.get("per_config") or raw.get("subtasks") or {}
        if isinstance(cfgs, dict):
            for k, v in cfgs.items():
                val = v.get("accuracy") if isinstance(v, dict) else v
                if isinstance(val, (int, float)):
                    blimp_cfg[mid][k].append(val)
        elif isinstance(cfgs, list):
            for v in cfgs:
                if isinstance(v, dict):
                    k = v.get("config") or v.get("name")
                    val = v.get("accuracy")
                    if k and isinstance(val, (int, float)):
                        blimp_cfg[mid][k].append(val)
out["downstream_means_201M"] = {mid: {f"{k[0]}:{k[1]}": st.mean(v) for k, v in d.items()} for mid, d in ds.items()}
if blimp_cfg.get("autopareto_292") and blimp_cfg.get("autoresearch_compatible"):
    a, b = blimp_cfg["autopareto_292"], blimp_cfg["autoresearch_compatible"]
    keys = sorted(set(a) & set(b))
    diffs = [st.mean(a[k]) - st.mean(b[k]) for k in keys]
    out["blimp_config_comparison"] = {"n_configs": len(keys), "ap_lower": sum(d < 0 for d in diffs), "tied": sum(d == 0 for d in diffs), "ap_higher": sum(d > 0 for d in diffs)}
else:
    out["blimp_config_comparison"] = "per-config BLiMP values not found in raw_json"
depth = {"baseline": 8, "autoresearch_compatible": 8, "tpe_quality": 6, "motpe_quality": 6, "autopareto_292": 4, "autopareto_299": 3, "autopareto_294": 3}
bl = [out["downstream_means_201M"][m]["blimp:macro_accuracy"] for m in mids]
out["correlations_7_recipes"] = {"blimp_vs_heldout_pearson": pearson(bl, hv), "blimp_vs_depth_pearson": pearson(bl, [depth[m] for m in mids]),
                                 "blimp_vs_params_pearson": pearson(bl, [params[m] for m in mids]), "n": len(mids),
                                 "depth_assumed_from_sources": depth}

# --- Lineage of experiment 134 ---------------------------------------------------------------
chain = []
cur = 134
while cur:
    r = con.execute("select id, experiment_number, parent_experiment_id, val_bpb, median_cached_decode_tokens_per_second d, campaign_id from experiments where id=?", (cur,)).fetchone()
    if not r:
        break
    chain.append(dict(r))
    cur = r["parent_experiment_id"]
out["lineage_exp134"] = chain
s0 = {r["experiment_number"]: dict(r) for r in con.execute("select id, experiment_number, val_bpb, median_cached_decode_tokens_per_second d, status, change_category, parent_experiment_id from experiments where campaign_id='autopareto-seed0-20260731'")}
out["seed0_campaign"] = s0

# --- Edit categories across the three AutoPareto campaigns -----------------------------------
cats = Counter(r[0] for r in con.execute("select change_category from experiments where campaign_id in ('autopareto-seed0-20260731','autopareto-seed1-20260731','autopareto-seed42-20260727')"))
out["ap_change_categories_db"] = dict(cats)
out["ap_attempt_rows"] = con.execute("select count(*) from experiments where campaign_id in ('autopareto-seed0-20260731','autopareto-seed1-20260731','autopareto-seed42-20260727')").fetchone()[0]

# --- Discovery-contract exposure for the headline pair ---------------------------------------
out["discovery_exposure"] = {str(i): dict(con.execute("select id, total_training_tokens, parameter_count, val_bpb, training_seconds from experiments where id=?", (i,)).fetchone()) for i in (68, 134, 136, 137, 183, 234, 4)}

json.dump(out, open(OUT, "w"), indent=2, default=str)
h = out["headline"]
print("decode_ratio %.4f params_red %.3f%% mem_red %.3f%% heldout_cost %.6f val_cost %.6f" % (h["decode_ratio"], h["param_reduction_pct"], h["memory_reduction_pct"], h["heldout_cost"], h["val_cost_at_201M"]))
print("panel val db-vs-json max diff", out["panel_val_db_vs_json"]["max_abs_diff"], "heldout db-vs-json max diff", out["heldout_db_vs_json_max_abs_diff"])
print("fixed-time AP292-AR by seed", pair, "mean", out["fixed_time_AP292_minus_AR_mean"])
print("ablation", {k: round(v, 5) if isinstance(v, float) else v for k, v in out["ablation_decomposition_seeds0_2"].items() if k != "raw"})
print("spearman disc vs heldout", round(out["discovery_vs_panel"]["spearman_discovery_vs_heldout"], 3), "pearson", round(out["discovery_vs_panel"]["pearson_discovery_vs_heldout"], 3))
print("discovery vs fixed-time confirm", out["discovery_vs_fixed_time_confirm"])
print("corr", out["correlations_7_recipes"])
print("blimp cfg", out["blimp_config_comparison"])
print("AP292 vs AR downstream:", {k: (round(out['downstream_means_201M']['autopareto_292'][k], 4), round(out['downstream_means_201M']['autoresearch_compatible'][k], 4)) for k in out['downstream_means_201M']['autopareto_292']})
print("categories", out["ap_change_categories_db"], "rows", out["ap_attempt_rows"])
print("lineage", [(c["id"], c["experiment_number"], round(c["val_bpb"], 4) if c["val_bpb"] else None) for c in chain])
print("fixed-time heldout (downstream on fixed-time ckpts)", out["fixed_time_checkpoint_heldout"])
