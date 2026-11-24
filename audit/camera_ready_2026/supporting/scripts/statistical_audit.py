"""Offline statistical and causal audit of existing AutoPareto records (no new training).

Usage: python3 -I statistical_audit.py <repo_root> <claims_json> <out_json>
"""
import json
import math
import os
import sqlite3
import statistics as st
import sys
from collections import defaultdict
from itertools import product

REPO, CLAIMS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
C = json.load(open(CLAIMS))
SNAP = os.path.join(REPO, "results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db")
con = sqlite3.connect(f"file:{SNAP}?mode=ro&immutable=1", uri=True)
con.row_factory = sqlite3.Row
T_CRIT = {2: 4.302652729749464, 4: 2.7764451051977987, 5: 2.570581835636314}
out = {}


def paired(xs, ys):
    d = [a - b for a, b in zip(xs, ys)]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    half = T_CRIT[n - 1] * se
    t = m / se if se > 0 else float("inf")
    return {"n": n, "diffs": d, "mean": m, "sd": sd, "ci95": [m - half, m + half], "t": t, "all_same_sign": all(x < 0 for x in d) or all(x > 0 for x in d)}


def welch(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    vx, vy = st.variance(xs), st.variance(ys)
    se = math.sqrt(vx / len(xs) + vy / len(ys))
    return {"mean_diff": mx - my, "se": se, "t": (mx - my) / se if se else float("inf")}


t1 = C["table1"]
ap, ar = t1["autopareto_292"]["per_seed"], t1["autoresearch_compatible"]["per_seed"]
seeds = ["3", "4", "5"] if "3" in ap else [3, 4, 5]
out["matched_token_AP292_minus_AR"] = {
    "heldout": paired([ap[s]["heldout"] for s in seeds], [ar[s]["heldout"] for s in seeds]),
    "validation": paired([ap[s]["val"] for s in seeds], [ar[s]["val"] for s in seeds]),
    "decode_ratio_by_seed": [ap[s]["decode"] / ar[s]["decode"] for s in seeds],
}
ft = C["fixed_time_confirmations"]
f134 = ft["134"]["confirm_vals_by_seed"]
f68 = ft["68"]["confirm_vals_by_seed"]
out["fixed_time_AP292_minus_AR_val_seeds012"] = paired([f134[s] for s in ("0", "1", "2")], [f68[s] for s in ("0", "1", "2")])

# Reversal decomposition table.
d = C["discovery_exposure"]
out["reversal_table"] = [
    {"comparison": "discovery archive (unmatched seeds 0 vs 42)", "contract": "300 s", "eval": "validation", "delta_AP292_minus_AR": d["134"]["val_bpb"] - d["68"]["val_bpb"],
     "tokens_AP292": d["134"]["total_training_tokens"], "tokens_AR": d["68"]["total_training_tokens"], "n": "1 vs 1"},
    {"comparison": "fixed-time re-runs (matched seeds 0,1,2)", "contract": "300 s", "eval": "validation", "delta_AP292_minus_AR": out["fixed_time_AP292_minus_AR_val_seeds012"]["mean"],
     "tokens_AP292": ft["134"]["confirm_tokens_by_seed"], "tokens_AR": ft["68"]["confirm_tokens_by_seed"], "n": "3 paired"},
    {"comparison": "fixed-time checkpoints, held-out shard (AP292 seed1 vs AR seed2; unmatched single runs)", "contract": "300 s", "eval": "held-out",
     "delta_AP292_minus_AR": C["fixed_time_checkpoint_heldout"]["292"] - C["fixed_time_checkpoint_heldout"]["77"], "n": "1 vs 1"},
    {"comparison": "matched tokens 201,326,592 (matched seeds 3,4,5)", "contract": "201M tokens", "eval": "validation", "delta_AP292_minus_AR": out["matched_token_AP292_minus_AR"]["validation"]["mean"], "n": "3 paired"},
    {"comparison": "matched tokens 201,326,592 (matched seeds 3,4,5)", "contract": "201M tokens", "eval": "held-out", "delta_AP292_minus_AR": out["matched_token_AP292_minus_AR"]["heldout"]["mean"], "n": "3 paired"},
]

# Exposure response: fixed-time confirmation means (seeds 0-2) vs 201M validation means (seeds 3-5).
base_fixed = {s: v for s, v in C["baseline_seed_effects"]["cached_fresh_seeds0_4"].items() if s in ("0", "1", "2", 0, 1, 2)}
base_tokens = [r[0] for r in con.execute("select total_training_tokens from experiments where campaign_id='baseline-a40-cached-fresh-20260722' and training_seed in (0,1,2)")]
resp = []
for label, fixed_mean, fixed_tok, panel_key in [
    ("baseline", st.mean(base_fixed.values()), st.mean(base_tokens), "baseline"),
    ("autoresearch_exp68", ft["68"]["confirm_mean"], st.mean(ft["68"]["confirm_tokens_by_seed"].values()), "autoresearch_compatible"),
    ("AP292_exp134", ft["134"]["confirm_mean"], st.mean(ft["134"]["confirm_tokens_by_seed"].values()), "autopareto_292"),
    ("AP299_exp137", ft["137"]["confirm_mean"], st.mean(ft["137"]["confirm_tokens_by_seed"].values()), "autopareto_299"),
    ("AP294_exp136", ft["136"]["confirm_mean"], st.mean(ft["136"]["confirm_tokens_by_seed"].values()), "autopareto_294"),
]:
    pv = t1[panel_key]["final_val_mean"]
    resp.append({"recipe": label, "fixed_time_val_mean_seeds012": fixed_mean, "fixed_time_tokens_mean": fixed_tok, "panel_val_mean_seeds345": pv,
                 "token_ratio_panel_over_fixed": 201326592 / fixed_tok, "log2_token_ratio": math.log2(201326592 / fixed_tok), "delta_val": pv - fixed_mean})
out["exposure_response"] = resp
xs = [r["log2_token_ratio"] for r in resp]
ys = [r["delta_val"] for r in resp]
mx, my = st.mean(xs), st.mean(ys)
out["exposure_response_pearson"] = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))

# Within-recipe seed variability.
sdv = {k: v["confirm_sd"] for k, v in ft.items()}
sd201 = {k: v["final_val_sd"] for k, v in t1.items()}
sdho = {k: v["heldout_sd"] for k, v in t1.items()}
out["seed_variability"] = {"fixed_time_val_sd_by_recipe": sdv, "pooled_fixed_time_val_sd": math.sqrt(st.mean([v ** 2 for v in sdv.values()])),
                           "panel_val_sd_by_recipe": sd201, "panel_heldout_sd_by_recipe": sdho,
                           "pooled_panel_heldout_sd": math.sqrt(st.mean([v ** 2 for v in sdho.values()])),
                           "baseline_fixed_time_seed_sd": C["baseline_seed_effects"]["cached_fresh_sd"],
                           "baseline_seed42_minus_mean_seeds0_4": C["baseline_seed_effects"]["seed42_minus_cached_fresh_mean"]}
pooled = out["seed_variability"]["pooled_panel_heldout_sd"]
out["minimum_detectable_difference_n3_paired_approx"] = {"assumed_sd_of_paired_diff": pooled * math.sqrt(2) * 0.5,
                                                         "ci95_halfwidth": T_CRIT[2] * pooled * math.sqrt(2) * 0.5 / math.sqrt(3),
                                                         "note": "rough: assumes paired-difference SD ~ sqrt(2)*pooled SD*0.5 (positive seed correlation); report as order of magnitude only"}

# Token-exposure noise under the fixed-time contract (same recipe, different sessions/seeds).
tok_cv = {}
for k, v in ft.items():
    toks = list(v["confirm_tokens_by_seed"].values()) + [v["discovery_tokens"]]
    tok_cv[k] = {"discovery_tokens": v["discovery_tokens"], "confirm_tokens": v["confirm_tokens_by_seed"], "cv_pct": 100 * st.stdev(toks) / st.mean(toks),
                 "max_rel_dev_pct": 100 * (max(toks) - min(toks)) / st.mean(toks)}
out["fixed_time_token_exposure_variation"] = tok_cv

# Selection optimism among fixed-time-confirmed candidates.
out["selection_optimism"] = {k: {"discovery": v["discovery_val"], "discovery_seed": v["discovery_seed"], "confirm_mean_seeds012": v["confirm_mean"],
                                 "optimism": v["optimism_discovery_minus_confirm_mean"]} for k, v in ft.items()}
# Expected best-of-N optimism if candidates were identical (Gaussian noise sd = pooled fixed-time val sd).
def e_max_normal(n, sims=None):
    # Blom approximation for the expected maximum of n standard normals.
    from statistics import NormalDist
    return NormalDist().inv_cdf((n - 0.375) / (n + 0.25))
sd0 = out["seed_variability"]["pooled_fixed_time_val_sd"]
out["null_best_of_N_optimism"] = {str(n): sd0 * e_max_normal(n) for n in (5, 25, 50)}

# Gaps between top discovery candidates per campaign vs noise floor.
gaps = {}
for camp in ("autopareto-seed0-20260731", "autopareto-seed1-20260731", "autopareto-seed42-20260727", "autoresearch-seed42-20260723", "autoresearch-diagnostic5-seed42-20260724", "tpe-seed42-20260805", "motpe-seed42-20260805"):
    vals = [r[0] for r in con.execute("select val_bpb from experiments where campaign_id=? and val_bpb is not null order by val_bpb limit 5", (camp,))]
    gaps[camp] = {"top5": vals, "gap_best_to_2nd": vals[1] - vals[0] if len(vals) > 1 else None, "gap_best_to_5th": vals[-1] - vals[0] if len(vals) > 4 else None}
out["top_candidate_gaps_vs_noise"] = {"pooled_seed_sd": sd0, "campaigns": gaps}

# Pareto frontier stability across seeds in the 201M panel (minimize heldout, maximize decode, minimize memory).
mem = {k: v["peak_mem_values"][0] for k, v in t1.items()}
def frontier(points):
    nd = []
    for a, (ha, da, ma) in points.items():
        dominated = any((hb <= ha and db >= da and mb <= ma) and (hb < ha or db > da or mb < ma) for b, (hb, db, mb) in points.items() if b != a)
        if not dominated:
            nd.append(a)
    return sorted(nd)
fr = {}
for s in seeds:
    pts = {k: (v["per_seed"][s]["heldout"], v["per_seed"][s]["decode"], mem[k]) for k, v in t1.items()}
    fr[str(s)] = frontier(pts)
fr["means"] = frontier({k: (v["heldout_mean"], v["decode_mean"], mem[k]) for k, v in t1.items()})
out["panel_frontier_by_seed"] = fr

# Pair-A per-group summary (descriptive).
import csv
reg = list(csv.DictReader(open(os.path.join(os.path.dirname(OUT), "..", "02_EXPERIMENT_REGISTRY.csv"))))
pa = defaultdict(list)
for r in reg:
    if r["era"] == "pair_a_contract_study":
        pa[(r["pair_a_group"], r["arm"])].append(r)
summ = {}
for (g, arm), rows in sorted(pa.items()):
    rows.sort(key=lambda r: int(r["attempt"]))
    comp = [r for r in rows if r["val_bpb"] not in ("", "UNKNOWN")]
    batches = [(int(r["attempt"]), r["total_batch_size"]) for r in rows if r["total_batch_size"] not in ("", "UNKNOWN")]
    first = batches[0][1] if batches else None
    changes = []
    prev = None
    for a, b in batches:
        if prev is not None and b != prev:
            changes.append((a, prev, b))
        prev = b
    summ[f"{g}|{arm}"] = {"attempts": len(rows), "completed_with_bpb": len(comp),
                          "best_bpb": min((float(r["val_bpb"]) for r in comp), default=None),
                          "n_bpb_above_3": sum(float(r["val_bpb"]) > 3 for r in comp),
                          "batch_sizes_in_order": [b for _, b in batches], "batch_changes(attempt,from,to)": changes,
                          "tokens_set": sorted({r["total_training_tokens"] for r in rows if r["total_training_tokens"] not in ("", "UNKNOWN")})[:3],
                          "agent": sorted({r["agent_model"] + "/" + r["reasoning_level"] for r in rows})}
out["pair_a_group_summary"] = summ
json.dump(out, open(OUT, "w"), indent=2, default=str)

print("matched heldout", {k: v for k, v in out["matched_token_AP292_minus_AR"]["heldout"].items() if k != "diffs"})
print("matched val", {k: v for k, v in out["matched_token_AP292_minus_AR"]["validation"].items() if k != "diffs"})
print("fixed-time val", {k: v for k, v in out["fixed_time_AP292_minus_AR_val_seeds012"].items() if k != "diffs"})
print("decode ratio by seed", out["matched_token_AP292_minus_AR"]["decode_ratio_by_seed"])
for r in out["reversal_table"]: print("REV", r["comparison"], r["eval"], round(r["delta_AP292_minus_AR"], 5))
for r in resp: print("EXPO", r["recipe"], round(r["fixed_time_val_mean_seeds012"], 4), round(r["fixed_time_tokens_mean"] / 1e6, 1), "->", round(r["panel_val_mean_seeds345"], 4), "ratio", round(r["token_ratio_panel_over_fixed"], 3), "delta", round(r["delta_val"], 4))
print("expo pearson", round(out["exposure_response_pearson"], 3))
print("seed var", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out["seed_variability"].items() if not isinstance(v, dict)})
print("mdd", out["minimum_detectable_difference_n3_paired_approx"])
print("token cv", {k: round(v["max_rel_dev_pct"], 1) for k, v in tok_cv.items()})
print("optimism", {k: round(v["optimism"], 5) for k, v in out["selection_optimism"].items()}, "null best-of-N", out["null_best_of_N_optimism"])
print("gaps", {k: (round(v["gap_best_to_2nd"], 5) if v["gap_best_to_2nd"] else None, round(v["gap_best_to_5th"], 5) if v["gap_best_to_5th"] else None) for k, v in gaps.items()})
print("frontiers", fr)
for k, v in summ.items(): print("PAIRA", k, v["attempts"], v["completed_with_bpb"], v["best_bpb"], "above3:", v["n_bpb_above_3"], "changes:", v["batch_changes(attempt,from,to)"], v["tokens_set"], v["agent"])
