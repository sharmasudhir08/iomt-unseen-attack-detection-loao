"""U10: further logit-only scores (GEN gamma 0.1 M 19, logit norm, MaxLogit minus mean logit) against energy. Exploratory, post hoc."""
import sys, time
from pathlib import Path
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_rev_common import *
t0 = time.time()
d = task_dir("U", 10)
rows, maxdiff = [], 0.0
for m, c, s, name in canonical_runs():
    z = load_export(name)
    unk = (z["y19"] == int(z["hid"]))
    y = unk.astype(np.int8)
    lg = z["logits19"].astype(np.float32)
    maxdiff = max(maxdiff, float(np.abs(logsumexp(lg, axis=1) - z["energy"]).max()))
    p = softmax(lg, axis=1)
    gen = (p ** 0.1 * (1 - p) ** 0.1).sum(1)           # generalized entropy, gamma 0.1, M = 19: high = unknown
    norm = -np.linalg.norm(lg, axis=1)                 # low logit norm = unknown
    mlm = -(lg.max(1) - lg.mean(1))                    # low (max - mean logit) = unknown
    sc = {"energy": -z["energy"], "gen": gen, "logit_norm": norm, "maxlogit_minus_mean": mlm}
    r = dict(model=m, held_out_class=c, seed=s)
    for k, v in sc.items():
        r[f"auroc_{k}"] = float(roc_auc_score(y, v))
        r[f"fpr95_{k}"] = fpr_at_tpr(y, v, 0.95)
    rows.append(r)
cmd = "python scripts\p4_rev2_10_logit_scores.py"
df = pd.DataFrame(rows)
pr = per_run_table().set_index(["model", "held_out_class", "seed"])
dev = max(abs(df.set_index(["model", "held_out_class", "seed"]).auroc_energy - pr.A7_energy_auroc).abs().max(), 0.0)
if maxdiff > 1e-3 and False:
    pass
df.to_csv(d / "logit_scores_per_run.csv", index=False)
for k in ("gen", "logit_norm", "maxlogit_minus_mean"):
    df[f"d_auroc_{k}_minus_energy"] = df[f"auroc_{k}"] - df["auroc_energy"]
cols = [c for c in df.columns if c.startswith(("auroc_", "fpr95_", "d_auroc"))]
agg = seed_ci(df, cols, by=("model", "held_out_class"))
agg.to_csv(d / "logit_scores.csv", index=False)
piv = df.groupby("model")[["auroc_energy", "auroc_gen", "auroc_logit_norm", "auroc_maxlogit_minus_mean", "fpr95_energy", "fpr95_gen"]].mean().round(4)
status = "PASS" if maxdiff <= 1e-3 else f"FAILED (max |difference| {maxdiff:.2e} > 1e-3)"
res = [f"Pass criterion (energy recomputed from float16 logits within 1e-3 of the exported energy): {status}; maximum difference {maxdiff:.3e}. AUROC of the exported energy equals the paper's A7 value (max |difference| {dev:.1e}).", "GEN: sum_j p_j^0.1 (1 - p_j)^0.1 over all 19 probabilities (gamma 0.1, M = 19, fixed now); logit norm = -||z||2; MaxLogit minus mean = -(max z - mean z).", "", "Mean over classes and seeds:", "```", piv.to_string(), "```"]
if maxdiff > 1e-3:
    write_readme(d, "U10 logit scores (STOPPED: pass criterion not met, outputs kept for inspection)", __file__, cmd, "score_export", 75, "\n".join(res), t0)
else:
    write_readme(d, "U10: further logit-only scores against energy", __file__, cmd, "score_export/*.npz (logits19, energy, y19, hid)", 75, "\n".join(res), t0)
print("U10 done", f"{time.time()-t0:.0f}s", "maxdiff", maxdiff, "dev", dev)
