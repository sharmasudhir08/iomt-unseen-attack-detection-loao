"""T2: operating-point metrics for energy, MSP, MaxLogit (FPR@95, OSCR, AUPR, TPR@FPR1%) and temperature grids (energy T 0.5, 2, 5; MSP T 0.5, 1, 2, 5). Exploratory, post hoc.
No temperature is selected or recommended. OSCR follows Dhamija et al. (author to supply a verified citation before use in the paper)."""
import sys, time
from pathlib import Path
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_curve, auc, average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_rev_common import *
t0 = time.time()
d = task_dir("T", 2)


def curve(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    ok95 = tpr >= 0.95
    ok1 = fpr <= 0.01
    return float(auc(fpr, tpr)), float(fpr[ok95].min()) if ok95.any() else 1.0, float(tpr[ok1].max()) if ok1.any() else 0.0


def oscr(conf, correct, unk):
    """conf: higher = more known. CCR = known and correctly classified and accepted, over all known; FPR = held-out accepted, over all held-out."""
    o = np.argsort(-conf, kind="stable")
    k_ok = (~unk[o]) & correct[o]
    u = unk[o]
    ccr = np.concatenate([[0.0], np.cumsum(k_ok) / max((~unk).sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(u) / max(u.sum(), 1)])
    return float(np.trapezoid(ccr, fpr))


rows, rowsT = [], []
for m, c, s, name in canonical_runs():
    z = load_export(name)
    y19, hid = z["y19"].astype(int), int(z["hid"])
    unk = y19 == hid
    y = unk.astype(np.int8)
    lg = z["logits19"].astype(np.float32)
    lk = lg.copy()
    lk[:, hid] = -np.inf
    correct = lk.argmax(1) == y19
    r = dict(model=m, held_out_class=c, seed=s)
    for nm, sc, conf in (("energy", -z["energy"], z["energy"]), ("msp", -z["msp"], z["msp"]), ("maxlogit", -z["maxlogit"], z["maxlogit"])):
        a, f95, t1 = curve(y, sc)
        r[f"{nm}_auroc"], r[f"{nm}_fpr95"], r[f"{nm}_tpr_at_fpr1"] = a, f95, t1
        r[f"{nm}_aupr"] = float(average_precision_score(y, sc))
        r[f"{nm}_oscr"] = oscr(conf, correct, unk)
    for T in (0.5, 2.0, 5.0):
        a, f95, _ = curve(y, -(T * logsumexp(lg / T, axis=1)))
        r[f"energy_T{T:g}_auroc"], r[f"energy_T{T:g}_fpr95"] = a, f95
    for T in (0.5, 2.0, 5.0):
        a, f95, _ = curve(y, -softmax(lg / T, axis=1).max(1))
        r[f"msp_T{T:g}_auroc"], r[f"msp_T{T:g}_fpr95"] = a, f95
    rows.append(r)
    print("done", name, f"{time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
pr = per_run_table().set_index(["model", "held_out_class", "seed"])
x = df.set_index(["model", "held_out_class", "seed"])
dev = {"energy T1": (x.energy_auroc - pr.A7_energy_auroc).abs().max(), "msp T1": (x.msp_auroc - pr.A7_msp_auroc).abs().max()}
for T in ("0.5", "2", "5"):
    dev[f"energy T{T}"] = (x[f"energy_T{T}_auroc"] - pr[f"A18_energy_T{T}_auroc"]).abs().max()
cmd = "python scripts\p4_rev_2_operating_points.py"
if max(dev.values()) > 1e-9:
    df.to_csv(d / "per_run_operating_points_UNVERIFIED.csv", index=False)
    fail_readme(d, "T2 operating points", __file__, cmd, f"reproduction failed: {dev}", t0)
    sys.exit(1)
df.to_csv(d / "per_run_operating_points.csv", index=False)
cols = [c for c in df.columns if c not in ("model", "held_out_class", "seed")]
sm = seed_ci(df, cols, by=("model", "held_out_class"))
sm.to_csv(d / "summary_operating_points.csv", index=False)
for k in ("auroc", "fpr95", "oscr"):
    df[f"d_{k}_energy_minus_msp"] = df[f"energy_{k}"] - df[f"msp_{k}"]
pdf = seed_ci(df, [f"d_{k}_energy_minus_msp" for k in ("auroc", "fpr95", "oscr")], by=("model", "held_out_class"))
pdf.to_csv(d / "paired_energy_minus_msp.csv", index=False)
tl = []
for T, key in ((0.5, "T0.5"), (1.0, ""), (2.0, "T2"), (5.0, "T5")):
    e = f"energy_{key}_auroc" if key else "energy_auroc"
    ms = f"msp_{key}_auroc" if key else "msp_auroc"
    g = df.assign(energy=df[e], msp=df[ms]).groupby(["model", "held_out_class"])[["energy", "msp"]].mean().reset_index().assign(T=T)
    tl.append(g)
pd.concat(tl).rename(columns={"energy": "energy_auroc_mean", "msp": "msp_auroc_mean"}).to_csv(d / "temperature_energy_msp.csv", index=False)
tex = df.groupby(["model", "held_out_class"])[["energy_fpr95", "msp_fpr95", "maxlogit_fpr95"]].mean().reset_index().round(4).to_latex(index=False, escape=True)
(d / "fpr95_table.tex").write_text(tex, encoding="utf-8")
pool = df.groupby("model")[["energy_auroc", "msp_auroc", "energy_fpr95", "msp_fpr95", "energy_oscr", "msp_oscr", "energy_aupr", "msp_aupr"]].mean().round(4)
res = [f"Reproduction (max |difference| against per_run.csv A7/A18): {({k: float(v) for k, v in dev.items()})}; all <= 1e-9: PASS.", "Temperature grids fixed now: energy T in {0.5, 1, 2, 5}, MSP T in {0.5, 1, 2, 5}; none is selected or recommended. FPR@95 = smallest FPR on known windows at TPR >= 0.95 on held-out windows.", "", "Mean over classes and seeds:", "```", pool.to_string(), "```"]
write_readme(d, "T2: operating points and temperature grids", __file__, cmd, "score_export/*.npz; saved_output_analyses_20261118/per_run.csv", 75, "\n".join(res), t0)
print("T2 done", f"{time.time()-t0:.0f}s", dev)
