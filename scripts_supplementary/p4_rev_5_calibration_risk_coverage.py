"""T5: calibration (ECE, 15 equal-width bins fixed now) and risk-coverage curves. Exploratory, post hoc."""
import sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_rev_common import *
t0 = time.time()
d = task_dir("T", 5)
(d / "figures").mkdir(exist_ok=True)
NB = 15


def ece_bins(conf, correct):
    edges = np.linspace(0, 1, NB + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, NB - 1)
    n = len(conf)
    cnt = np.bincount(idx, minlength=NB).astype(float)
    sc = np.bincount(idx, weights=conf, minlength=NB)
    sa = np.bincount(idx, weights=correct.astype(float), minlength=NB)
    ok = cnt > 0
    e = float((np.abs(sa[ok] / cnt[ok] - sc[ok] / cnt[ok]) * cnt[ok] / n).sum())
    return e, cnt, sc, sa


def risk_curve(conf, bad):
    o = np.argsort(-conf, kind="stable")
    w = bad[o].astype(np.float64)
    risk = np.cumsum(w) / np.arange(1, len(w) + 1)
    return risk


grid = np.linspace(0.01, 1.0, 100)
rows, rel19, rc_acc = [], {m: np.zeros((NB, 3)) for m in MODELS}, {m: {sc: [] for sc in ("energy", "msp")} for m in MODELS}
maxdev = 0.0
for m, c, s, name in canonical_runs():
    z = load_export(name)
    p = np.load(ZD / name / "predictions.npz")
    y19, hid = z["y19"].astype(int), int(z["hid"])
    if not np.array_equal(p["y_true_label19"].astype(int), y19):
        fail_readme(d, "T5 calibration", __file__, "python scripts/p4_rev_5_calibration_risk_coverage.py", f"label order differs between predictions.npz and score_export ({name})", t0)
        print("STOP label order", name)
        sys.exit(1)
    unk = y19 == hid
    kn = ~unk
    pred = p["y_pred_label19"].astype(int)
    p19 = p["proba_label19"]
    conf19 = p19.max(1)
    ece19, cnt, sc, sa = ece_bins(conf19[kn], (pred[kn] == y19[kn]))
    rel19[m] += np.stack([cnt, sc, sa], 1)
    y2 = p["y_true_label2"].astype(int)
    pa = p["proba_label2"][:, 1]
    conf2 = np.maximum(pa, 1 - pa)
    ece2, _, _, _ = ece_bins(conf2[kn], (p["y_pred_label2"][kn] == y2[kn]))
    bad = unk | (pred != y19)
    r = dict(model=m, held_out_class=c, seed=s, ece_19class_known=ece19, ece_binary_known=ece2, risk_at_full_coverage=float(bad.mean()), indep_risk=float((unk.sum() + (kn & (pred != y19)).sum()) / len(y19)))
    maxdev = max(maxdev, abs(r["risk_at_full_coverage"] - r["indep_risk"]))
    for nm, conf in (("energy", z["energy"]), ("msp", z["msp"])):
        rk = risk_curve(conf, bad)
        r[f"aurc_{nm}"] = float(rk.mean())
        r[f"risk_at_full_{nm}"] = float(rk[-1])
        maxdev = max(maxdev, abs(rk[-1] - r["indep_risk"]))
        idx = (grid * len(rk)).astype(int) - 1
        rc_acc[m][nm].append(rk[idx])
    rows.append(r)
    print("done", name, f"{time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
cmd = "python scripts/p4_rev_5_calibration_risk_coverage.py"
if maxdev > 1e-9:
    fail_readme(d, "T5 calibration", __file__, cmd, f"risk at coverage 1.0 differs from the independent value by {maxdev:.2e}", t0)
    sys.exit(1)
df.to_csv(d / "ece_aurc_per_run.csv", index=False)
agg = seed_ci(df, ["ece_19class_known", "ece_binary_known", "aurc_energy", "aurc_msp"], by=("model", "held_out_class"))
agg.to_csv(d / "ece_aurc_summary.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for i, m in enumerate(MODELS):
    cnt, sc, sa = rel19[m][:, 0], rel19[m][:, 1], rel19[m][:, 2]
    ok = cnt > 0
    ax[i].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[i].plot(sc[ok] / cnt[ok], sa[ok] / cnt[ok], "o-", ms=3)
    ax[i].set_title(f"{m}: 19-class reliability, known test windows, pooled", fontsize=8)
    ax[i].set_xlabel("mean confidence")
    ax[i].set_ylabel("accuracy")
plt.tight_layout()
fig.savefig(d / "figures" / "reliability_19class.pdf")
fig2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
for i, m in enumerate(MODELS):
    for nm in ("energy", "msp"):
        ax2[i].plot(grid, np.mean(rc_acc[m][nm], axis=0), label=nm)
    ax2[i].set_title(f"{m}: risk-coverage (risk = held-out or misclassified known), mean of 25 runs", fontsize=7)
    ax2[i].set_xlabel("coverage")
    ax2[i].set_ylabel("risk")
    ax2[i].legend(fontsize=7)
plt.tight_layout()
fig2.savefig(d / "figures" / "risk_coverage.pdf")
pool = df.groupby("model")[["ece_19class_known", "ece_binary_known", "aurc_energy", "aurc_msp"]].mean().round(4)
res = [f"Pass criterion: risk at coverage 1.0 equals (held-out + misclassified known) / total computed independently (max |difference| {maxdev:.1e}); label order of predictions.npz equals score_export y19 for all runs.", f"ECE uses {NB} equal-width bins (fixed now) on known test windows; AURC treats a held-out window and a misclassified known window as errors; lower is better.", "", "Mean over classes and seeds:", "```", pool.to_string(), "```", "", "Figures: figures/reliability_19class.pdf, figures/risk_coverage.pdf."]
write_readme(d, "T5: calibration and risk-coverage", __file__, cmd, "score_export/*.npz; zero_day/<run>/predictions.npz", 75, "\n".join(res), t0)
print("T5 done", f"{time.time()-t0:.0f}s", maxdev)
