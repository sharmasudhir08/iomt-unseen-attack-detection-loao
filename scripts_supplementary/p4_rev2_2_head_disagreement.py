"""U2: disagreement between the 2-, 6- and 19-class heads as an unknown-attack score. Exploratory, post hoc.
Score A = KL(aggregated p19 over the 6 categories || p6) + KL(aggregated p6 over the 2 groups || p2); Score B = 1 if the argmax labels are inconsistent across the heads.
Fixed combination rule: flag if the energy score is flagged (below the run's threshold) OR score B = 1."""
import sys, time
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_rev_common import *
t0 = time.time()
d = task_dir("U", 2)
pr = per_run_table().set_index(["model", "held_out_class", "seed"])


def agg(p, parent, n_out):
    out = np.zeros((p.shape[0], n_out), dtype=np.float64)
    for k, g in enumerate(parent):
        out[:, g] += p[:, k]
    return out


def kl(a, b):
    a = np.clip(a, 1e-12, None)
    b = np.clip(b, 1e-12, None)
    return (a * (np.log(a) - np.log(b))).sum(1)


rows = []
maxdev = 0.0
for m, c, s, name in canonical_runs():
    z = load_export(name)
    p = np.load(ZD / name / "predictions.npz")
    mj = json.loads((ZD / name / "metrics.json").read_text(encoding="utf-8"))["hierarchy"]
    par19, par6 = np.array(mj["parent19"]), np.array(mj["parent6"])
    y19, hid = z["y19"].astype(int), int(z["hid"])
    unk = y19 == hid
    y = unk.astype(np.int8)
    p19, p6, p2 = p["proba_label19"].astype(np.float64), p["proba_label6"].astype(np.float64), p["proba_label2"].astype(np.float64)
    a6 = agg(p19, par19, 6)
    a2 = agg(p6, par6, 2)
    scoreA = kl(a6, p6) + kl(a2, p2)
    a19, a6m, a2m = p19.argmax(1), p6.argmax(1), p2.argmax(1)
    B = ((par19[a19] != a6m) | (par6[a6m] != a2m))
    en, thr = z["energy"], float(z["thr"])
    flag_en = en < thr
    or_flag = flag_en | B
    auc_en = float(roc_auc_score(y, -en))
    maxdev = max(maxdev, abs(auc_en - float(pr.loc[(m, c, s), "A7_energy_auroc"])))
    r = dict(model=m, held_out_class=c, seed=s, auroc_energy=auc_en, auroc_scoreA=float(roc_auc_score(y, scoreA)), auroc_scoreB=float(roc_auc_score(y, B.astype(np.int8))),
             fpr95_scoreA=fpr_at_tpr(y, scoreA, 0.95), frac_inconsistent_heldout=float(B[unk].mean()), frac_inconsistent_known=float(B[~unk].mean()),
             caught_energy=float(flag_en[unk].mean()), far_energy=float(flag_en[~unk].mean()), caught_or=float(or_flag[unk].mean()), far_or=float(or_flag[~unk].mean()))
    rows.append(r)
    print("done", name, f"{time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
cmd = "python scripts/p4_rev2_2_head_disagreement.py"
if maxdev > 1e-6:
    fail_readme(d, "U2 head disagreement", __file__, cmd, f"energy AUROC differs from the paper value by {maxdev:.2e}", t0)
    sys.exit(1)
df.to_csv(d / "head_disagreement_per_run.csv", index=False)
df["d_auroc_scoreA_minus_energy"] = df.auroc_scoreA - df.auroc_energy
df["d_auroc_scoreB_minus_energy"] = df.auroc_scoreB - df.auroc_energy
cols = [c for c in df.columns if c not in ("model", "held_out_class", "seed")]
sm = seed_ci(df, cols, by=("model", "held_out_class"))
sm.to_csv(d / "head_disagreement_auroc.csv", index=False)
orr = sm[["model", "held_out_class", "caught_energy", "caught_energy_lo", "caught_energy_hi", "far_energy", "caught_or", "caught_or_lo", "caught_or_hi", "far_or", "far_or_lo", "far_or_hi", "frac_inconsistent_heldout", "frac_inconsistent_known"]]
orr.to_csv(d / "head_disagreement_or_rule.csv", index=False)
pool = df.groupby("model")[["auroc_energy", "auroc_scoreA", "auroc_scoreB", "frac_inconsistent_heldout", "frac_inconsistent_known", "caught_energy", "caught_or", "far_energy", "far_or"]].mean().round(4)
res = [f"Pass criterion: the energy AUROC recomputed from score_export equals the paper value (A7) within 1e-6 (max |difference| {maxdev:.1e}): PASS.", "Score B = argmax labels inconsistent across the three heads (19 to 6 category, 6 to 2 group). The OR rule adds this flag to the energy flag; the known false-alarm rate rises accordingly (not tuned).", "", "Mean over classes and seeds:", "```", pool.to_string(), "```"]
write_readme(d, "U2: head disagreement as an unknown score", __file__, cmd, "zero_day/<run>/predictions.npz (proba_label2/6/19), score_export/*.npz, hierarchy in metrics.json", 75, "\n".join(res), t0)
print("U2 done", f"{time.time()-t0:.0f}s", maxdev)
