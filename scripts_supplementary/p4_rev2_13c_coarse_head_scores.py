"""U13 analysis from the float32 re-export (score_export_coarse_<date>): MSP and energy of the 2-, 6- and 19-class heads as unknown-attack scores. Exploratory, post hoc.
Unknown score: -MSP (MSP = max softmax probability, float32) and -energy (energy = logsumexp of the head's logits, higher = more known). Held-out = y19 == hid; known = all other test windows.
Per run and head: AUROC (ties averaged, scikit-learn), FPR at 95% TPR, share of windows whose MSP equals 1.0 (held-out and known separately).
Paired seed-bootstrap differences (paired_diff_ci, 10,000 resamples, rng 42) to the 19-class energy AUROC and, for MSP and energy, of the 2- and 6-class head to the 19-class head.
Pass criteria: C1 energy AUROC from score_export equals A7_energy_auroc within 1e-6; C2 the 19-class MSP AUROC from the new float32 export equals A7_msp_auroc within 1e-6
(the float32 export of this run also reproduces score_export energy and msp exactly: see the consistency record of each npz). Stops if a criterion fails.
Usage: python scripts/p4_rev2_13c_coarse_head_scores.py --date 20261123"""
import argparse, sys, time
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_rev_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); a = ap.parse_args()
t0 = time.time()
d = RS / f"p4_review_analyses_{a.date}" / "part2" / "U13"
CX = RS / f"score_export_coarse_{a.date}"
assert d.exists() and CX.exists()
for f in ("coarse_head_scores_per_run.csv", "coarse_head_scores_summary.csv"):
    assert not (d / f).exists(), f"exists: {f}"
if (d / "README.md").exists():   # the README of the stopped first attempt is kept under a new name
    assert not (d / "README_attempt1_STOPPED.md").exists()
    (d / "README.md").rename(d / "README_attempt1_STOPPED.md")
pr = per_run_table().set_index(["model", "held_out_class", "seed"])
H = {"2": "2-class", "6": "6-class", "19": "19-class"}
rows = []; dev_e = 0.0; dev_m = 0.0; ex_e = 0.0; ex_m = 0.0; flat = []
for m, c, s, name in canonical_runs():
    z = load_export(name); x = np.load(CX / f"{name}.npz", allow_pickle=False)
    assert (x["y19"].astype(int) == z["y19"].astype(int)).all() and int(x["hid"]) == int(z["hid"]), name
    y19, hid = z["y19"].astype(int), int(z["hid"]); unk = y19 == hid; y = unk.astype(np.int8)
    aucE_export = float(roc_auc_score(y, -z["energy"].astype(np.float64)))
    dev_e = max(dev_e, abs(aucE_export - float(pr.loc[(m, c, s), "A7_energy_auroc"])))
    r = dict(model=m, held_out_class=c, seed=s, n_heldout=int(unk.sum()), n_known=int((~unk).sum()))
    for h in H:
        msp = x[f"msp{h}"].astype(np.float64); en = x[f"energy{h}"].astype(np.float64)
        r[f"auroc_msp_h{h}"] = float(roc_auc_score(y, -msp)); r[f"fpr95_msp_h{h}"] = fpr_at_tpr(y, -msp, 0.95)
        r[f"tie_msp_heldout_h{h}"] = float((msp[unk] == 1.0).mean()); r[f"tie_msp_known_h{h}"] = float((msp[~unk] == 1.0).mean())
        r[f"msp_tie_dominated_h{h}"] = bool(r[f"tie_msp_heldout_h{h}"] >= 0.5 or r[f"tie_msp_known_h{h}"] >= 0.5)
        if np.isfinite(en).all():
            r[f"auroc_energy_h{h}"] = float(roc_auc_score(y, -en)); r[f"fpr95_energy_h{h}"] = fpr_at_tpr(y, -en, 0.95)
        else:
            r[f"auroc_energy_h{h}"] = np.nan; r[f"fpr95_energy_h{h}"] = np.nan; flat.append((m, h))
    dev_m = max(dev_m, abs(r["auroc_msp_h19"] - float(pr.loc[(m, c, s), "A7_msp_auroc"])))
    ex_e = max(ex_e, abs(r["auroc_energy_h19"] - float(pr.loc[(m, c, s), "A7_energy_auroc"])))
    for h in H:
        r[f"d_msp_h{h}_minus_energy_h19"] = r[f"auroc_msp_h{h}"] - r["auroc_energy_h19"]
        r[f"d_energy_h{h}_minus_energy_h19"] = r[f"auroc_energy_h{h}"] - r["auroc_energy_h19"]
    for h in ("2", "6"):
        r[f"d_msp_h{h}_minus_msp_h19"] = r[f"auroc_msp_h{h}"] - r["auroc_msp_h19"]
        r[f"d_energy_h{h}_minus_energy_h19_pair"] = r[f"auroc_energy_h{h}"] - r["auroc_energy_h19"]
    rows.append(r); print("done", name, f"{time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows); cmd = f"python scripts/p4_rev2_13c_coarse_head_scores.py --date {a.date}"
ok1, ok2 = dev_e <= 1e-6, dev_m <= 1e-6
print(f"C1 energy AUROC (score_export) vs A7: max|diff| {dev_e:.3e} {'PASS' if ok1 else 'FAIL'}; C2 float32-export MSP19 AUROC vs A7: {dev_m:.3e} {'PASS' if ok2 else 'FAIL'}; export energy19 AUROC vs A7: {ex_e:.3e}")
if not (ok1 and ok2):
    fail_readme(d, "U13 coarse heads", __file__, cmd, f"pass criterion failed: C1 {dev_e:.2e}, C2 {dev_m:.2e} (limit 1e-6)", t0); sys.exit(1)
df.to_csv(d / "coarse_head_scores_per_run.csv", index=False)
met = [c for c in df.columns if not c.startswith("d_") and c not in ("model", "held_out_class", "seed") and not c.startswith("msp_tie_dominated")]
sm = seed_ci(df, met)
pairs = [("auroc_msp_h2", "auroc_energy_h19"), ("auroc_msp_h6", "auroc_energy_h19"), ("auroc_msp_h19", "auroc_energy_h19"), ("auroc_energy_h2", "auroc_energy_h19"), ("auroc_energy_h6", "auroc_energy_h19"),
         ("auroc_msp_h2", "auroc_msp_h19"), ("auroc_msp_h6", "auroc_msp_h19")]
for p_, q_ in pairs:
    t = df.dropna(subset=[p_, q_]); sm = sm.merge(paired_diff_ci(t, p_, q_).drop(columns="n_seeds"), on=["model", "held_out_class"], how="left")
for h in H:
    sm[f"msp_tie_dominated_h{h}"] = [bool(sm_r[f"tie_msp_heldout_h{h}"] >= 0.5 or sm_r[f"tie_msp_known_h{h}"] >= 0.5) for _, sm_r in sm.iterrows()]
mo = {k: i for i, k in enumerate(MODELS)}; co = {k: i for i, k in enumerate(CLASSES)}
sm = sm.sort_values(["model", "held_out_class"], key=lambda col: col.map(mo if col.name == "model" else co)).reset_index(drop=True)
pool = df.groupby("model")[met].mean().reset_index(); pool.insert(1, "held_out_class", "pooled (25 runs)"); pool.insert(2, "n_seeds", 25)
pd.concat([sm, pool], ignore_index=True).to_csv(d / "coarse_head_scores_summary.csv", index=False)

def f4(v): return "n/a" if not np.isfinite(v) else f"{v:.4f}"
def ci_(r, k): return f"{f4(r[k])} [{f4(r[k + '_lo'])}, {f4(r[k + '_hi'])}]"
L = ["| Model | Held-out class | MSP AUROC 2 / 6 / 19-class | MSP ties held-out/known 2 | 6 | 19 | Energy AUROC 2 / 6 / 19-class |", "|---|---|---|---|---|---|---|"]
for _, r in sm.iterrows():
    msp = " / ".join(f4(r[f"auroc_msp_h{h}"]) + ("*" if r[f"msp_tie_dominated_h{h}"] else "") for h in H)
    ties = [f"{r[f'tie_msp_heldout_h{h}']:.2f}/{r[f'tie_msp_known_h{h}']:.2f}" for h in H]
    en = " / ".join(f4(r[f"auroc_energy_h{h}"]) for h in H)
    L.append(f"| {r.model} | {r.held_out_class} | {msp} | {ties[0]} | {ties[1]} | {ties[2]} | {en} |")
D = ["| Model | Held-out class | MSP2 - E19 [CI] | MSP6 - E19 [CI] | Energy2 - E19 [CI] | Energy6 - E19 [CI] |", "|---|---|---|---|---|---|"]
for _, r in sm.iterrows():
    D.append(f"| {r.model} | {r.held_out_class} | {ci_(r, 'auroc_msp_h2_minus_auroc_energy_h19')} | {ci_(r, 'auroc_msp_h6_minus_auroc_energy_h19')} | {ci_(r, 'auroc_energy_h2_minus_auroc_energy_h19')} | {ci_(r, 'auroc_energy_h6_minus_auroc_energy_h19')} |")
FL = []
for _, r in sm[sm.held_out_class.isin(["TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"])].iterrows():
    for sc, k in (("MSP", "auroc_msp"), ("energy", "auroc_energy")):
        for h in ("2", "6"):
            kk = f"{k}_h{h}_minus_auroc_energy_h19"; lo, hi = r[kk + "_lo"], r[kk + "_hi"]
            if not np.isfinite(r[kk]): FL.append(f"- {r.model} / {r.held_out_class} / {H[h]} {sc}: not available (this model has no separate {H[h]} head logits)"); continue
            tag = "interval above 0" if lo > 0 else "interval below 0" if hi < 0 else "interval includes 0"
            tie = f"; MSP tie shares held-out/known {r[f'tie_msp_heldout_h{h}']:.2f}/{r[f'tie_msp_known_h{h}']:.2f}" + (" (tie-dominated)" if sc == "MSP" and r[f"msp_tie_dominated_h{h}"] else "") if sc == "MSP" else ""
            FL.append(f"- {r.model} / {r.held_out_class} / {H[h]} {sc} minus 19-class energy: {r[kk]:+.4f} [{lo:+.4f}, {hi:+.4f}] ({tag}){tie}")
res = [f"Pass criterion C1: energy AUROC recomputed from score_export equals A7_energy_auroc within 1e-6: max |difference| {dev_e:.3e}: PASS.",
       f"Pass criterion C2: the 19-class MSP AUROC recomputed from the new float32 export equals A7_msp_auroc within 1e-6: max |difference| {dev_m:.3e}: PASS (the 19-class energy AUROC of the new export differs from A7 by at most {ex_e:.3e}).", "",
       "Data: inference-only re-export of the saved models (author-approved 23 Nov 2026), float32 head outputs, mixed precision as in the original export; `score_export_coarse_" + a.date + "/`. Models without separate 2- and 6-class logits (if any) have no energy for those heads; their MSP uses the summed child probabilities.",
       "Method: held-out = y19 == hid, known = all other test windows; unknown score = -MSP or -energy; AUROC with ties averaged; tie share = fraction of windows with MSP exactly 1.0 (float32), held-out/known separately. Intervals: mean over five seeds, percentile bootstrap 95% (10,000 resamples, rng 42); differences paired over seeds (paired_diff_ci). Rule fixed now: an MSP cell is marked tie-dominated (`*`, `msp_tie_dominated_h*`) if either tie share is 0.5 or more.", "",
       "AUROC per head (means over five seeds; * = MSP tie-dominated):", ""] + L + ["", "Paired differences to the 19-class energy AUROC (seed bootstrap):", ""] + D + ["", "Flood classes:", ""] + FL
extra = ("Float32 export: MSP values are float32, so the float16 rounding of the stored probabilities does not apply; ties at exactly 1.0 remain where the softmax saturates in float32, and the tie share is reported next to each MSP AUROC. "
         "Exploratory and post hoc; H1 to H5 and their verdicts do not change. Nothing was trained.")
write_readme(d, "U13: MSP and energy from the 2-, 6- and 19-class heads", __file__, cmd, f"score_export_coarse_{a.date}/*.npz, score_export/*.npz, saved_output_analyses_20261118/per_run.csv", 75, "\n".join(res), t0, extra)
print("U13 done", f"{time.time() - t0:.0f}s", "flat heads without logits:", sorted(set(flat)))
