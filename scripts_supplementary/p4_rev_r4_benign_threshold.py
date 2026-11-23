"""R4 (author-approved 23 Nov 2026): Benign-calibrated energy thresholds from saved scores. Exploratory, post hoc; H1-H5 and verdicts unchanged. Saved files only: no training, no inference.
DESIGN (fixed before any test result was seen):
1. Inputs: canonical 75 runs; score_export/<run>.npz (energy, p_attack, thr, val_energy, y19, hid); zero_day/<run>/predictions.npz (val_y_true_label19); label19 index of Benign checked.
2. Thresholds per run, from VALIDATION windows only: reference thr_all = 5th percentile of all val_energy (must equal the saved thr); Benign-calibrated thr_bq = q-th percentile of val_energy on
   Benign validation windows, q in {1, 2, 5}. A window is flagged when energy < threshold. Safety-view alert = energy flag OR p_attack > 0.5.
3. Test metrics per run and threshold: Benign energy-flag rate; Benign safety-view alert rate; held-out caught by energy only; held-out caught-any (safety view); flag rate on known attack windows;
   overall known false-alarm rate.
4. Event level: the T9 event rule (scripts/p4_rev_9_event_level.py, unchanged except the threshold) with rules 2 of 4, 8 of 16 and 32 of 64 per capture file (file = contiguous label block of the test
   order, as in T9): Benign events per 10,000 Benign windows, held-out windows triggering among those with a full history, held-out files detected.
5. Summaries: mean with seed interval per model and class (seed_ci, 10,000 resamples, rng 42) and pooled means per model.
Checks (stop if any fails): C1 thr_all equals the saved thr within 1e-6 for all 75 runs; C2 val_energy and val_y_true_label19 have the same length (the window order is confirmed from the code, see README);
C3 at thr_all the held-out caught-any equals A2_caught_any (per_run.csv) within 1e-6 and the event-level values equal the saved 20 Nov T9 per-run values.
Usage: python scripts/p4_rev_r4_benign_threshold.py --date 20261123"""
import argparse, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_rev_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); a = ap.parse_args()
t0 = time.time()
d = RS / f"p4_review_analyses_{a.date}" / "R4"; d.mkdir(parents=True, exist_ok=False)
pr = per_run_table().set_index(["model", "held_out_class", "seed"])
T9OLD = pd.read_csv(OUT / "T9" / "event_level_per_run.csv").set_index(["model", "held_out_class", "seed", "n", "m"])
RULES = [(4, 2), (16, 8), (64, 32)]
QS = [1, 2, 5]; TH = ["reference"] + [f"benign_q{q}" for q in QS]


def segments(y):
    ch = np.flatnonzero(np.diff(y) != 0) + 1
    return list(zip(np.r_[0, ch], np.r_[ch, len(y)]))


def triggers(al, n, m):
    cs = np.concatenate([[0], np.cumsum(al.astype(np.int64))]); cnt = cs[n:] - cs[:-n]
    t = np.zeros(len(al), dtype=bool); t[n - 1:] = cnt >= m
    return t


def events(t): return int((t & ~np.r_[False, t[:-1]]).sum())


rows, ev_rows = [], []; c1 = c3a = c3b = 0.0; nben = []; len_ok = True; ben_idx_ok = True
for mdl, c, s, name in canonical_runs():
    z = load_export(name); p = np.load(ZD / name / "predictions.npz")
    names, _ = classes19(name); ben = names.index("Benign"); ben_idx_ok &= (ben == 1)
    y19, hid = z["y19"].astype(int), int(z["hid"]); en = z["energy"].astype(np.float64); pa = z["p_attack"].astype(np.float64)
    ve = z["val_energy"].astype(np.float64); vy = p["val_y_true_label19"].astype(int)
    len_ok &= (len(ve) == len(vy))
    vb = ve[vy == ben]; nben.append(len(vb))
    thr = {"reference": float(np.percentile(ve, 5))}
    for q in QS: thr[f"benign_q{q}"] = float(np.percentile(vb, q))
    c1 = max(c1, abs(thr["reference"] - float(z["thr"])))
    unk, kb = y19 == hid, y19 == ben; ka = (~unk) & (~kb); bn = pa > 0.5
    segs = segments(y19); assert len(segs) == len(np.unique(y19)), name
    for tn in TH:
        flag = en < thr[tn]; al = flag | bn
        r = dict(model=mdl, held_out_class=c, seed=s, threshold=tn, thr_value=thr[tn], n_benign_val=len(vb), benign_flag_rate=float(flag[kb].mean()), benign_alert_rate=float(al[kb].mean()),
                 heldout_caught_energy=float(flag[unk].mean()), heldout_caught_any=float(al[unk].mean()), known_attack_flag_rate=float(flag[ka].mean()), known_attack_alert_rate=float(al[ka].mean()),
                 known_far_flag=float(flag[~unk].mean()), known_far_alert=float(al[~unk].mean()))
        rows.append(r)
        if tn == "reference": c3a = max(c3a, abs(r["heldout_caught_any"] - float(pr.loc[(mdl, c, s), "A2_caught_any"])))
        for n, mm in RULES:
            evb = nb = tu = nu = ka_t = ka_n = 0; hf = False
            for a0, a1 in segs:
                lab = y19[a0]; valid = a1 - a0 - (n - 1)
                if valid <= 0: continue
                tr = triggers(al[a0:a1], n, mm)
                if lab == ben: evb += events(tr); nb += valid
                elif lab == hid: tu += int(tr[n - 1:].sum()); nu += valid; hf = bool(tr.any())
                else: ka_t += int(tr[n - 1:].sum()); ka_n += valid
            e = dict(model=mdl, held_out_class=c, seed=s, threshold=tn, n=n, m=mm, benign_events_per_10k_windows=1e4 * evb / nb if nb else np.nan, heldout_window_detection=tu / nu if nu else np.nan,
                     heldout_file_detected=int(hf), known_attack_window_trigger_rate=ka_t / ka_n if ka_n else np.nan)
            ev_rows.append(e)
            if tn == "reference":
                o = T9OLD.loc[(mdl, c, s, n, mm)]
                for k in ("benign_events_per_10k_windows", "heldout_window_detection", "heldout_file_detected", "known_attack_window_trigger_rate"):
                    c3b = max(c3b, abs(e[k] - float(o[k])) if np.isfinite(e[k]) else 0.0)
    print("done", name, f"{time.time() - t0:.0f}s", flush=True)
cmd = f"python scripts/p4_rev_r4_benign_threshold.py --date {a.date}"
res_c = [("C1", c1 <= 1e-6, f"max |thr_all - saved thr| = {c1:.2e}"), ("C2", bool(len_ok), "val_energy and val_y_true_label19 have equal length in all 75 runs"),
         ("C3", c3a <= 1e-6 and c3b <= 1e-9, f"held-out caught-any at thr_all vs A2_caught_any max |diff| {c3a:.2e}; event-level values at thr_all vs saved 20 Nov T9 per-run values max |diff| {c3b:.2e}"), ("Benign index", bool(ben_idx_ok), "label19 index of Benign is 1 in all runs")]
for k, ok, msg in res_c: print(k, "PASS" if ok else "FAIL", msg)
if not all(ok for _, ok, _ in res_c):
    fail_readme(d, "R4 Benign-calibrated thresholds", __file__, cmd, "; ".join(f"{k} {'PASS' if ok else 'FAIL'}: {m}" for k, ok, m in res_c), t0); sys.exit(1)
dr, de = pd.DataFrame(rows), pd.DataFrame(ev_rows)
dr.to_csv(d / "r4_per_run.csv", index=False); de.to_csv(d / "r4_event_level.csv", index=False)
mc = [c_ for c_ in dr.columns if c_ not in ("model", "held_out_class", "seed", "threshold")]
sm = seed_ci(dr, mc, by=("model", "held_out_class", "threshold"))
pool = dr.groupby(["model", "threshold"])[mc].mean().reset_index(); pool.insert(1, "held_out_class", "pooled (25 runs)"); pool.insert(3, "n_seeds", 25)
pd.concat([sm, pool], ignore_index=True).to_csv(d / "r4_summary.csv", index=False)
emc = ["benign_events_per_10k_windows", "heldout_window_detection", "heldout_file_detected", "known_attack_window_trigger_rate"]
es = seed_ci(de, emc, by=("model", "held_out_class", "threshold", "n", "m"))
epool = de.groupby(["model", "threshold", "n", "m"])[emc].mean().reset_index(); epool.insert(1, "held_out_class", "pooled (25 runs)"); epool.insert(4, "n_seeds", 25)
pd.concat([es, epool], ignore_index=True).to_csv(d / "r4_event_level_summary.csv", index=False)

def f(v, k=4): return f"{v:.{k}f}"
mo = {k: i for i, k in enumerate(MODELS)}
T1 = ["| Model | Threshold | Benign flag | Benign alert (safety view) | Held-out caught (energy only) | Held-out caught-any | Known-attack flag | Known false-alarm rate (flag) |", "|---|---|---|---|---|---|---|---|"]
for mdl in MODELS:
    for tn in TH:
        g = dr[(dr.model == mdl) & (dr.threshold == tn)]; cs = sm[(sm.model == mdl) & (sm.threshold == tn)]
        T1.append(f"| {mdl} | {tn} | " + " | ".join(f(g[k].mean()) for k in ("benign_flag_rate", "benign_alert_rate", "heldout_caught_energy", "heldout_caught_any", "known_attack_flag_rate", "known_far_flag")) + " |")
T2 = ["| Model | Threshold | Rule | Benign events per 10,000 | Held-out windows triggering (full history) | Held-out files detected | Known-attack windows triggering |", "|---|---|---|---|---|---|---|"]
for mdl in MODELS:
    for tn in TH:
        for n, mm in RULES:
            g = de[(de.model == mdl) & (de.threshold == tn) & (de.n == n) & (de.m == mm)]
            T2.append(f"| {mdl} | {tn} | {mm} of {n} | {g.benign_events_per_10k_windows.mean():.2f} | {f(g.heldout_window_detection.mean())} | {g.heldout_file_detected.mean():.2f} | {f(g.known_attack_window_trigger_rate.mean())} |")
T3 = ["| Held-out class (TRACE-MoE) | Threshold | Benign flag [CI] | Benign alert [CI] | Held-out caught (energy) [CI] | Held-out caught-any [CI] |", "|---|---|---|---|---|---|"]
for c in CLASSES:
    for tn in TH:
        r = sm[(sm.model == "trace_moe") & (sm.held_out_class == c) & (sm.threshold == tn)].iloc[0]
        cell = lambda k: f"{f(r[k])} [{f(r[k + '_lo'])}, {f(r[k + '_hi'])}]"
        T3.append(f"| {c} | {tn} | {cell('benign_flag_rate')} | {cell('benign_alert_rate')} | {cell('heldout_caught_energy')} | {cell('heldout_caught_any')} |")
res = ["Checks: " + "; ".join(f"{k} {'PASS' if ok else 'FAIL'} ({m})" for k, ok, m in res_c) + ".", "",
       f"Benign validation windows per run: min {min(nben):,}, median {int(np.median(nben)):,}, max {max(nben):,}. A 1% threshold rests on about {int(round(np.median(nben) * 0.01))} tail windows (median run), so thr_bq1 is a noisy estimate.", "",
       "Window order (C2): `val_energy` is written by scripts/p4_export_scores.py as scores(model, wd, wd.starts['val']) (batches of that start array, concatenated in order); `val_y_true_label19` is written by src/trace_iomt/experiment.py from evaluate(model, wd, 'val', ...) in src/trace_iomt/train.py, which uses starts = wd.starts['val'] and y_true = wd.y[lvl][starts + T - 1]. Both therefore follow the same start array; the lengths are equal in all 75 runs.", "",
       "Design (fixed before any test result was seen): thr_all = 5th percentile of all validation energies (equals the saved thr); thr_bq = q-th percentile of the validation energies of Benign windows, q = 1, 2, 5; a window is flagged when energy < threshold; safety-view alert = flag OR p_attack > 0.5; event rules 2 of 4, 8 of 16 and 32 of 64 per capture file as in T9; summaries are means with seed intervals (seed_ci, 10,000 resamples, rng 42).", "",
       "Pooled means over the 75 runs per model and threshold:", ""] + T1 + ["", "Event-level rules (means over the five classes and seeds):", ""] + T2 + ["", "TRACE-MoE per held-out class (mean over five seeds, 95% interval over seeds):", ""] + T3
write_readme(d, "R4: Benign-calibrated energy thresholds", __file__, cmd, "score_export/*.npz, zero_day/<run>/predictions.npz (val_y_true_label19), saved_output_analyses_20261118/per_run.csv, p4_review_analyses_20261120/T9/event_level_per_run.csv", 75, "\n".join(res), t0,
             "Exploratory and post hoc; no threshold is recommended; the thresholds use validation windows only and the test windows are used for reporting. H1 to H5 and their verdicts do not change. No training and no inference were run.")
print("R4 done", f"{time.time() - t0:.0f}s")
