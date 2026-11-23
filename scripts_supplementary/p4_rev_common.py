"""Shared helpers for the Paper 4 review analyses (tasks T1-T11 of future-work.md and U1-U12 of future-work_part2.md).
Rules: canonical 75 runs only; test windows are for reporting only; exploratory, post hoc; new files only; seed 42.
Outputs go to results_v3_paper3/p4_review_analyses_<DATE>/ (T tasks: T<n>/, U tasks: part2/U<n>/)."""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "results_v3_paper3"
EXP = RS / "score_export"
ZD = RS / "zero_day"
DATE = "20261120"
OUT = RS / f"p4_review_analyses_{DATE}"
MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = ["ARP_Spoofing", "Recon-VulScan", "MQTT-Malformed_Data", "TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
SEEDS = [42, 43, 44, 45, 46]
EXPLORATORY = "Exploratory, post hoc analysis; no change to H1-H5, their decision rules or verdicts. Test windows used for reporting only."


def canonical_runs():
    """The 75 canonical runs (model, class, seed, run name); never the _1epoch_backup, __v2 or __v3 folders."""
    out = []
    for m in MODELS:
        for c in CLASSES:
            for s in SEEDS:
                name = f"{m}_loao_T16_{c}_seed{s}"
                assert (EXP / f"{name}.npz").exists() and (ZD / name / "metrics.json").exists(), name
                out.append((m, c, s, name))
    assert len(out) == 75
    return out


def load_export(name):
    return np.load(EXP / f"{name}.npz", allow_pickle=False)


def classes19(name):
    h = json.loads((ZD / name / "metrics.json").read_text(encoding="utf-8"))["hierarchy"]
    return h["classes"]["label19"], h["parent19"]


def per_run_table():
    return pd.read_csv(RS / "saved_output_analyses_20261118" / "per_run.csv")


def seed_ci(df, cols, by=("model", "held_out_class"), B=10000):
    """Mean over the five seeds with a percentile bootstrap 95% interval (10,000 resamples of the five seed values, rng seed 42)."""
    rng = np.random.default_rng(42)
    rows = []
    for key, g in df.groupby(list(by)):
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec["n_seeds"] = len(g)
        for c in cols:
            v = g[c].to_numpy(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            bs = rng.choice(v, size=(B, len(v)), replace=True).mean(1)
            rec[c], rec[c + "_lo"], rec[c + "_hi"] = float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_diff_ci(df, a, b, by=("model", "held_out_class"), B=10000):
    """Paired seed bootstrap of the difference df[a] - df[b] (same runs)."""
    d = df.copy()
    d["_diff"] = d[a] - d[b]
    r = seed_ci(d, ["_diff"], by=by, B=B)
    return r.rename(columns={"_diff": f"{a}_minus_{b}", "_diff_lo": f"{a}_minus_{b}_lo", "_diff_hi": f"{a}_minus_{b}_hi"})


def tpr_at_fpr(y, s, fpr_target):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def fpr_at_tpr(y, s, tpr_target=0.95):
    """Smallest false-positive rate (on y == 0) at a true-positive rate (on y == 1) of at least tpr_target; higher s = more positive."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    ok = tpr >= tpr_target
    return float(fpr[ok].min()) if ok.any() else 1.0


def task_dir(kind, n):
    d = OUT / (f"T{n}" if kind == "T" else f"part2/U{n}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_readme(d, title, script, command, inputs, runs, results, t0, extra=""):
    sha = hashlib.sha256(Path(script).read_bytes()).hexdigest()
    import scipy
    import sklearn
    txt = [f"# {title}", "", f"**Status:** {EXPLORATORY}", "",
           f"- Command: `{command}`", f"- Script: `{Path(script).name}`, SHA-256 `{sha}`",
           f"- Inputs: {inputs}", f"- Canonical runs read: {runs}", "- rng seed: 42",
           f"- Versions: Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}",
           f"- Start {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}, end {time.strftime('%Y-%m-%d %H:%M:%S')}, duration {time.time() - t0:.0f} s", "", "## Result", results, ""]
    if extra:
        txt += ["## Notes", extra, ""]
    (d / "README.md").write_text("\n".join(txt), encoding="utf-8")


def fail_readme(d, title, script, command, reason, t0):
    write_readme(d, title + " (STOPPED)", script, command, "see reason", "n/a", f"**STOPPED: {reason}**", t0)
