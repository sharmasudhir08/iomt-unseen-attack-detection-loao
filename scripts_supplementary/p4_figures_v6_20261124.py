"""Paper 4: four figures redrawn from saved results only (no training, no new experiment). CPU only.
All text is 8 pt at the stated print width; model names TRACE-MoE, Transformer, BiLSTM-DNN; class names as in the paper.
Figures (documents_v2/paper4/figures_v6/):
  fig_h3.pdf               H3 energy-score AUROC by model and class, 8.6 cm wide (one column); saved_output_analyses_20261118/per_run.csv
  fig_risk_coverage.pdf    risk-coverage, one panel per architecture, 14.5 cm; T5/risk_coverage_mean_over_seeds.csv (AURC from T5/aurc_per_run.csv)
  fig_reliability.pdf      19-class reliability, 15 equal-width bins, known test windows pooled over classes and seeds, 14.5 cm; score_export/<run>.npz (logits19, as in p4_rev_t5_t9.py)
  fig_flood_subclusters.pdf  PCA view of the k = 3 k-means sub-clusters of the held-out flood windows, one panel per class (seed 42), 17.8 cm;
                           settings of scripts/p4_rev_t10_flood_clusters.py (k = 3, random_state 42, fit on at most 50,000 windows, PCA fit on the same windows,
                           silhouette on 10,000 windows); pcrp_cache_20261121 and score_export
Checks (written to FIGURES_V6_REPORT.md): values against the paper's tables and the saved CSV files; minimum font size of every text; PDF page size.
Usage: python scripts/p4_figures_v6_20261124.py [--only h3 rc rel flood]"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "results_v3_paper3"
T5 = RS / "p4_review_analyses_20261121" / "T5"
T10 = RS / "p4_review_analyses_20261121" / "T10"
OUT = ROOT / "documents_v2" / "paper4" / "figures_v6"
PREV = OUT / "preview"
MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = ["ARP_Spoofing", "Recon-VulScan", "MQTT-Malformed_Data", "TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
SEEDS = [42, 43, 44, 45, 46]
MN = {"trace_moe": "TRACE-MoE", "transformer": "Transformer", "bilstm_dnn": "BiLSTM-DNN"}
CN = {"ARP_Spoofing": "ARP spoofing", "Recon-VulScan": "Recon-VulScan", "MQTT-Malformed_Data": "Malformed MQTT traffic",
      "TCP_IP-DoS-UDP": "TCP/IP DoS-UDP", "MQTT-DDoS-Publish_Flood": "MQTT-DDoS publish flooding"}
CN2 = {"ARP_Spoofing": "ARP spoofing", "Recon-VulScan": "Recon-VulScan", "MQTT-Malformed_Data": "Malformed MQTT\ntraffic",
       "TCP_IP-DoS-UDP": "TCP/IP DoS-UDP", "MQTT-DDoS-Publish_Flood": "MQTT-DDoS\npublish flooding"}
MC = {"trace_moe": "#1f77b4", "transformer": "#ff7f0e", "bilstm_dnn": "#2ca02c"}
FS = 8.0
CM = 1 / 2.54
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": FS,
                     "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
                     "figure.titlesize": FS, "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "lines.linewidth": 1.1})
REP = []
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((bool(ok), name, detail))
    print(("PASS " if ok else "FAIL ") + name + (" | " + detail if detail else ""), flush=True)


def finish(fig, stem, width_cm):
    """Font check, save PDF at the exact size and a PNG preview."""
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    check(f"{stem}: smallest text {min(sizes):.1f} pt (needs >= 8)", min(sizes) >= FS - 1e-6, f"{len(sizes)} texts")
    w, h = fig.get_size_inches()
    check(f"{stem}: figure width {w / CM:.2f} cm (target {width_cm})", abs(w / CM - width_cm) < 0.02, f"height {h / CM:.2f} cm")
    OUT.mkdir(parents=True, exist_ok=True)
    PREV.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(PREV / f"{stem}.png", dpi=300)
    plt.close(fig)
    from pypdf import PdfReader
    pg = PdfReader(str(OUT / f"{stem}.pdf")).pages[0]
    pw = float(pg.mediabox.width) / 72 / CM
    check(f"{stem}: PDF page width {pw:.2f} cm", abs(pw - width_cm) < 0.05)


def per_run():
    return pd.read_csv(RS / "saved_output_analyses_20261118" / "per_run.csv")


# ---------------------------------------------------------------- Figure: H3
def fig_h3():
    d = per_run()
    g = d.groupby(["model", "held_out_class"])["A7_energy_auroc"].agg(["mean", "min", "max", "count"]).reset_index()
    assert (g["count"] == 5).all() and len(g) == 15
    paper = {("trace_moe", "ARP_Spoofing"): 0.9658, ("trace_moe", "Recon-VulScan"): 0.9692, ("trace_moe", "MQTT-Malformed_Data"): 0.9810,
             ("trace_moe", "TCP_IP-DoS-UDP"): 0.3139, ("trace_moe", "MQTT-DDoS-Publish_Flood"): 0.3256,
             ("transformer", "ARP_Spoofing"): 0.9821, ("transformer", "Recon-VulScan"): 0.9743, ("transformer", "MQTT-Malformed_Data"): 0.9816,
             ("transformer", "TCP_IP-DoS-UDP"): 0.6673, ("transformer", "MQTT-DDoS-Publish_Flood"): 0.1159,
             ("bilstm_dnn", "ARP_Spoofing"): 0.8574, ("bilstm_dnn", "Recon-VulScan"): 0.7278, ("bilstm_dnn", "MQTT-Malformed_Data"): 0.9058,
             ("bilstm_dnn", "TCP_IP-DoS-UDP"): 0.6650, ("bilstm_dnn", "MQTT-DDoS-Publish_Flood"): 0.3341}
    dev = max(abs(r["mean"] - paper[(r["model"], r["held_out_class"])]) for _, r in g.iterrows())
    check("fig_h3: 15 plotted means equal the energy-score AUROC column T = 1 of the temperature table", dev < 5.1e-5, f"max |diff| {dev:.1e}")
    fig, ax = plt.subplots(figsize=(8.6 * CM, 8.4 * CM))
    n = len(CLASSES)
    bh = 0.26
    for j, m in enumerate(MODELS):
        for i, c in enumerate(CLASSES):
            r = g[(g.model == m) & (g.held_out_class == c)].iloc[0]
            y = (n - 1 - i) + (1 - j) * bh
            ax.barh(y, r["mean"], height=bh * 0.92, color=MC[m], label=MN[m] if i == 0 else None, zorder=2)
            ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["min"]], [r["max"] - r["mean"]]], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([CN2[c] for c in CLASSES][::-1])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Energy-score AUROC (mean of five seeds)")
    ax.axvline(0.5, color="gray", lw=0.6, ls=":", zorder=1)
    ax.grid(axis="x", color="0.88", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, frameon=False, columnspacing=0.8, handlelength=1.0, handletextpad=0.4)
    fig.subplots_adjust(left=0.33, right=0.98, top=0.98, bottom=0.25)
    finish(fig, "fig_h3", 8.6)


# ---------------------------------------------------------------- Figure: risk-coverage
def fig_rc():
    rc = pd.read_csv(T5 / "risk_coverage_mean_over_seeds.csv")
    check("fig_risk_coverage: CSV has 3 models x 5 classes x 2 scores x 20 coverage points", len(rc) == 600 and rc.groupby(["model", "held_out_class", "score"]).size().eq(20).all(), f"{len(rc)} rows")
    au = pd.read_csv(T5 / "aurc_per_run.csv")
    tab = au.groupby(["model", "score"])["aurc"].mean()
    paper = {("trace_moe", "energy"): 0.0336, ("trace_moe", "msp"): 0.0392, ("transformer", "energy"): 0.0293, ("transformer", "msp"): 0.0251,
             ("bilstm_dnn", "energy"): 0.0259, ("bilstm_dnn", "msp"): 0.0245}
    dev = max(abs(round(tab[k], 4) - v) for k, v in paper.items())
    check("fig_risk_coverage: AURC values shown (mean of 25 runs) equal Table 15 / S12 'all classes' rows", dev < 1e-9, " ".join(f"{MN[m]} {s} {tab[(m, s)]:.4f}" for m in MODELS for s in ("energy", "msp")))
    cnt = au.groupby(["model", "score"]).size()
    check("fig_risk_coverage: 25 runs per model and score in aurc_per_run.csv", (cnt == 25).all())
    fig, axs = plt.subplots(1, 3, figsize=(14.5 * CM, 5.6 * CM))
    trap = {}
    for ax, m in zip(axs, MODELS):
        for sc, ls, col, lab in (("energy", "-", "#1f4e9c", "Energy"), ("msp", "--", "#c0504d", "MSP")):
            c = rc[(rc.model == m) & (rc.score == sc)].groupby("coverage")["risk"].mean().sort_index()
            ax.plot(c.index, c.values, ls=ls, color=col, label=f"{lab} (AURC {tab[(m, sc)]:.4f})")
            x = np.r_[0.0, c.index.to_numpy()]
            y = np.r_[c.values[0], c.values]
            trap[(m, sc)] = float(np.trapezoid(y, x))
        ax.set_title(MN[m])
        ax.set_xlabel("Coverage")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        top = max(rc[(rc.model == m)].groupby(["score", "coverage"])["risk"].mean().max(), 1e-9)
        ax.set_ylim(0, top * 1.55)
        ax.grid(color="0.9", lw=0.5)
        ax.legend(loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.2)
    axs[0].set_ylabel("Risk")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.20, wspace=0.30)
    REP.append("Risk-coverage: AURC of the plotted 20-point curves (trapezoid) against the per-run mean in the tables (information only): " +
               "; ".join(f"{MN[m]} {s} {trap[(m, s)]:.4f} vs {tab[(m, s)]:.4f}" for m in MODELS for s in ("energy", "msp")))
    finish(fig, "fig_risk_coverage", 14.5)


# ---------------------------------------------------------------- Figure: reliability
NB = 15


def ece_bins(conf, correct):
    """Same definition as scripts/p4_rev_t5_t9.py (the source of T5/ece_per_run.csv): 15 equal-width bins, bin = floor(conf * 15),
    ECE = sum over bins of |sum(correct) - sum(conf)| / N."""
    b = np.minimum((conf * NB).astype(int), NB - 1)
    cnt = np.bincount(b, minlength=NB).astype(float)
    sa = np.bincount(b, weights=correct, minlength=NB)
    sc = np.bincount(b, weights=conf, minlength=NB)
    e = float(np.sum(np.abs(sa - sc)) / cnt.sum())
    return e, cnt, sc, sa


def fig_rel():
    cache = OUT / "reliability_bins_cache_v2.npz"
    ece_ref = pd.read_csv(T5 / "ece_per_run.csv")
    if cache.exists():
        z = np.load(cache)
        pooled = {m: z[m] for m in MODELS}
        run_ece = pd.DataFrame(json.loads(str(z["run_ece"])))
        print("reliability bins read from cache", cache, flush=True)
    else:
        pooled = {m: np.zeros((NB, 3)) for m in MODELS}
        rows = []
        t0 = time.time()
        for m in MODELS:
            for c in CLASSES:
                for s in SEEDS:
                    name = f"{m}_loao_T16_{c}_seed{s}"
                    ex = np.load(RS / "score_export" / f"{name}.npz")
                    y19, hid = ex["y19"].astype(int), int(ex["hid"])
                    kn = y19 != hid
                    lg = ex["logits19"].astype(np.float32)
                    pr = np.exp(lg - lg.max(1, keepdims=True))
                    pr /= pr.sum(1, keepdims=True)
                    pred = lg.argmax(1)
                    conf = pr.max(1)[kn]
                    e, cnt, sc, sa = ece_bins(conf, (pred[kn] == y19[kn]).astype(float))
                    pooled[m] += np.stack([cnt, sc, sa], 1)
                    rows.append(dict(model=m, held_out_class=c, seed=s, ece=e))
                    print(name, f"{time.time() - t0:.0f}s", flush=True)
        run_ece = pd.DataFrame(rows)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(cache, run_ece=json.dumps(run_ece.to_dict("list")), **pooled)
    mg = run_ece.merge(ece_ref, on=["model", "held_out_class", "seed"])
    check("fig_reliability: 75 per-run ECE values recomputed from predictions.npz equal ece_per_run.csv", len(mg) == 75 and (mg["ece"] - mg["ece_label19"]).abs().max() < 1e-9, f"max |diff| {(mg['ece'] - mg['ece_label19']).abs().max():.1e}")
    mean_ece = run_ece.groupby("model")["ece"].mean()
    paper = {"bilstm_dnn": 0.0160, "trace_moe": 0.0079, "transformer": 0.0071}
    dev = max(abs(round(mean_ece[m], 4) - v) for m, v in paper.items())
    check("fig_reliability: ECE (mean of 25 runs) equals 0.0160, 0.0079, 0.0071", dev < 1e-9, " ".join(f"{MN[m]} {mean_ece[m]:.4f}" for m in MODELS))
    pooled_ece = {}
    for m in MODELS:
        cnt, sc, sa = pooled[m][:, 0], pooled[m][:, 1], pooled[m][:, 2]
        ok = cnt > 0
        pooled_ece[m] = float((np.abs(sa[ok] / cnt[ok] - sc[ok] / cnt[ok]) * cnt[ok] / cnt.sum()).sum())
    REP.append("Reliability: ECE of the pooled bins plotted (information only): " + "; ".join(f"{MN[m]} {pooled_ece[m]:.4f} (mean of 25 runs {mean_ece[m]:.4f})" for m in MODELS))
    fig, axs = plt.subplots(1, 3, figsize=(14.5 * CM, 5.6 * CM))
    for ax, m in zip(axs, MODELS):
        cnt, sc, sa = pooled[m][:, 0], pooled[m][:, 1], pooled[m][:, 2]
        ok = cnt > 0
        ax.plot([0, 1], [0, 1], color="0.4", ls="--", lw=0.8, label="Perfect calibration")
        ax.plot(sc[ok] / cnt[ok], sa[ok] / cnt[ok], "o-", ms=3.0, color=MC[m], label=f"ECE {mean_ece[m]:.4f}")
        ax.set_title(MN[m])
        ax.set_xlabel("Mean confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.grid(color="0.9", lw=0.5)
        ax.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.2)
    axs[0].set_ylabel("Accuracy")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.20, wspace=0.30)
    finish(fig, "fig_reliability", 14.5)


# ---------------------------------------------------------------- Figure: flood sub-clusters
def fig_flood():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, silhouette_score
    flood = ["TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
    sil = pd.read_csv(T10 / "silhouette.csv")
    ab = pd.read_csv(T10 / "auroc_by_cluster.csv")
    sdev, adev, ncmp = 0.0, 0.0, 0
    keep = {}
    for held in flood:
        for seed in SEEDS:
            c = np.load(RS / "pcrp_cache_20261121" / f"{held}_seed{seed}.npz")
            y19, hid = c["y19"].astype(int), int(c["hid"])
            unk = y19 == hid
            X = c["unk_X_last"].astype(np.float64)
            rng = np.random.default_rng(42)
            fit_idx = np.sort(rng.choice(len(X), min(50_000, len(X)), replace=False))
            sil_idx = rng.choice(fit_idx, min(10_000, len(fit_idx)), replace=False)
            km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X[fit_idx])
            lab = km.predict(X)
            s_val = float(silhouette_score(X[sil_idx], km.predict(X[sil_idx])))
            ref = sil[(sil.held_out_class == held) & (sil.seed == seed) & (sil.k == 3)].silhouette.iloc[0]
            sdev = max(sdev, abs(s_val - ref))
            au = {}
            for m in MODELS:
                en = -np.load(RS / "score_export" / f"{m}_loao_T16_{held}_seed{seed}.npz")["energy"].astype(np.float64)
                known = en[~unk]
                for j in range(3):
                    mj = lab == j
                    v = float(roc_auc_score(np.r_[np.ones(mj.sum()), np.zeros(len(known))], np.r_[en[unk][mj], known]))
                    r = ab[(ab.held_out_class == held) & (ab.seed == seed) & (ab.model == m) & (ab.k == 3) & (ab.cluster == j)].energy_auroc.iloc[0]
                    adev = max(adev, abs(v - r))
                    ncmp += 1
                    au[(m, j)] = v
            print(held, seed, "silhouette", round(s_val, 4), flush=True)
            if seed == 42:
                p = PCA(n_components=2, random_state=42).fit(X[fit_idx])
                keep[held] = dict(xy=p.transform(X[sil_idx]), lab=km.predict(X[sil_idx]), lab_all=lab, au=au, sil=s_val,
                                  evr=p.explained_variance_ratio_)
    check("fig_flood_subclusters: 10 silhouettes (2 classes x 5 seeds, k = 3) equal T10/silhouette.csv", sdev < 1e-9, f"max |diff| {sdev:.1e}")
    check("fig_flood_subclusters: 90 cluster AUROCs (3 models x 3 clusters x 10 runs) equal T10/auroc_by_cluster.csv", adev < 1e-9, f"{ncmp} values, max |diff| {adev:.1e}")
    fig, axs = plt.subplots(1, 2, figsize=(17.8 * CM, 8.4 * CM))
    cols = ["#1f77b4", "#d62728", "#2ca02c"]
    for ax, held in zip(axs, flood):
        d = keep[held]
        sizes = np.array([(d["lab_all"] == j).sum() for j in range(3)])
        order = np.argsort(-sizes)
        for rank, j in enumerate(order):
            mk = d["lab"] == j
            ax.scatter(d["xy"][mk, 0], d["xy"][mk, 1], s=1.5, color=cols[rank], rasterized=True, linewidths=0,
                       label=f"Cluster {rank + 1} ({100 * sizes[j] / sizes.sum():.1f}% of windows), AUROC {d['au'][('trace_moe', j)]:.2f}")
        ax.set_title(f"{CN[held]} (seed 42, silhouette {d['sil']:.2f})")
        ax.set_xlabel(f"Principal component 1 ({100 * d['evr'][0]:.0f}% of variance)")
        ax.set_ylabel(f"Principal component 2 ({100 * d['evr'][1]:.0f}%)")
        ax.grid(color="0.9", lw=0.5)
        lg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, markerscale=5, handletextpad=0.2, borderaxespad=0.0, labelspacing=0.3,
                       title="AUROC: TRACE-MoE energy score", alignment="center")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.36, wspace=0.22)
    finish(fig, "fig_flood_subclusters", 17.8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["h3", "rc", "rel", "flood"])
    a = ap.parse_args()
    t0 = time.time()
    for key, fn in (("h3", fig_h3), ("rc", fig_rc), ("rel", fig_rel), ("flood", fig_flood)):
        if key in a.only:
            fn()
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lines = ["# Paper 4 figures v6 (redrawn from saved results only)", "",
             f"Script `scripts/p4_figures_v6_20261124.py`, SHA-256 `{sha}`; Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, matplotlib {matplotlib.__version__}; CPU only; {time.strftime('%Y-%m-%d %H:%M:%S')}; {time.time() - t0:.0f} s.",
             "No training and no new experiment; inputs: score_export (19-class logits), pcrp_cache_20261121, T5, T10, saved_output_analyses_20261118.", "", "## Checks", ""]
    lines += [f"- [{'PASS' if ok else 'FAIL'}] {n}" + (f" ({d})" if d else "") for ok, n, d in CHECKS]
    lines += ["", "## Information"] + [f"- {r}" for r in REP]
    (OUT / "FIGURES_V6_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("report written;", sum(1 for c in CHECKS if not c[0]), "failed checks", flush=True)
