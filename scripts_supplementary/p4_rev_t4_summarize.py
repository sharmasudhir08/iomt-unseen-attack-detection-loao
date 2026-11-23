"""Paper 4 T4 summary (post-hoc exploratory): PCRP covariance-estimator and window-aggregation variants, from the saved per-seed output of
scripts/p4_rev_pcrp_t4_u3.py (run 21 Nov 2026, script SHA-256 1328fbd6...). No model fitting here; reads CSV only.
Mean over 5 seeds and percentile bootstrap 95% CI over seeds (10,000 resamples, rng 42). No variant is declared best; no verdict changes.
Usage: python scripts/p4_rev_t4_summarize.py --date 20261123
"""
import argparse, hashlib, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; R3 = ROOT / "results_v3_paper3"
ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); a = ap.parse_args()
SRC = R3 / "p4_review_analyses_20261121" / "T4"; OUT = R3 / f"p4_review_analyses_{a.date}" / "T4"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
d = pd.read_csv(SRC / "pcrp_variants_per_seed.csv"); st = pd.read_csv(SRC / "estimator_status.csv")
rows = []
for (c, e, g, k), x in d.groupby(["held_out_class", "estimator", "aggregation", "k"], sort=False):
    for met in ("auroc", "fpr95"):
        v = x.sort_values("seed")[met].to_numpy(); bs = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
        rows.append(dict(held_out_class=c, estimator=e, aggregation=g, k=k, metric=met, n_seeds=len(v), mean=v.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
s = pd.DataFrame(rows); s.to_csv(OUT / "pcrp_variants_summary.csv", index=False)
# pass checks
ref = s[(s.estimator == "reference") & (s.aggregation == "last") & (s.metric == "auroc")].set_index("held_out_class")["mean"]
h5 = pd.read_csv(R3 / "h8_verdict" / "twin_h5_verdict.csv"); h5 = h5[h5.model == "trace_moe"].set_index("held_out_class")["mean_auroc"]
dref = float((ref - h5.reindex(ref.index)).abs().max())
nok = int((st.status == "ok").sum()); ntot = len(st)
(OUT / "README.md").write_text(
    "# T4 PCRP variants (post-hoc exploratory)\n\nStatus: exploratory, post hoc; no change to H1-H5, their decision rules or verdicts. Saved artifacts only; no training.\n\n"
    "Computation (run 2026-11-21 11:55-12:25): `scripts/p4_rev_pcrp_t4_u3.py`, SHA-256 1328fbd64f1091c3b7b50914515a2eebf525f773803cbea105f4c62b07623bae (identical in trace-iomt/scripts and the repository copy); inputs: twin profiles and prepared test windows; per-seed raw output in `../../p4_review_analyses_20261121/T4/` (not modified).\n"
    f"Summary: `scripts/p4_rev_t4_summarize.py` SHA-256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}; bootstrap 10,000 resamples over 5 seeds, rng 42; {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
    f"Pass checks: estimator status ok for {nok} of {ntot} (class, seed, estimator) rows; reference (original estimator, last timestep) AUROC reproduces the saved per-seed AUROC to <= 2.3e-16 in the 21 Nov log; mean reference AUROC versus twin_h5_verdict.csv max |difference| = {dref:.2e}.\n\n"
    "Variants: covariance = original (reference), Ledoit-Wolf, MinCovDet; aggregation of D_min over the last k timesteps: last (k=1), and the mean and the minimum over the last k = 4 and k = 16 timesteps. No variant is declared best.\n\n"
    + s[s.metric == "auroc"].round(4).to_markdown(index=False) + "\n", encoding="utf-8")
print("rows", len(s), "ok", nok, ntot, "dref", dref)
print(s[s.metric=="auroc"].groupby(["estimator","aggregation"])["mean"].mean().round(4))
print(d[['estimator','aggregation','k']].drop_duplicates().to_string())
