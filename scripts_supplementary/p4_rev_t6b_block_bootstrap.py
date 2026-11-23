"""Paper 4 T6b: block/cluster bootstrap that works when each held-out class has a single test file.
Post-hoc exploratory; no verdict is recomputed or changed. CPU only, saved artifacts only (score_export/, pcrp_cache_20261121/).

Design (fixed before running; see README of the output folder):
 - Each resample draws 5 seeds with replacement (paired across metrics), as in the paper's seed bootstrap.
 - Known side: the test source files are resampled with replacement (cluster bootstrap over the 20 known files; the same draw for every seed).
 - Held-out side: the held-out windows of the single held-out file, taken in file order, are cut into consecutive non-overlapping blocks of L windows
   (last block shorter) and the blocks are resampled with replacement (block count kept), separately for each (resample, seed) draw, shared by all scores.
   L = 1000 (primary, fixed in advance); L = 250 and 4000 are sensitivity checks. With fewer than 2 blocks the interval is undefined and reported as nan.
 - 2,000 resamples, rng 42. AUROC from per-block / per-file histograms of 4,096 quantile bins (ties in a bin count one half); binned point estimate checked
   against the exact AUROC.
Usage: python scripts/p4_rev_t6b_block_bootstrap.py --date 20261123
"""
import argparse, hashlib, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results_v3_paper3"
MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = ["ARP_Spoofing", "Recon-VulScan", "MQTT-Malformed_Data", "TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
SEEDS = [42, 43, 44, 45, 46]
LS = [1000, 250, 4000]
NB, NBIN = 2000, 4096
ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); ap.add_argument("--cache", default="pcrp_cache_20261121"); a = ap.parse_args()
OUT = R3 / f"p4_review_analyses_{a.date}" / "T6"; OUT.mkdir(parents=True, exist_ok=False)
CACHE = R3 / a.cache
rng = np.random.default_rng(42)


def bins_of(score):
    edges = np.unique(np.quantile(score, np.linspace(0, 1, NBIN + 1)))
    return np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 1), len(edges)


def auc_rows(HU, HK):
    """AUROC per row for matrices of unknown / known histograms (rows = resamples)."""
    cum_k = np.cumsum(HK, axis=1) - HK
    return (HU * (cum_k + 0.5 * HK)).sum(1) / (HU.sum(1) * HK.sum(1))


def main():
    t0 = time.time()
    rows, status, maxdiff = [], [], 0.0
    point = {}
    for held in CLASSES:
        safe = held
        data = {s: np.load(CACHE / f"{safe}_seed{s}.npz") for s in SEEDS}
        names = sorted(set().union(*[set(data[s]["file_names"].tolist()) for s in SEEDS])); nidx = {n: i for i, n in enumerate(names)}; nf = len(names)
        per = {}   # (kind, s) -> dict(Hk (nf x nb), Hub{L} (nblocks x nb), caught arrays)
        nunk = {}
        for s in SEEDS:
            c = data[s]
            fid = np.array([nidx[n] for n in c["file_names"].tolist()])[c["file_id"]]
            unk = c["y19"].astype(int) == int(c["hid"])
            ui = np.flatnonzero(unk)
            assert (np.diff(ui) == 1).all() or len(set(fid[ui])) == 1, "held-out windows not in a single file"
            assert len(set(fid[ui])) == 1, "held-out class spans more than one file"
            nunk[s] = len(ui)
            scores = {"pcrp": c["dmin"].astype(np.float64)}
            for m in MODELS:
                se = np.load(R3 / "score_export" / f"{m}_loao_T16_{safe}_seed{s}.npz")
                assert len(se["y19"]) == len(unk) and (se["y19"].astype(int) == c["y19"].astype(int)).all(), "order mismatch"
                scores[f"energy_{m}"] = -se["energy"].astype(np.float64); scores[f"msp_{m}"] = -se["msp"].astype(np.float64)
                if m == "trace_moe":
                    caught = ((se["energy"] < float(se["thr"])) | (se["p_attack"] > 0.5))[ui]
            for k, sc in scores.items():
                b, nb = bins_of(sc)
                Hk = np.bincount(fid[~unk] * nb + b[~unk], minlength=nf * nb).reshape(nf, nb).astype(np.float64)
                d = {"Hk": Hk}
                for L in LS:
                    blk = np.arange(len(ui)) // L; nbk = int(blk.max()) + 1
                    d[L] = np.bincount(blk * nb + b[ui], minlength=nbk * nb).reshape(nbk, nb).astype(np.float64)
                per[(k, s)] = d
                ex = float(roc_auc_score(unk, sc))
                bd = auc_rows(d["Hk"].sum(0, keepdims=True), d["Hk"].sum(0, keepdims=True)) if False else None
                hu_all, hk_all = d[LS[0]].sum(0, keepdims=True), Hk.sum(0, keepdims=True)
                maxdiff = max(maxdiff, abs(float(auc_rows(hu_all, hk_all)[0]) - ex))
                point[(held, k, s)] = ex
            for L in LS:
                blk = np.arange(len(ui)) // L
                nbk = int(blk.max()) + 1
                per[("caught", s)] = per.get(("caught", s), {})
                per[("caught", s)][L] = (np.bincount(blk, weights=caught.astype(float), minlength=nbk), np.bincount(blk, minlength=nbk).astype(float))
            point[(held, "caught", s)] = float(caught.mean())
        hu_files = sorted({i for s in SEEDS for i in [nidx[n] for n in data[s]["file_names"].tolist()] if False})
        kn_files = [i for i in range(nf) if sum(per[("pcrp", s)]["Hk"][i].sum() for s in SEEDS) > 0]
        status.append(dict(held_out_class=held, n_heldout_files=1, n_known_files=len(kn_files), **{f"n_blocks_L{L}": int(np.ceil(nunk[SEEDS[0]] / L)) for L in LS}, n_heldout_windows_seed42=nunk[42]))
        print(held, "known files", len(kn_files), "held-out windows", nunk[42], f"{time.time()-t0:.0f}s", flush=True)

        seed_draw = rng.integers(0, len(SEEDS), (NB, len(SEEDS)))
        wk = np.zeros((NB, nf))
        for b in range(NB):
            wk[b] = np.bincount(rng.choice(kn_files, len(kn_files)), minlength=nf)
        ones_k = np.zeros(nf); ones_k[kn_files] = 1
        kinds = ["pcrp"] + [f"energy_{m}" for m in MODELS] + [f"msp_{m}" for m in MODELS]

        def seed_only(k, s):
            hk = per[(k, s)]["Hk"]
            return float(auc_rows(per[(k, s)][LS[0]].sum(0, keepdims=True), (ones_k @ hk)[None, :])[0])
        so = {k: np.array([seed_only(k, s) for s in SEEDS]) for k in kinds}
        so["caught"] = np.array([per[("caught", s)][LS[0]][0].sum() / per[("caught", s)][LS[0]][1].sum() for s in SEEDS])
        # the seed-only resample uses the file/window set as observed
        res_so = {}
        for L in LS:
            nbk = {s: per[("pcrp", s)][L].shape[0] for s in SEEDS}
            if min(nbk.values()) < 2:
                res_blk = None
            else:
                vals = {k: np.zeros((NB, len(SEEDS))) for k in kinds + ["caught"]}
                for j, s in enumerate(SEEDS):
                    n_b = nbk[s]
                    W = np.zeros((NB, n_b))
                    for b in range(NB):
                        W[b] = np.bincount(rng.integers(0, n_b, n_b), minlength=n_b)
                    for k in kinds:
                        HU = W @ per[(k, s)][L]
                        HK = wk @ per[(k, s)]["Hk"]
                        vals[k][:, j] = auc_rows(HU, HK)
                    nc, nn = per[("caught", s)][L]
                    vals["caught"][:, j] = (W @ nc) / (W @ nn)
                res_blk = vals
            res_so[L] = res_blk
        targets = [("H1_caught_any_trace_moe", "caught", None), ("H5_pcrp_auroc", "pcrp", None)] + [(f"H2_energy_minus_msp_{m}", f"energy_{m}", f"msp_{m}") for m in MODELS]
        for name, k1, k2 in targets:
            f = (lambda D: D[k1] - D[k2]) if k2 else (lambda D: D[k1])
            pt = float(f(so).mean())
            soi = np.array([f(so)[draw].mean() for draw in seed_draw])
            row = dict(held_out_class=held, quantity=name, point_mean_over_seeds=pt, seed_only_lo=float(np.percentile(soi, 2.5)), seed_only_hi=float(np.percentile(soi, 97.5)))
            for L in LS:
                v = res_so[L]
                if v is None:
                    row[f"block{L}_lo"] = row[f"block{L}_hi"] = np.nan
                else:
                    arr = np.array([f({k: v[k][b] for k in ([k1, k2] if k2 else [k1])})[seed_draw[b]].mean() for b in range(NB)])
                    row[f"block{L}_lo"], row[f"block{L}_hi"] = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "block_bootstrap.csv", index=False)
    pd.DataFrame(status).to_csv(OUT / "file_block_counts.csv", index=False)
    h1 = pd.read_csv(R3 / "h8_verdict" / "H1_safety_view.csv").set_index("class")["mean"]
    h5 = pd.read_csv(R3 / "h8_verdict" / "twin_h5_verdict.csv"); h5 = h5[h5.model == "trace_moe"].set_index("held_out_class")["mean_auroc"]
    d1 = max(abs(float(out[(out.held_out_class == c) & (out.quantity == "H1_caught_any_trace_moe")].point_mean_over_seeds.iloc[0]) - h1[c]) for c in CLASSES)
    d5 = max(abs(float(out[(out.held_out_class == c) & (out.quantity == "H5_pcrp_auroc")].point_mean_over_seeds.iloc[0]) - h5[c]) for c in CLASSES)
    (OUT / "README.md").write_text(
        "# T6b block/cluster bootstrap (post-hoc exploratory)\n\n"
        "Status: exploratory, post hoc; no change to H1-H5, their decision rules or verdicts. Saved artifacts only (score_export/, " + a.cache + "); no training.\n\n"
        f"Script `scripts/p4_rev_t6b_block_bootstrap.py` SHA-256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}; rng 42; {NB} resamples; run {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
        "Design: seeds resampled with replacement (paired across metrics); known test files resampled with replacement (cluster bootstrap over the known files); the held-out windows (a single file, in file order) cut into consecutive blocks of L windows and the blocks resampled with replacement. L = 1000 primary, 250 and 4000 sensitivity (all fixed before running). Fewer than 2 blocks: undefined (nan).\n\n"
        f"Checks: max |binned AUROC - exact AUROC| over all runs = {maxdiff:.2e}; max |point H1 mean - H1_safety_view.csv| = {d1:.2e}; max |point H5 mean - twin_h5_verdict.csv| = {d5:.2e}.\n"
        "Seed-only intervals use 2,000 resamples here and may differ slightly from the paper's 10,000-resample intervals.\n\n"
        + out.round(4).to_markdown(index=False) + "\n\n" + pd.DataFrame(status).to_markdown(index=False) + "\n", encoding="utf-8")
    print("done", f"max binned diff {maxdiff:.2e}", f"H1 diff {d1:.2e}", f"H5 diff {d5:.2e}", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
