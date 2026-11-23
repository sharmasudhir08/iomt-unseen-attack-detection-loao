"""U13 (author-approved 23 Nov 2026): inference-only re-export of the 2-, 6- and 19-class head outputs for the 75 canonical LOAO runs. No training, no result file modified.
For each run: loads the saved model.pt, rebuilds the run's test windows with the project's own pipeline (same code path as scripts/p4_export_scores.py), and writes
results_v3_paper3/score_export_coarse_<date>/<run>.npz with y19, hid, msp2/msp6/msp19 (float32, max of the head's softmax probabilities), energy2/energy6/energy19
(float32, logsumexp of the head's logits, T=1, same convention as energy_score: higher = more known; NaN for a head the model does not have), logits2 and logits6 (float32,
only for models with those heads) and a consistency record against the existing score_export (y19, hid, energy19, msp19). Mixed precision as in the original export.
Usage: python scripts/p4_rev2_13b_coarse_head_export.py --date 20261123"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trace_iomt.config import load_config  # noqa: E402
from trace_iomt.experiment import load_prepared  # noqa: E402
from trace_iomt.data.windows import build_windowed  # noqa: E402
from trace_iomt.models.factory import build_model  # noqa: E402
from trace_iomt.models.heads import energy_score  # noqa: E402
from trace_iomt.train import _to_tensors  # noqa: E402

MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = ["ARP_Spoofing", "Recon-VulScan", "MQTT-Malformed_Data", "TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
SEEDS = [42, 43, 44, 45, 46]
ZD = ROOT / "results_v3_paper3" / "zero_day"; EXP = ROOT / "results_v3_paper3" / "score_export"
LV = ["label2", "label6", "label19"]


@torch.no_grad()
def head_scores(model, wd, starts, device, amp, bs=1024):
    model.eval()
    use_amp = amp and device.type == "cuda"
    acc = {f"{k}{l[5:]}": [] for l in LV for k in ("msp", "energy")}
    lg = {"label2": [], "label6": []}; have = set()
    for i in range(0, len(starts), bs):
        x, p, _ = _to_tensors(wd, starts[i:i + bs], device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, _h = model(x, p)
        pr = model.heads.probabilities(logits)
        for l in LV:
            acc[f"msp{l[5:]}"].append(pr[l].max(-1).values.float().cpu().numpy())
            if l in logits:
                have.add(l)
                acc[f"energy{l[5:]}"].append(energy_score(logits[l]).cpu().numpy())
                if l in lg: lg[l].append(logits[l].float().cpu().numpy())
    out = {k: np.concatenate(v).astype(np.float32) for k, v in acc.items() if v}
    n = len(starts)
    for l in LV:
        if f"energy{l[5:]}" not in out: out[f"energy{l[5:]}"] = np.full(n, np.nan, np.float32)
    for l in lg:
        if lg[l]: out[f"logits{l[5:]}"] = np.concatenate(lg[l]).astype(np.float32)
    return out, sorted(have)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); a = ap.parse_args()
    OUT = ROOT / "results_v3_paper3" / f"score_export_coarse_{a.date}"
    OUT.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "project rule: GPU only"
    cfg = load_config(str(ROOT / "configs" / "paper3_openset.yaml"))
    device = torch.device("cuda"); amp = bool(cfg["train"]["amp"]); T = int(cfg["windows"]["length"])
    df = load_prepared(cfg)
    log = open(ROOT / "logs_v3_paper3" / f"coarse_export_{a.date}.log", "a", encoding="utf-8")
    t_all = time.time()
    def say(s):
        line = time.strftime("%Y-%m-%d %H:%M:%S") + " | " + s
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say(f"start; GPU {torch.cuda.get_device_name(0)}; torch {torch.__version__}; cuda {torch.version.cuda}; amp {amp}")
    for held in CLASSES:
        for seed in SEEDS:
            todo = [m for m in MODELS if not (OUT / f"{m}_loao_T16_{held}_seed{seed}.npz").exists()]
            if not todo: continue
            c2 = {**cfg, "split": {**cfg["split"], "regime": "loao", "loao_class": held}}
            wd = build_windowed(df, c2, regime="loao", T=T, seed=seed)
            hid = wd.hierarchy.index["label19"][held]
            y19 = wd.labels("test", "label19")
            for m in todo:
                run = f"{m}_loao_T16_{held}_seed{seed}"
                rd = ZD / run
                model = build_model(m, len(wd.features), wd.hierarchy, cfg["model"]).to(device)
                model.load_state_dict(torch.load(rd / "model.pt", map_location="cpu", weights_only=True))
                t0 = time.time()
                te, have = head_scores(model, wd, wd.starts["test"], device, amp)
                z = np.load(EXP / f"{run}.npz", allow_pickle=False)
                cons = dict(y19_equal=bool((z["y19"].astype(int) == y19.astype(int)).all()), hid_equal=bool(int(z["hid"]) == int(hid)),
                            energy19_maxabsdiff=float(np.abs(te["energy19"].astype(np.float64) - z["energy"].astype(np.float64)).max()),
                            msp19_maxabsdiff=float(np.abs(te["msp19"].astype(np.float64) - z["msp"].astype(np.float64)).max()), heads_with_logits=have)
                np.savez_compressed(OUT / f"{run}.npz", y19=y19.astype(np.int8), hid=hid, consistency=json.dumps(cons), **te)
                say(f"{run}: n_test={len(y19):,} heads {have}; y19_equal {cons['y19_equal']}; energy19 maxdiff {cons['energy19_maxabsdiff']:.2e}; msp19 maxdiff {cons['msp19_maxabsdiff']:.2e}; {time.time() - t0:.0f}s")
                del model
                torch.cuda.empty_cache()
            del wd; gc.collect()
    say(f"export finished; wall {time.time() - t_all:.0f}s; peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
