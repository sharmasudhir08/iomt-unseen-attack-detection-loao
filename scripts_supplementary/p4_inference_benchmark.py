"""Paper 4 dedicated inference benchmark (D3). Inference only: no training, no result file is modified.

Measures per-window inference time of the three architectures (TRACE-MoE, Transformer, BiLSTM-DNN) on the
RTX 3050 laptop GPU under controlled conditions: AC power (checked every round), a single process, warm-up
iterations, CUDA synchronisation around every timed call, several rounds, and three batch sizes.

Weights: the canonical seed-42 ARP_Spoofing run of each architecture (all LOAO runs of an architecture have the
same architecture and parameter count, so timing does not depend on which run is loaded).

Inputs (two modes, both reported):
  real       : 16,384 random test windows of the ARP_Spoofing LOAO split (seed 42), built with the project's own
               windowing; needed because TRACE-MoE's top-1 dispatch is data dependent.
  synthetic  : standard-normal (B,16,45) features with random protocol tokens (shape-only control).

Two timing scopes per batch size:
  gpu_forward : inputs already on the GPU; model forward + head probabilities + energy score; synchronised.
  end_to_end  : host numpy windows -> GPU -> forward -> probabilities and energy -> host numpy (data transfer included).

Output: results_v3_paper3/inference_benchmark_20261118/{benchmark_rounds.csv, benchmark_summary.csv, benchmark_summary.md, environment.json}
"""
from __future__ import annotations
import argparse, ctypes, gc, json, os, platform, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ["P4_RELEASE_TABLE"] = "1"

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trace_iomt.config import load_config  # noqa: E402
from trace_iomt.data.labels import Hierarchy  # noqa: E402
from trace_iomt.models.factory import build_model, count_parameters  # noqa: E402
from trace_iomt.models.heads import energy_score  # noqa: E402

MODELS = ["bilstm_dnn", "transformer", "trace_moe"]
OUT = ROOT / "results_v3_paper3" / "inference_benchmark_20261118"
ZD = ROOT / "results_v3_paper3" / "zero_day"
RUN = "{m}_loao_T16_ARP_Spoofing_seed42"


class PowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte), ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte), ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def on_ac() -> bool:
    s = PowerStatus()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    return s.ACLineStatus == 1


def gpu_state() -> dict:
    q = "clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu,memory.used"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    v = [x.strip() for x in out.split(",")]
    return dict(zip(q.split(","), v))


def other_training_processes() -> list[str]:
    cmd = ["powershell", "-NoProfile", "-Command",
           "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match '07_unknown|p4_retrain|24_twin' } | ForEach-Object { $_.ProcessId }"]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.split()


def load_model(name, hier, n_feat, cfg, device):
    m = build_model(name, n_feat, hier, cfg["model"])
    sd = torch.load(ZD / RUN.format(m=name) / "model.pt", map_location="cpu")
    m.load_state_dict(sd)
    return m.to(device).eval()


def real_windows(cfg, n, seed=0):
    from trace_iomt.experiment import load_prepared, build_run_windows
    df = load_prepared(cfg)
    c2 = {**cfg, "split": {**cfg["split"], "regime": "loao", "loao_class": "ARP_Spoofing"}}
    holder = [df]
    del df
    from trace_iomt.data.windows import build_windowed
    wd = build_windowed(holder, c2, regime="loao", T=int(cfg["windows"]["length"]), seed=42, low_memory=True)
    del holder
    gc.collect()
    rng = np.random.default_rng(seed)
    st = wd.starts["test"]
    pick = np.sort(rng.choice(st, size=min(n, len(st)), replace=False))
    x, p, _ = wd.gather(pick)
    feats = list(wd.features)
    del wd
    gc.collect()
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(p), feats


@torch.no_grad()
def step(model, x, p, use_amp, device):
    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits, _ = model(x, p)
    pr = model.heads.probabilities(logits)
    e = energy_score(logits["label19"])
    return pr, e


def time_scope(model, X, P, bs, scope, device, use_amp, warm, max_batches):
    n = (len(X) // bs) * bs
    nb = min(n // bs, max_batches)
    if scope == "gpu_forward":
        xg = torch.from_numpy(X[: nb * bs]).to(device)
        pg = torch.from_numpy(P[: nb * bs]).to(device)
    # warm-up
    for i in range(warm):
        j = (i % nb) * bs
        if scope == "gpu_forward":
            step(model, xg[j:j + bs], pg[j:j + bs], use_amp, device)
        else:
            xt = torch.from_numpy(X[j:j + bs]).to(device); pt = torch.from_numpy(P[j:j + bs]).to(device)
            pr, e = step(model, xt, pt, use_amp, device)
            _ = e.cpu().numpy()
    torch.cuda.synchronize()
    times = []
    for b in range(nb):
        j = b * bs
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if scope == "gpu_forward":
            pr, e = step(model, xg[j:j + bs], pg[j:j + bs], use_amp, device)
            torch.cuda.synchronize()
        else:
            xt = torch.from_numpy(X[j:j + bs]).to(device); pt = torch.from_numpy(P[j:j + bs]).to(device)
            pr, e = step(model, xt, pt, use_amp, device)
            _ = [v.cpu().numpy() for v in pr.values()]; _ = e.cpu().numpy()
        times.append(time.perf_counter() - t0)
    t = np.array(times)
    return t.sum() / (nb * bs), t / bs, nb


def time_cpu(model, X, P, bs, warm, max_batches):
    """Single-thread CPU, end to end (host numpy -> tensor -> forward -> probabilities and energy -> numpy), float32, no autocast."""
    cpu = torch.device("cpu")
    n = (len(X) // bs) * bs
    nb = min(n // bs, max_batches)
    with torch.no_grad():
        for i in range(warm):
            j = (i % nb) * bs
            pr, e = step(model, torch.from_numpy(X[j:j + bs]), torch.from_numpy(P[j:j + bs]), False, cpu)
        times = []
        for b in range(nb):
            j = b * bs
            t0 = time.perf_counter()
            pr, e = step(model, torch.from_numpy(X[j:j + bs]), torch.from_numpy(P[j:j + bs]), False, cpu)
            _ = [v.numpy() for v in pr.values()]; _ = e.numpy()
            times.append(time.perf_counter() - t0)
    t = np.array(times)
    return t.sum() / (nb * bs), t / bs, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--n-real", type=int, default=16384)
    ap.add_argument("--no-real", action="store_true", help="synthetic input only")
    ap.add_argument("--out-dir", default=None, help="smoke runs: write elsewhere")
    ap.add_argument("--allow-battery", action="store_true", help="NOT for reported numbers")
    a = ap.parse_args()

    assert torch.cuda.is_available(), "project rule: GPU only"
    if not on_ac() and not a.allow_battery:
        sys.exit("ABORT: not on AC power.")
    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    ms = MemStatus(); ms.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    free_gb = ms.ullAvailPhys / 2**30
    if not a.no_real and free_gb < 5.0:
        sys.exit(f"ABORT: free RAM {free_gb:.1f} GB < 5 GB needed to build the real test windows.")
    other = other_training_processes()
    if other:
        sys.exit(f"ABORT: other project processes running: {other}")
    out = Path(a.out_dir) if a.out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "paper3_openset.yaml"))
    device = torch.device("cuda")
    use_amp = bool(cfg["train"]["amp"])
    T = int(cfg["windows"]["length"])

    first = json.loads((ZD / RUN.format(m="trace_moe") / "metrics.json").read_text(encoding="utf-8"))
    cl = first["hierarchy"]["classes"]["label19"]
    l6 = first["hierarchy"]["classes"]["label6"]
    p19 = first["hierarchy"]["parent19"]
    hier = Hierarchy(cl, {c: l6[p19[i]] for i, c in enumerate(cl)})

    inputs = {}
    if not a.no_real:
        t0 = time.time()
        X, P, feats = real_windows(cfg, a.n_real)
        inputs["real"] = (X, P)
        n_feat = len(feats)
        print(f"[real] windows {X.shape}, built in {time.time() - t0:.0f} s")
    else:
        n_feat = 45
    rng = np.random.default_rng(1)
    Xs = rng.standard_normal((a.n_real, T, n_feat)).astype(np.float32)
    Ps = rng.integers(0, int(inputs["real"][1].max()) + 1 if "real" in inputs else 5, size=(a.n_real, T)).astype(inputs["real"][1].dtype if "real" in inputs else np.int64)
    inputs["synthetic"] = (Xs, Ps)

    models = {m: load_model(m, hier, n_feat, cfg, device) for m in MODELS}
    cpu_models = {m: load_model(m, hier, n_feat, cfg, torch.device("cpu")) for m in MODELS}
    params = {m: count_parameters(models[m]) for m in MODELS}
    print("parameters:", params)
    env = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), platform=platform.platform(), amp=use_amp, params=params, ac_power_start=on_ac(),
               gpu_state_start=gpu_state(), rounds=a.rounds, input_modes=list(inputs), weights_from=RUN)
    # TRACE-MoE routing utilisation on each input mode (dispatch is data dependent)
    util = {}
    for mode, (X, P) in inputs.items():
        with torch.no_grad():
            r = []
            for j in range(0, min(len(X), 4096), 1024):
                step(models["trace_moe"], torch.from_numpy(X[j:j + 1024]).to(device), torch.from_numpy(P[j:j + 1024]).to(device), use_amp, device)
                r.append(models["trace_moe"].last_route.cpu().numpy())
            r = np.concatenate(r)
            util[mode] = np.bincount(r, minlength=4).tolist()
    env["trace_moe_expert_counts_first_4096"] = util
    print("TRACE-MoE expert counts:", util)

    BS = {1: dict(warm=100, max_batches=1500), 64: dict(warm=50, max_batches=256), 1024: dict(warm=20, max_batches=16)}
    rows = []
    for rnd in range(1, a.rounds + 1):
        if not on_ac() and not a.allow_battery:
            sys.exit(f"ABORT in round {rnd}: AC power lost.")
        order = MODELS if rnd % 2 else MODELS[::-1]
        for mode, (X, P) in inputs.items():
            for m in order:
                for bs, kw in BS.items():
                    for scope in ("gpu_forward", "end_to_end"):
                        mean_s, per_win, nb = time_scope(models[m], X, P, bs, scope, device, use_amp, kw["warm"], kw["max_batches"])
                        g = gpu_state()
                        rows.append(dict(round=rnd, mode=mode, model=m, batch=bs, scope=scope, n_batches=nb,
                                         ms_per_window_mean=mean_s * 1e3, ms_per_window_median=float(np.median(per_win)) * 1e3,
                                         ms_per_window_p95=float(np.percentile(per_win, 95)) * 1e3,
                                         windows_per_s=1.0 / mean_s, gpu_clock_mhz=g["clocks.sm"], gpu_temp_c=g["temperature.gpu"],
                                         ac=on_ac()))
        # single-thread CPU scope (same windows, end to end), batch 1 and 1024
        orig_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        for mode, (X, P) in inputs.items():
            for m in order:
                for bs, kw in ((1, dict(warm=50, max_batches=400)), (1024, dict(warm=1, max_batches=6))):
                    mean_s, per_win, nb = time_cpu(cpu_models[m], X, P, bs, kw["warm"], kw["max_batches"])
                    rows.append(dict(round=rnd, mode=mode, model=m, batch=bs, scope="cpu_1thread_end_to_end", n_batches=nb,
                                     ms_per_window_mean=mean_s * 1e3, ms_per_window_median=float(np.median(per_win)) * 1e3,
                                     ms_per_window_p95=float(np.percentile(per_win, 95)) * 1e3, windows_per_s=1.0 / mean_s,
                                     gpu_clock_mhz="", gpu_temp_c="", ac=on_ac()))
        torch.set_num_threads(orig_threads)
        print(f"round {rnd}/{a.rounds} done; gpu {gpu_state()}")
    env["gpu_state_end"] = gpu_state(); env["ac_power_end"] = on_ac()
    df = pd.DataFrame(rows)
    df.to_csv(out / "benchmark_rounds.csv", index=False)
    g = df.groupby(["mode", "scope", "batch", "model"]).agg(
        ms_mean=("ms_per_window_mean", "mean"), ms_sd_across_rounds=("ms_per_window_mean", "std"),
        ms_median_of_batches=("ms_per_window_median", "mean"), ms_p95_of_batches=("ms_per_window_p95", "mean"),
        windows_per_s=("windows_per_s", "mean"), n_rounds=("round", "nunique")).reset_index()
    g.to_csv(out / "benchmark_summary.csv", index=False)
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    L = ["# Inference benchmark summary (mean over rounds, per window)\n",
         f"AC power throughout: {bool(df.ac.all())}. Rounds: {a.rounds}. Environment: `environment.json`.\n",
         "| Input | Scope | Batch | Model | ms/window mean (SD across rounds) | windows/s |", "|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        L.append(f"| {r['mode']} | {r.scope} | {r.batch} | {r.model} | {r.ms_mean:.5f} ({r.ms_sd_across_rounds:.5f}) | {r.windows_per_s:,.0f} |")
    (out / "benchmark_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
