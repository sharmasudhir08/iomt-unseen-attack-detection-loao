# T2: operating points and temperature grids

**Status:** Exploratory, post hoc analysis; no change to H1-H5, their decision rules or verdicts. Test windows used for reporting only.

- Command: `python scripts\p4_rev_2_operating_points.py`
- Script: `p4_rev_2_operating_points.py`, SHA-256 `e2f5cdc946c96811a64aae60a26254d0301503e0df9751b321dce0a4bcc1ff73`
- Inputs: score_export/*.npz; saved_output_analyses_20261118/per_run.csv
- Canonical runs read: 75
- rng seed: 42
- Versions: Python 3.12.4, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1
- Start 2026-11-20 11:32:13, end 2026-11-20 11:39:15, duration 423 s

## Result
Reproduction (max |difference| against per_run.csv A7/A18): {'energy T1': 1.1102230246251565e-16, 'msp T1': 1.1102230246251565e-16, 'energy T0.5': 1.1102230246251565e-16, 'energy T2': 1.1102230246251565e-16, 'energy T5': 1.1102230246251565e-16}; all <= 1e-9: PASS.
Temperature grids fixed now: energy T in {0.5, 1, 2, 5}, MSP T in {0.5, 1, 2, 5}; none is selected or recommended. FPR@95 = smallest FPR on known windows at TPR >= 0.95 on held-out windows.

Mean over classes and seeds:
```
             energy_auroc  msp_auroc  energy_fpr95  msp_fpr95  energy_oscr  msp_oscr  energy_aupr  msp_aupr
model                                                                                                      
bilstm_dnn         0.6980     0.7964        0.5988     0.5371       0.6878    0.7547       0.0578    0.0593
trace_moe          0.7111     0.7487        0.3596     0.4058       0.7061    0.7095       0.0435    0.0557
transformer        0.7442     0.7454        0.3154     0.4308       0.7379    0.7393       0.0713    0.0562
```
