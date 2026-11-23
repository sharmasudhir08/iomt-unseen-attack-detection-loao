# R4: Benign-calibrated energy thresholds

**Status:** Exploratory, post hoc analysis; no change to H1-H5, their decision rules or verdicts. Test windows used for reporting only.

- Command: `python scripts/p4_rev_r4_benign_threshold.py --date 20261123`
- Script: `p4_rev_r4_benign_threshold.py`, SHA-256 `7ffb47138b589cb125c8e81abd4fd09713c0cd5c117afa0d72073d35eed56efa`
- Inputs: score_export/*.npz, zero_day/<run>/predictions.npz (val_y_true_label19), saved_output_analyses_20261118/per_run.csv, p4_review_analyses_20261120/T9/event_level_per_run.csv
- Canonical runs read: 75
- rng seed: 42
- Versions: Python 3.12.4, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1
- Start 2026-11-23 15:29:33, end 2026-11-23 15:29:54, duration 21 s

## Result
Checks: C1 PASS (max |thr_all - saved thr| = 4.77e-07); C2 PASS (val_energy and val_y_true_label19 have equal length in all 75 runs); C3 PASS (held-out caught-any at thr_all vs A2_caught_any max |diff| 1.11e-16; event-level values at thr_all vs saved 20 Nov T9 per-run values max |diff| 5.68e-14); Benign index PASS (label19 index of Benign is 1 in all runs).

Benign validation windows per run: min 1,806, median 1,806, max 1,806. A 1% threshold rests on about 18 tail windows (median run), so thr_bq1 is a noisy estimate.

Window order (C2): `val_energy` is written by scripts/p4_export_scores.py as scores(model, wd, wd.starts['val']) (batches of that start array, concatenated in order); `val_y_true_label19` is written by src/trace_iomt/experiment.py from evaluate(model, wd, 'val', ...) in src/trace_iomt/train.py, which uses starts = wd.starts['val'] and y_true = wd.y[lvl][starts + T - 1]. Both therefore follow the same start array; the lengths are equal in all 75 runs.

Design (fixed before any test result was seen): thr_all = 5th percentile of all validation energies (equals the saved thr); thr_bq = q-th percentile of the validation energies of Benign windows, q = 1, 2, 5; a window is flagged when energy < threshold; safety-view alert = flag OR p_attack > 0.5; event rules 2 of 4, 8 of 16 and 32 of 64 per capture file as in T9; summaries are means with seed intervals (seed_ci, 10,000 resamples, rng 42).

Pooled means over the 75 runs per model and threshold:

| Model | Threshold | Benign flag | Benign alert (safety view) | Held-out caught (energy only) | Held-out caught-any | Known-attack flag | Known false-alarm rate (flag) |
|---|---|---|---|---|---|---|---|
| trace_moe | reference | 0.8748 | 0.8833 | 0.5009 | 0.9836 | 0.0212 | 0.0418 |
| trace_moe | benign_q1 | 0.0201 | 0.0428 | 0.0989 | 0.7232 | 0.0023 | 0.0027 |
| trace_moe | benign_q2 | 0.0341 | 0.0560 | 0.1400 | 0.7549 | 0.0045 | 0.0052 |
| trace_moe | benign_q5 | 0.0687 | 0.0890 | 0.2024 | 0.8064 | 0.0113 | 0.0126 |
| transformer | reference | 0.9808 | 0.9973 | 0.5605 | 0.9988 | 0.0207 | 0.0440 |
| transformer | benign_q1 | 0.0136 | 0.0560 | 0.0980 | 0.7129 | 0.0010 | 0.0013 |
| transformer | benign_q2 | 0.0251 | 0.0673 | 0.1370 | 0.7453 | 0.0015 | 0.0021 |
| transformer | benign_q5 | 0.0576 | 0.0994 | 0.2018 | 0.8049 | 0.0020 | 0.0033 |
| bilstm_dnn | reference | 0.4623 | 0.5015 | 0.3764 | 0.9172 | 0.0407 | 0.0510 |
| bilstm_dnn | benign_q1 | 0.0157 | 0.0784 | 0.0455 | 0.6902 | 0.0053 | 0.0056 |
| bilstm_dnn | benign_q2 | 0.0288 | 0.0912 | 0.0613 | 0.7024 | 0.0059 | 0.0064 |
| bilstm_dnn | benign_q5 | 0.0679 | 0.1293 | 0.1164 | 0.7453 | 0.0072 | 0.0087 |

Event-level rules (means over the five classes and seeds):

| Model | Threshold | Rule | Benign events per 10,000 | Held-out windows triggering (full history) | Held-out files detected | Known-attack windows triggering |
|---|---|---|---|---|---|---|
| trace_moe | reference | 2 of 4 | 25.94 | 0.9965 | 1.00 | 1.0000 |
| trace_moe | reference | 8 of 16 | 8.58 | 0.9964 | 1.00 | 1.0000 |
| trace_moe | reference | 32 of 64 | 2.30 | 0.9978 | 1.00 | 1.0000 |
| trace_moe | benign_q1 | 2 of 4 | 61.19 | 0.7935 | 1.00 | 0.9995 |
| trace_moe | benign_q1 | 8 of 16 | 14.71 | 0.7576 | 1.00 | 0.9995 |
| trace_moe | benign_q1 | 32 of 64 | 2.70 | 0.7477 | 1.00 | 0.9997 |
| trace_moe | benign_q2 | 2 of 4 | 79.54 | 0.8271 | 1.00 | 0.9996 |
| trace_moe | benign_q2 | 8 of 16 | 20.15 | 0.7953 | 1.00 | 0.9996 |
| trace_moe | benign_q2 | 32 of 64 | 4.37 | 0.8032 | 1.00 | 0.9997 |
| trace_moe | benign_q5 | 2 of 4 | 113.91 | 0.8803 | 1.00 | 0.9997 |
| trace_moe | benign_q5 | 8 of 16 | 33.19 | 0.8611 | 1.00 | 0.9997 |
| trace_moe | benign_q5 | 32 of 64 | 7.93 | 0.8797 | 1.00 | 0.9998 |
| transformer | reference | 2 of 4 | 1.09 | 0.9999 | 1.00 | 1.0000 |
| transformer | reference | 8 of 16 | 0.28 | 1.0000 | 1.00 | 1.0000 |
| transformer | reference | 32 of 64 | 0.27 | 1.0000 | 1.00 | 1.0000 |
| transformer | benign_q1 | 2 of 4 | 58.39 | 0.7765 | 1.00 | 0.9996 |
| transformer | benign_q1 | 8 of 16 | 15.56 | 0.7241 | 1.00 | 0.9995 |
| transformer | benign_q1 | 32 of 64 | 3.21 | 0.7072 | 1.00 | 0.9997 |
| transformer | benign_q2 | 2 of 4 | 77.37 | 0.8113 | 1.00 | 0.9996 |
| transformer | benign_q2 | 8 of 16 | 20.71 | 0.7683 | 1.00 | 0.9996 |
| transformer | benign_q2 | 32 of 64 | 3.92 | 0.7634 | 1.00 | 0.9997 |
| transformer | benign_q5 | 2 of 4 | 106.13 | 0.8797 | 1.00 | 0.9998 |
| transformer | benign_q5 | 8 of 16 | 35.09 | 0.8547 | 1.00 | 0.9998 |
| transformer | benign_q5 | 32 of 64 | 9.26 | 0.8733 | 1.00 | 0.9999 |
| bilstm_dnn | reference | 2 of 4 | 212.45 | 0.9629 | 1.00 | 0.9999 |
| bilstm_dnn | reference | 8 of 16 | 64.07 | 0.9526 | 1.00 | 0.9999 |
| bilstm_dnn | reference | 32 of 64 | 18.67 | 0.9484 | 1.00 | 1.0000 |
| bilstm_dnn | benign_q1 | 2 of 4 | 145.96 | 0.7465 | 1.00 | 0.9990 |
| bilstm_dnn | benign_q1 | 8 of 16 | 31.97 | 0.6876 | 1.00 | 0.9989 |
| bilstm_dnn | benign_q1 | 32 of 64 | 8.08 | 0.6581 | 1.00 | 0.9991 |
| bilstm_dnn | benign_q2 | 2 of 4 | 158.26 | 0.7631 | 1.00 | 0.9991 |
| bilstm_dnn | benign_q2 | 8 of 16 | 31.86 | 0.7053 | 1.00 | 0.9990 |
| bilstm_dnn | benign_q2 | 32 of 64 | 7.29 | 0.6819 | 1.00 | 0.9992 |
| bilstm_dnn | benign_q5 | 2 of 4 | 207.16 | 0.8155 | 1.00 | 0.9993 |
| bilstm_dnn | benign_q5 | 8 of 16 | 41.75 | 0.7696 | 1.00 | 0.9992 |
| bilstm_dnn | benign_q5 | 32 of 64 | 9.85 | 0.7736 | 1.00 | 0.9994 |

TRACE-MoE per held-out class (mean over five seeds, 95% interval over seeds):

| Held-out class (TRACE-MoE) | Threshold | Benign flag [CI] | Benign alert [CI] | Held-out caught (energy) [CI] | Held-out caught-any [CI] |
|---|---|---|---|---|---|
| ARP_Spoofing | reference | 0.8164 [0.4595, 0.9998] | 0.8166 [0.4597, 1.0000] | 0.8172 [0.5940, 0.9727] | 0.9239 [0.7808, 0.9991] |
| ARP_Spoofing | benign_q1 | 0.0213 [0.0078, 0.0409] | 0.0220 [0.0087, 0.0415] | 0.2249 [0.0969, 0.3924] | 0.5158 [0.4326, 0.6330] |
| ARP_Spoofing | benign_q2 | 0.0358 [0.0195, 0.0552] | 0.0365 [0.0201, 0.0558] | 0.2945 [0.1328, 0.4794] | 0.5717 [0.4644, 0.7025] |
| ARP_Spoofing | benign_q5 | 0.0700 [0.0507, 0.0861] | 0.0706 [0.0515, 0.0866] | 0.3951 [0.1968, 0.5935] | 0.6554 [0.5240, 0.7910] |
| Recon-VulScan | reference | 0.9934 [0.9872, 0.9993] | 0.9999 [0.9998, 1.0000] | 0.7890 [0.7219, 0.8354] | 0.9998 [0.9994, 1.0000] |
| Recon-VulScan | benign_q1 | 0.0106 [0.0072, 0.0150] | 0.0319 [0.0240, 0.0384] | 0.1274 [0.0904, 0.1530] | 0.6600 [0.6268, 0.6933] |
| Recon-VulScan | benign_q2 | 0.0230 [0.0168, 0.0326] | 0.0441 [0.0354, 0.0528] | 0.1770 [0.1319, 0.2055] | 0.6998 [0.6679, 0.7297] |
| Recon-VulScan | benign_q5 | 0.0606 [0.0513, 0.0687] | 0.0817 [0.0704, 0.0923] | 0.2646 [0.2171, 0.2990] | 0.7746 [0.7431, 0.8022] |
| MQTT-Malformed_Data | reference | 0.8232 [0.6112, 0.9977] | 0.8335 [0.6323, 0.9997] | 0.8755 [0.7954, 0.9331] | 0.9941 [0.9858, 1.0000] |
| MQTT-Malformed_Data | benign_q1 | 0.0168 [0.0098, 0.0239] | 0.0390 [0.0304, 0.0483] | 0.1298 [0.0873, 0.1723] | 0.4404 [0.3901, 0.4860] |
| MQTT-Malformed_Data | benign_q2 | 0.0328 [0.0211, 0.0461] | 0.0547 [0.0430, 0.0664] | 0.2148 [0.1415, 0.2881] | 0.5033 [0.4553, 0.5543] |
| MQTT-Malformed_Data | benign_q5 | 0.0655 [0.0472, 0.0888] | 0.0869 [0.0685, 0.1089] | 0.3366 [0.2432, 0.4300] | 0.6018 [0.5580, 0.6548] |
| TCP_IP-DoS-UDP | reference | 0.7544 [0.3815, 0.9753] | 0.7780 [0.3906, 0.9986] | 0.0001 [0.0001, 0.0001] | 1.0000 [1.0000, 1.0000] |
| TCP_IP-DoS-UDP | benign_q1 | 0.0271 [0.0098, 0.0512] | 0.0633 [0.0358, 0.0937] | 0.0000 [0.0000, 0.0001] | 1.0000 [1.0000, 1.0000] |
| TCP_IP-DoS-UDP | benign_q2 | 0.0385 [0.0198, 0.0621] | 0.0740 [0.0467, 0.0997] | 0.0001 [0.0000, 0.0001] | 1.0000 [1.0000, 1.0000] |
| TCP_IP-DoS-UDP | benign_q5 | 0.0684 [0.0445, 0.0933] | 0.1026 [0.0690, 0.1305] | 0.0001 [0.0001, 0.0001] | 1.0000 [1.0000, 1.0000] |
| MQTT-DDoS-Publish_Flood | reference | 0.9867 [0.9642, 0.9996] | 0.9886 [0.9660, 1.0000] | 0.0225 [0.0085, 0.0398] | 1.0000 [0.9999, 1.0000] |
| MQTT-DDoS-Publish_Flood | benign_q1 | 0.0245 [0.0121, 0.0368] | 0.0578 [0.0347, 0.0911] | 0.0123 [0.0048, 0.0198] | 0.9999 [0.9998, 1.0000] |
| MQTT-DDoS-Publish_Flood | benign_q2 | 0.0406 [0.0215, 0.0621] | 0.0705 [0.0449, 0.1074] | 0.0137 [0.0054, 0.0220] | 0.9999 [0.9998, 1.0000] |
| MQTT-DDoS-Publish_Flood | benign_q5 | 0.0787 [0.0565, 0.1014] | 0.1033 [0.0810, 0.1315] | 0.0158 [0.0058, 0.0267] | 0.9999 [0.9998, 1.0000] |

## Notes
Exploratory and post hoc; no threshold is recommended; the thresholds use validation windows only and the test windows are used for reporting. H1 to H5 and their verdicts do not change. No training and no inference were run.
