# U13: MSP and energy from the 2-, 6- and 19-class heads

**Status:** Exploratory, post hoc analysis; no change to H1-H5, their decision rules or verdicts. Test windows used for reporting only.

- Command: `python scripts/p4_rev2_13c_coarse_head_scores.py --date 20261123`
- Script: `p4_rev2_13c_coarse_head_scores.py`, SHA-256 `85016b3bfcfe8da74cc344a9a3cf80acd311bc959531e4d63f52db7bf0a786fc`
- Inputs: score_export_coarse_20261123/*.npz, score_export/*.npz, saved_output_analyses_20261118/per_run.csv
- Canonical runs read: 75
- rng seed: 42
- Versions: Python 3.12.4, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1
- Start 2026-11-23 15:26:35, end 2026-11-23 15:29:26, duration 171 s

## Result
Pass criterion C1: energy AUROC recomputed from score_export equals A7_energy_auroc within 1e-6: max |difference| 1.110e-16: PASS.
Pass criterion C2: the 19-class MSP AUROC recomputed from the new float32 export equals A7_msp_auroc within 1e-6: max |difference| 1.110e-16: PASS (the 19-class energy AUROC of the new export differs from A7 by at most 1.110e-16).

Data: inference-only re-export of the saved models (author-approved 23 Nov 2026), float32 head outputs, mixed precision as in the original export; `score_export_coarse_20261123/`. Models without separate 2- and 6-class logits (if any) have no energy for those heads; their MSP uses the summed child probabilities.
Method: held-out = y19 == hid, known = all other test windows; unknown score = -MSP or -energy; AUROC with ties averaged; tie share = fraction of windows with MSP exactly 1.0 (float32), held-out/known separately. Intervals: mean over five seeds, percentile bootstrap 95% (10,000 resamples, rng 42); differences paired over seeds (paired_diff_ci). Rule fixed now: an MSP cell is marked tie-dominated (`*`, `msp_tie_dominated_h*`) if either tie share is 0.5 or more.

AUROC per head (means over five seeds; * = MSP tie-dominated):

| Model | Held-out class | MSP AUROC 2 / 6 / 19-class | MSP ties held-out/known 2 | 6 | 19 | Energy AUROC 2 / 6 / 19-class |
|---|---|---|---|---|---|---|
| trace_moe | ARP_Spoofing | 0.9493* / 0.9334* / 0.9685* | 0.08/0.96 | 0.03/0.64 | 0.00/0.61 | 0.9712 / 0.9693 / 0.9658 |
| trace_moe | Recon-VulScan | 0.8156* / 0.8839* / 0.9715* | 0.35/0.97 | 0.10/0.67 | 0.00/0.64 | 0.9078 / 0.9759 / 0.9692 |
| trace_moe | MQTT-Malformed_Data | 0.9749* / 0.9754* / 0.9817* | 0.03/0.94 | 0.01/0.60 | 0.00/0.55 | 0.9860 / 0.9881 / 0.9810 |
| trace_moe | TCP_IP-DoS-UDP | 0.4544* / 0.3722* / 0.4425* | 0.97/0.88 | 0.62/0.41 | 0.58/0.46 | 0.5384 / 0.3387 / 0.3139 |
| trace_moe | MQTT-DDoS-Publish_Flood | 0.4938* / 0.3871* / 0.3794* | 0.97/0.96 | 0.96/0.73 | 0.92/0.68 | 0.3080 / 0.3981 / 0.3256 |
| transformer | ARP_Spoofing | 0.9895 / 0.9783 / 0.9614 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.9893 / 0.9871 / 0.9821 |
| transformer | Recon-VulScan | 0.8592 / 0.8765 / 0.9345 | 0.01/0.00 | 0.00/0.00 | 0.00/0.00 | 0.8479 / 0.9760 / 0.9743 |
| transformer | MQTT-Malformed_Data | 0.9497 / 0.9441 / 0.9555 | 0.01/0.00 | 0.00/0.00 | 0.00/0.00 | 0.9423 / 0.9834 / 0.9816 |
| transformer | TCP_IP-DoS-UDP | 0.2538 / 0.6734 / 0.5785 | 0.00/0.03 | 0.00/0.00 | 0.00/0.00 | 0.2992 / 0.5916 / 0.6673 |
| transformer | MQTT-DDoS-Publish_Flood | 0.2385 / 0.1583 / 0.2970 | 0.18/0.00 | 0.00/0.00 | 0.00/0.00 | 0.2079 / 0.5070 / 0.1159 |
| bilstm_dnn | ARP_Spoofing | 0.9949* / 0.9903* / 0.9749* | 0.00/0.97 | 0.00/0.83 | 0.00/0.63 | 0.9962 / 0.9856 / 0.8574 |
| bilstm_dnn | Recon-VulScan | 0.7918* / 0.7972* / 0.8515* | 0.39/0.97 | 0.23/0.64 | 0.13/0.52 | 0.9563 / 0.8206 / 0.7278 |
| bilstm_dnn | MQTT-Malformed_Data | 0.9555* / 0.9671* / 0.9707* | 0.07/0.97 | 0.04/0.89 | 0.00/0.78 | 0.9904 / 0.9603 / 0.9058 |
| bilstm_dnn | TCP_IP-DoS-UDP | 0.4840* / 0.6913* / 0.7425* | 1.00/0.97 | 0.52/0.87 | 0.24/0.65 | 0.7531 / 0.7579 / 0.6650 |
| bilstm_dnn | MQTT-DDoS-Publish_Flood | 0.5196* / 0.5414* / 0.4423* | 0.93/0.97 | 0.73/0.78 | 0.77/0.61 | 0.5201 / 0.6405 / 0.3341 |

Paired differences to the 19-class energy AUROC (seed bootstrap):

| Model | Held-out class | MSP2 - E19 [CI] | MSP6 - E19 [CI] | Energy2 - E19 [CI] | Energy6 - E19 [CI] |
|---|---|---|---|---|---|
| trace_moe | ARP_Spoofing | -0.0166 [-0.0676, 0.0383] | -0.0324 [-0.0692, 0.0028] | 0.0053 [-0.0304, 0.0464] | 0.0035 [-0.0042, 0.0112] |
| trace_moe | Recon-VulScan | -0.1536 [-0.1734, -0.1346] | -0.0852 [-0.0947, -0.0738] | -0.0614 [-0.0800, -0.0366] | 0.0068 [0.0012, 0.0135] |
| trace_moe | MQTT-Malformed_Data | -0.0062 [-0.0159, 0.0035] | -0.0056 [-0.0203, 0.0058] | 0.0050 [-0.0005, 0.0095] | 0.0070 [0.0032, 0.0110] |
| trace_moe | TCP_IP-DoS-UDP | 0.1405 [-0.0377, 0.2595] | 0.0583 [-0.0098, 0.1264] | 0.2246 [0.0370, 0.4871] | 0.0248 [0.0055, 0.0425] |
| trace_moe | MQTT-DDoS-Publish_Flood | 0.1682 [0.0604, 0.2759] | 0.0615 [-0.0738, 0.1968] | -0.0176 [-0.1361, 0.1054] | 0.0725 [-0.0339, 0.1895] |
| transformer | ARP_Spoofing | 0.0074 [0.0032, 0.0115] | -0.0038 [-0.0110, 0.0034] | 0.0072 [0.0029, 0.0116] | 0.0050 [-0.0004, 0.0106] |
| transformer | Recon-VulScan | -0.1151 [-0.1511, -0.0721] | -0.0978 [-0.1269, -0.0784] | -0.1264 [-0.1454, -0.1049] | 0.0017 [-0.0013, 0.0049] |
| transformer | MQTT-Malformed_Data | -0.0319 [-0.0692, -0.0010] | -0.0375 [-0.0756, -0.0042] | -0.0393 [-0.0764, -0.0100] | 0.0018 [0.0005, 0.0034] |
| transformer | TCP_IP-DoS-UDP | -0.4135 [-0.5847, -0.2342] | 0.0061 [-0.0880, 0.0822] | -0.3681 [-0.5335, -0.1900] | -0.0758 [-0.1435, -0.0080] |
| transformer | MQTT-DDoS-Publish_Flood | 0.1226 [-0.0786, 0.3239] | 0.0424 [-0.0569, 0.1437] | 0.0920 [-0.0977, 0.3046] | 0.3911 [0.2315, 0.6318] |
| bilstm_dnn | ARP_Spoofing | 0.1374 [0.0931, 0.1877] | 0.1329 [0.0888, 0.1825] | 0.1387 [0.0949, 0.1884] | 0.1282 [0.0837, 0.1773] |
| bilstm_dnn | Recon-VulScan | 0.0639 [-0.0066, 0.1345] | 0.0693 [0.0006, 0.1381] | 0.2285 [0.1453, 0.2983] | 0.0928 [0.0377, 0.1335] |
| bilstm_dnn | MQTT-Malformed_Data | 0.0497 [0.0139, 0.1073] | 0.0613 [0.0258, 0.1139] | 0.0846 [0.0374, 0.1490] | 0.0545 [0.0286, 0.0804] |
| bilstm_dnn | TCP_IP-DoS-UDP | -0.1810 [-0.3241, -0.0378] | 0.0263 [-0.0037, 0.0551] | 0.0881 [-0.1880, 0.3186] | 0.0929 [0.0026, 0.1884] |
| bilstm_dnn | MQTT-DDoS-Publish_Flood | 0.1855 [0.0499, 0.3088] | 0.2073 [0.0690, 0.3456] | 0.1859 [0.0939, 0.3008] | 0.3063 [0.1444, 0.4663] |

Flood classes:

- trace_moe / TCP_IP-DoS-UDP / 2-class MSP minus 19-class energy: +0.1405 [-0.0377, +0.2595] (interval includes 0); MSP tie shares held-out/known 0.97/0.88 (tie-dominated)
- trace_moe / TCP_IP-DoS-UDP / 6-class MSP minus 19-class energy: +0.0583 [-0.0098, +0.1264] (interval includes 0); MSP tie shares held-out/known 0.62/0.41 (tie-dominated)
- trace_moe / TCP_IP-DoS-UDP / 2-class energy minus 19-class energy: +0.2246 [+0.0370, +0.4871] (interval above 0)
- trace_moe / TCP_IP-DoS-UDP / 6-class energy minus 19-class energy: +0.0248 [+0.0055, +0.0425] (interval above 0)
- trace_moe / MQTT-DDoS-Publish_Flood / 2-class MSP minus 19-class energy: +0.1682 [+0.0604, +0.2759] (interval above 0); MSP tie shares held-out/known 0.97/0.96 (tie-dominated)
- trace_moe / MQTT-DDoS-Publish_Flood / 6-class MSP minus 19-class energy: +0.0615 [-0.0738, +0.1968] (interval includes 0); MSP tie shares held-out/known 0.96/0.73 (tie-dominated)
- trace_moe / MQTT-DDoS-Publish_Flood / 2-class energy minus 19-class energy: -0.0176 [-0.1361, +0.1054] (interval includes 0)
- trace_moe / MQTT-DDoS-Publish_Flood / 6-class energy minus 19-class energy: +0.0725 [-0.0339, +0.1895] (interval includes 0)
- transformer / TCP_IP-DoS-UDP / 2-class MSP minus 19-class energy: -0.4135 [-0.5847, -0.2342] (interval below 0); MSP tie shares held-out/known 0.00/0.03
- transformer / TCP_IP-DoS-UDP / 6-class MSP minus 19-class energy: +0.0061 [-0.0880, +0.0822] (interval includes 0); MSP tie shares held-out/known 0.00/0.00
- transformer / TCP_IP-DoS-UDP / 2-class energy minus 19-class energy: -0.3681 [-0.5335, -0.1900] (interval below 0)
- transformer / TCP_IP-DoS-UDP / 6-class energy minus 19-class energy: -0.0758 [-0.1435, -0.0080] (interval below 0)
- transformer / MQTT-DDoS-Publish_Flood / 2-class MSP minus 19-class energy: +0.1226 [-0.0786, +0.3239] (interval includes 0); MSP tie shares held-out/known 0.18/0.00
- transformer / MQTT-DDoS-Publish_Flood / 6-class MSP minus 19-class energy: +0.0424 [-0.0569, +0.1437] (interval includes 0); MSP tie shares held-out/known 0.00/0.00
- transformer / MQTT-DDoS-Publish_Flood / 2-class energy minus 19-class energy: +0.0920 [-0.0977, +0.3046] (interval includes 0)
- transformer / MQTT-DDoS-Publish_Flood / 6-class energy minus 19-class energy: +0.3911 [+0.2315, +0.6318] (interval above 0)
- bilstm_dnn / TCP_IP-DoS-UDP / 2-class MSP minus 19-class energy: -0.1810 [-0.3241, -0.0378] (interval below 0); MSP tie shares held-out/known 1.00/0.97 (tie-dominated)
- bilstm_dnn / TCP_IP-DoS-UDP / 6-class MSP minus 19-class energy: +0.0263 [-0.0037, +0.0551] (interval includes 0); MSP tie shares held-out/known 0.52/0.87 (tie-dominated)
- bilstm_dnn / TCP_IP-DoS-UDP / 2-class energy minus 19-class energy: +0.0881 [-0.1880, +0.3186] (interval includes 0)
- bilstm_dnn / TCP_IP-DoS-UDP / 6-class energy minus 19-class energy: +0.0929 [+0.0026, +0.1884] (interval above 0)
- bilstm_dnn / MQTT-DDoS-Publish_Flood / 2-class MSP minus 19-class energy: +0.1855 [+0.0499, +0.3088] (interval above 0); MSP tie shares held-out/known 0.93/0.97 (tie-dominated)
- bilstm_dnn / MQTT-DDoS-Publish_Flood / 6-class MSP minus 19-class energy: +0.2073 [+0.0690, +0.3456] (interval above 0); MSP tie shares held-out/known 0.73/0.78 (tie-dominated)
- bilstm_dnn / MQTT-DDoS-Publish_Flood / 2-class energy minus 19-class energy: +0.1859 [+0.0939, +0.3008] (interval above 0)
- bilstm_dnn / MQTT-DDoS-Publish_Flood / 6-class energy minus 19-class energy: +0.3063 [+0.1444, +0.4663] (interval above 0)

## Notes
Float32 export: MSP values are float32, so the float16 rounding of the stored probabilities does not apply; ties at exactly 1.0 remain where the softmax saturates in float32, and the tie share is reported next to each MSP AUROC. Exploratory and post hoc; H1 to H5 and their verdicts do not change. Nothing was trained.
