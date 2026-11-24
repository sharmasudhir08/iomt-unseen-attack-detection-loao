# Paper 4 figures v6 (redrawn from saved results only)

Script `scripts/p4_figures_v6_20261124.py`, SHA-256 `d3814ff283a9db73d80320264cff3b964f219449150f817e7395a0a46e23250f`; Python 3.12.4, numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2; CPU only; 2026-11-24 10:07:56; 55 s.
No training and no new experiment; inputs: score_export (19-class logits), pcrp_cache_20261121, T5, T10, saved_output_analyses_20261118.

## Checks

- [PASS] fig_h3: 15 plotted means equal the energy-score AUROC column T = 1 of the temperature table (max |diff| 4.8e-05)
- [PASS] fig_h3: smallest text 8.0 pt (needs >= 8) (15 texts)
- [PASS] fig_h3: figure width 8.60 cm (target 8.6) (height 8.40 cm)
- [PASS] fig_h3: PDF page width 8.60 cm
- [PASS] fig_risk_coverage: CSV has 3 models x 5 classes x 2 scores x 20 coverage points (600 rows)
- [PASS] fig_risk_coverage: AURC values shown (mean of 25 runs) equal Table 15 / S12 'all classes' rows (TRACE-MoE energy 0.0336 TRACE-MoE msp 0.0392 Transformer energy 0.0293 Transformer msp 0.0251 BiLSTM-DNN energy 0.0259 BiLSTM-DNN msp 0.0245)
- [PASS] fig_risk_coverage: 25 runs per model and score in aurc_per_run.csv
- [PASS] fig_risk_coverage: smallest text 8.0 pt (needs >= 8) (43 texts)
- [PASS] fig_risk_coverage: figure width 14.50 cm (target 14.5) (height 5.60 cm)
- [PASS] fig_risk_coverage: PDF page width 14.50 cm
- [PASS] fig_reliability: 75 per-run ECE values recomputed from predictions.npz equal ece_per_run.csv (max |diff| 9.7e-17)
- [FAIL] fig_reliability: ECE (mean of 25 runs) equals 0.0160, 0.0079, 0.0071 (TRACE-MoE 0.0079 Transformer 0.0070 BiLSTM-DNN 0.0160)
- [PASS] fig_reliability: smallest text 8.0 pt (needs >= 8) (31 texts)
- [PASS] fig_reliability: figure width 14.50 cm (target 14.5) (height 5.60 cm)
- [PASS] fig_reliability: PDF page width 14.50 cm
- [PASS] fig_flood_subclusters: 10 silhouettes (2 classes x 5 seeds, k = 3) equal T10/silhouette.csv (max |diff| 5.6e-17)
- [PASS] fig_flood_subclusters: 90 cluster AUROCs (3 models x 3 clusters x 10 runs) equal T10/auroc_by_cluster.csv (90 values, max |diff| 1.1e-16)
- [PASS] fig_flood_subclusters: smallest text 8.0 pt (needs >= 8) (49 texts)
- [PASS] fig_flood_subclusters: figure width 17.80 cm (target 17.8) (height 8.40 cm)
- [PASS] fig_flood_subclusters: PDF page width 17.80 cm

## Information
- Risk-coverage: AURC of the plotted 20-point curves (trapezoid) against the per-run mean in the tables (information only): TRACE-MoE energy 0.0329 vs 0.0336; TRACE-MoE msp 0.0297 vs 0.0392; Transformer energy 0.0249 vs 0.0293; Transformer msp 0.0231 vs 0.0251; BiLSTM-DNN energy 0.0243 vs 0.0259; BiLSTM-DNN msp 0.0214 vs 0.0245
- Reliability: ECE of the pooled bins plotted (information only): TRACE-MoE 0.0079 (mean of 25 runs 0.0079); Transformer 0.0062 (mean of 25 runs 0.0070); BiLSTM-DNN 0.0159 (mean of 25 runs 0.0160)
