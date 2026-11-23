# Unseen-Attack Detection in the Internet of Medical Things: leave-one-attack-out evaluation

Companion repository of the manuscript "Unseen-Attack Detection in the Internet of Medical Things: A Leave-One-Attack-Out Evaluation of Sequence Models, Energy Scores and Feature-Space Profiles" (S. K. Sharma and S. Vasishth).

## For reviewers

Every table and figure of the manuscript (Tables 1 to 5, Figures 1 and 2) and of the supplementary material (Tables S1 to S15, Figures S1 to S7) is listed below with its source file(s) in this repository and the script that produced them. The raw data are at the CIC repository (see Dataset below); the model files and the large per-window exports are not included because of their size. The supplementary material source is in `supplementary_material/`.

| Manuscript or supplement item | Source file(s) | Script |
|---|---|---|
| Table 1 (H1, safety view) | `results/H1_safety_view.csv`; `supplementary_analyses/from_20261118/saved_output_analyses/per_run.csv` | `scripts/paper4_h8_verdict.py`; `scripts/07_unknown_attacks.py` |
| Table 2 (H3, architecture comparison) | `results/H3_trace_moe_vs_baselines.csv` | `scripts/paper4_h8_verdict.py` |
| Table 3 (H2, energy versus MSP) | `results/H2_energy_vs_msp.csv` | `scripts/paper4_h8_verdict.py` |
| Table 4 (H4, closed-set degradation) | `results/H4_closed_set_degradation.csv` | `scripts/paper4_h8_verdict.py` |
| Table 5 (H5, PCRP) | `results/twin_h5_verdict.csv` | `scripts/24_twin_score_paper4.py`; `scripts/25_twin_verdict_paper4.py` |
| Figure 1 (architecture) | `figures/architecture_schematic.png` | n/a |
| Figure 2 (H3 energy AUROC) | `figures/fig2_h3_architecture_comparison.pdf`; `results/H3_trace_moe_vs_baselines.csv` | `scripts/paper4_figures_v1_4.py` |
| Table S1 (training configuration) | `supplementary_analyses/S1_config/config_values_trace_moe_ARP_Spoofing_seed42.json`; `src/trace_iomt/train.py` | `scripts/07_unknown_attacks.py` |
| Table S2 (test-window counts) | `supplementary_analyses/T9/safety_view_false_alarms.csv` | `scripts_supplementary/p4_rev_t5_t9.py` |
| Table S3 (inference benchmark) | `supplementary_analyses/from_20261118/inference_benchmark/benchmark_summary.csv`; `supplementary_analyses/from_20261118/inference_benchmark/environment.json` | `scripts_supplementary/p4_inference_benchmark.py` |
| Table S4 (safety-view decomposition) | `supplementary_analyses/from_20261118/saved_output_analyses/summary_by_model_class.csv` | `scripts_supplementary/p4_analyze_exports.py`; `scripts_supplementary/p4_export_scores.py` |
| Table S5 (decomposition per seed) | `supplementary_analyses/from_20261118/saved_output_analyses/per_run.csv` | `scripts_supplementary/p4_analyze_exports.py` |
| Table S6 (ESA1 calibration) | `results/ESA1_calibration.csv` | `scripts/paper4_h8_verdict.py` |
| Table S7 (Benign alert rates; event-level rule) | `supplementary_analyses/from_20261120/V4/benign_flag_rate_per_run.csv`; `supplementary_analyses/T9/safety_view_false_alarms.csv`; `supplementary_analyses/from_20261120/T9/event_level.csv`; `supplementary_analyses/T9/event_level.csv` | `scripts_supplementary/p4_rev_v4_benign_disclosure.py`; `scripts_supplementary/p4_rev_t5_t9.py`; `scripts_supplementary/p4_rev_9_event_level.py` |
| Table S8 (ESA2 Spearman) | `supplementary_analyses/T11/esa2_spearman.csv` | `scripts_supplementary/p4_rev_saved_scores.py` |
| Table S9 (ESA3 further scores) | `supplementary_analyses/T3/summary_feature_scores.csv`; `supplementary_analyses/T3/paired_vs_energy.csv`; `supplementary_analyses/part2/pooled_table_for_manuscript.csv` | `scripts_supplementary/p4_rev_t3_feature_scores.py`; `scripts_supplementary/p4_rev_t3_summarize.py`; `scripts_supplementary/p4_rev2_10_logit_scores.py`; `scripts_supplementary/p4_rev2_2_head_disagreement.py`; `scripts_supplementary/p4_rev2_pooled_table.py` |
| Table S10 (temperature, operating points) | `supplementary_analyses/from_20261120/T2/temperature_energy_msp.csv`; `supplementary_analyses/from_20261120/T2/summary_operating_points.csv` | `scripts_supplementary/p4_rev_2_operating_points.py` |
| Table S11 (PCRP estimators) | `supplementary_analyses/T4/pcrp_variants_summary.csv` | `scripts_supplementary/p4_rev_pcrp_t4_u3.py`; `scripts_supplementary/p4_rev_t4_summarize.py` |
| Table S12 (ECE, AURC) | `supplementary_analyses/from_20261120/T5/ece_aurc_per_run.csv`; `supplementary_analyses/T5/summary_ece_aurc.csv`; `supplementary_analyses/T5/aurc_per_run.csv` | `scripts_supplementary/p4_rev_5_calibration_risk_coverage.py`; `scripts_supplementary/p4_rev_t5_t9.py` |
| Table S13 (block bootstrap; flood sub-clusters) | `supplementary_analyses/T6/block_bootstrap.csv`; `results/H1_safety_view.csv`; `results/H2_energy_vs_msp.csv`; `results/twin_h5_verdict.csv`; `supplementary_analyses/T10/silhouette.csv`; `supplementary_analyses/T10/auroc_by_cluster.csv` | `scripts_supplementary/p4_rev_t6b_block_bootstrap.py`; `scripts_supplementary/p4_rev_t10_flood_clusters.py` |
| Table S14 (coarser heads) | `supplementary_analyses/part2/U13/coarse_head_scores_summary.csv` | `scripts_supplementary/p4_rev2_13b_coarse_head_export.py`; `scripts_supplementary/p4_rev2_13c_coarse_head_scores.py` |
| Table S15 (Benign-calibrated thresholds) | `supplementary_analyses/R4/r4_per_run.csv`; `supplementary_analyses/R4/r4_summary.csv`; `supplementary_analyses/R4/r4_event_level.csv` | `scripts_supplementary/p4_rev_r4_benign_threshold.py` |
| Figures S1 and S2 (reliability, risk-coverage) | `supplementary_analyses/from_20261120/T5/figures/reliability_19class.pdf`; `supplementary_analyses/from_20261120/T5/figures/risk_coverage.pdf` | `scripts_supplementary/p4_rev_5_calibration_risk_coverage.py` |
| Figure S3 (flood sub-clusters) | `supplementary_analyses/T10/figures/pca_clusters_k3.pdf` | `scripts_supplementary/p4_rev_t10_flood_clusters.py` |
| Figures S4 to S7 (H1, H2, H4, H5) | `figures/fig1_h1_safety_view.pdf`; `figures/fig3_h2_energy_vs_msp.pdf`; `figures/fig4_h4_closed_set_degradation.pdf`; `figures/fig_y_h5_twin_auroc.pdf` | `scripts/paper4_figures_v1_4.py`; `scripts/paper4_figure_y_h5_twin.py` |

## What this study does
TRACE-MoE (a mixture-of-experts sequence model), a Transformer and a BiLSTM-DNN are evaluated under a leave-one-attack-out (LOAO) open-set protocol on the Wi-Fi and MQTT subset of CICIoMT2024: five held-out attack classes, three architectures, five seeds (42 to 46), 75 independently trained models. Five hypotheses (H1 to H5) are evaluated with paired bootstrap 95% confidence intervals over seeds, plus exploratory supplementary analyses.

## Verify a single number
1. Find a claim in the manuscript (for example the H1 detection rate for ARP spoofing, mean 0.9239).
2. Open `results/H1_safety_view.csv` (or `H2_energy_vs_msp.csv`, `H3_trace_moe_vs_baselines.csv`, `H4_closed_set_degradation.csv`, `twin_h5_verdict.csv`, `ESA1_calibration.csv`) and read the row for that class and model.
3. Compare with the `mean`, `ci_lo` and `ci_hi` columns.
4. `TRACEABILITY.md` maps every table and figure to its source file and reports the automated check of all table values; `SHASUMS` lists the SHA-256 of every file in this folder.

## Layout
| Path | Content |
|---|---|
| `results/` | final hypothesis tables (CSV and markdown), per-seed values, verdict files, `S1_before_after` |
| `figures/` | figures of the manuscript and their source files (`fig1` to `fig4` and `fig_y_h5`, PDF and PNG; `architecture_schematic.png` is the architecture figure) |
| `supplementary_analyses/` | post hoc exploratory analyses with a README each (see `SUPPLEMENTARY_ANALYSES.md`) |
| `scripts_supplementary/` | scripts of the supplementary analyses (and the score-export script they read) |
| `supplementary_material/` | the supplementary material of the manuscript as a stand-alone Overleaf project (`paper4_supplementary_overleaf_20261123_v4b/`: main tex file, `tables/` S1 to S15, `figures/` S1 to S7) and its zip; the tables are generated from the CSV files of `results/` and `supplementary_analyses/` |
| `ANALYSIS_PLAN.md` | hypotheses, decision rules and timeline of the analysis plan |
| `TRACEABILITY.md` | table-by-table and figure-by-figure source map, check results, reference list |
| `graphical_abstract.png`, `graphical_abstract.pdf` | graphical abstract |
| `src/trace_iomt/` | model, data-pipeline, training and abstention code imported by the scripts (17 modules) |
| `scripts/` | `07_unknown_attacks.py` (leave-one-attack-out training and evaluation, GPU), `24_twin_score_paper4.py` and `25_twin_verdict_paper4.py` (H5 profile score and verdict), `paper4_h8_verdict.py` (H1 to H4 and ESA1 aggregation and verdicts), `paper4_build_table.py`, `paper4_figures_v1_4.py`, `paper4_figure_y_h5_twin.py`, `_common.py` |
| `requirements.txt` | Python dependencies |
| `CITATION.cff` | citation metadata |

## Dataset
CICIoMT2024 (Wi-Fi and MQTT subset), Dadkhah et al. 2024, Internet of Things 28, DOI 10.1016/j.iot.2024.101351; also via IEEE DataPort, DOI 10.21227/tq0p-ag21. The raw data are not redistributed.

## Not included
Model checkpoints, per-run predictions and training logs, preprocessed caches and the large per-window exports are not included because of their size; they are reproducible from the code. The training settings are given in Section IV-C of the manuscript.

## Citation and license
See `CITATION.cff` and `LICENSE`.

## Added analyses (exploratory)
`supplementary_analyses/part2/U13/` (MSP and energy of the 2-, 6- and 19-class heads from float32 head outputs of the saved models; the export itself is large and not included) and `supplementary_analyses/R4/` (abstention thresholds from the validation energy of Benign windows only, with event-level rules) are post hoc exploratory analyses with a README each. They change no verdict of H1 to H5 and recommend no threshold. Scripts: `scripts_supplementary/p4_rev2_13b_coarse_head_export.py`, `p4_rev2_13c_coarse_head_scores.py`, `p4_rev_r4_benign_threshold.py`.
