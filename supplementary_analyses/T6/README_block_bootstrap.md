# T6b block/cluster bootstrap (post-hoc exploratory)

Status: exploratory, post hoc; no change to H1-H5, their decision rules or verdicts. Saved artifacts only (score_export/, pcrp_cache_20261121); no training.

Script `scripts/p4_rev_t6b_block_bootstrap.py` SHA-256 00b91301f26b6ef67ef4d73564c66a7c29bc834bab3599759c3eeacf7e7a4d15; Python 3.12.4, numpy 2.5.3, pandas 3.0.6; rng 42; 2000 resamples; run 2026-11-23 10:31:34.

Design: seeds resampled with replacement (paired across metrics); known test files resampled with replacement (cluster bootstrap over the known files); the held-out windows (a single file, in file order) cut into consecutive blocks of L windows and the blocks resampled with replacement. L = 1000 primary, 250 and 4000 sensitivity (all fixed before running). Fewer than 2 blocks: undefined (nan).

Checks: max |binned AUROC - exact AUROC| over all runs = 1.00e-04; max |point H1 mean - H1_safety_view.csv| = 1.11e-16; max |point H5 mean - twin_h5_verdict.csv| = 8.35e-06.
Seed-only intervals use 2,000 resamples here and may differ slightly from the paper's 10,000-resample intervals.

| held_out_class          | quantity                        |   point_mean_over_seeds |   seed_only_lo |   seed_only_hi |   block1000_lo |   block1000_hi |   block250_lo |   block250_hi |   block4000_lo |   block4000_hi |
|:------------------------|:--------------------------------|------------------------:|---------------:|---------------:|---------------:|---------------:|--------------:|--------------:|---------------:|---------------:|
| ARP_Spoofing            | H1_caught_any_trace_moe         |                  0.9239 |         0.7808 |         0.9992 |         0.7808 |         0.9995 |        0.7808 |        0.9993 |       nan      |       nan      |
| ARP_Spoofing            | H5_pcrp_auroc                   |                  0.9215 |         0.9196 |         0.923  |         0.8545 |         0.9617 |        0.8639 |        0.9613 |       nan      |       nan      |
| ARP_Spoofing            | H2_energy_minus_msp_trace_moe   |                 -0.0027 |        -0.0276 |         0.0174 |        -0.0402 |         0.0258 |       -0.041  |        0.0252 |       nan      |       nan      |
| ARP_Spoofing            | H2_energy_minus_msp_transformer |                  0.0207 |         0.0109 |         0.0304 |         0.0097 |         0.0332 |        0.0091 |        0.0345 |       nan      |       nan      |
| ARP_Spoofing            | H2_energy_minus_msp_bilstm_dnn  |                 -0.1175 |        -0.1692 |        -0.0693 |        -0.1892 |        -0.0451 |       -0.1897 |       -0.0456 |       nan      |       nan      |
| Recon-VulScan           | H1_caught_any_trace_moe         |                  0.9998 |         0.9994 |         1      |         0.9994 |         1      |        0.999  |        1      |       nan      |       nan      |
| Recon-VulScan           | H5_pcrp_auroc                   |                  0.7656 |         0.7642 |         0.7674 |         0.6972 |         0.9449 |        0.6069 |        0.8976 |       nan      |       nan      |
| Recon-VulScan           | H2_energy_minus_msp_trace_moe   |                 -0.0023 |        -0.007  |         0.0028 |        -0.0196 |         0.0116 |       -0.0195 |        0.0156 |       nan      |       nan      |
| Recon-VulScan           | H2_energy_minus_msp_transformer |                  0.0398 |         0.0175 |         0.066  |         0.006  |         0.0666 |        0.0121 |        0.0764 |       nan      |       nan      |
| Recon-VulScan           | H2_energy_minus_msp_bilstm_dnn  |                 -0.1237 |        -0.1775 |        -0.0677 |        -0.231  |        -0.0383 |       -0.224  |       -0.0204 |       nan      |       nan      |
| MQTT-Malformed_Data     | H1_caught_any_trace_moe         |                  0.9941 |         0.9865 |         1      |         0.9858 |         1      |        0.9844 |        1      |       nan      |       nan      |
| MQTT-Malformed_Data     | H5_pcrp_auroc                   |                  0.8879 |         0.8857 |         0.8896 |         0.804  |         0.9447 |        0.8061 |        0.945  |       nan      |       nan      |
| MQTT-Malformed_Data     | H2_energy_minus_msp_trace_moe   |                 -0.0006 |        -0.0067 |         0.0055 |        -0.01   |         0.0092 |       -0.0103 |        0.0079 |       nan      |       nan      |
| MQTT-Malformed_Data     | H2_energy_minus_msp_transformer |                  0.0261 |         0.008  |         0.051  |         0.0033 |         0.0696 |        0.0028 |        0.0608 |       nan      |       nan      |
| MQTT-Malformed_Data     | H2_energy_minus_msp_bilstm_dnn  |                 -0.0649 |        -0.1194 |        -0.0182 |        -0.1673 |        -0.0045 |       -0.1693 |       -0.0036 |       nan      |       nan      |
| TCP_IP-DoS-UDP          | H1_caught_any_trace_moe         |                  1      |         1      |         1      |         1      |         1      |        1      |        1      |         1      |         1      |
| TCP_IP-DoS-UDP          | H5_pcrp_auroc                   |                  0.9513 |         0.9314 |         0.9712 |         0.886  |         0.9978 |        0.8861 |        0.9977 |         0.8864 |         0.9978 |
| TCP_IP-DoS-UDP          | H2_energy_minus_msp_trace_moe   |                 -0.1286 |        -0.2424 |        -0.0143 |        -0.3147 |         0.0445 |       -0.3145 |        0.0446 |        -0.3148 |         0.0446 |
| TCP_IP-DoS-UDP          | H2_energy_minus_msp_transformer |                  0.0888 |         0.001  |         0.2091 |        -0.0094 |         0.254  |       -0.0095 |        0.2539 |        -0.0093 |         0.2543 |
| TCP_IP-DoS-UDP          | H2_energy_minus_msp_bilstm_dnn  |                 -0.0775 |        -0.2297 |         0.0714 |        -0.3235 |         0.1391 |       -0.3236 |        0.1386 |        -0.3245 |         0.1389 |
| MQTT-DDoS-Publish_Flood | H1_caught_any_trace_moe         |                  1      |         0.9999 |         1      |         0.9999 |         1      |        0.9999 |        1      |         0.9999 |         1      |
| MQTT-DDoS-Publish_Flood | H5_pcrp_auroc                   |                  0.9361 |         0.936  |         0.9363 |         0.8842 |         0.9718 |        0.8838 |        0.9716 |         0.8838 |         0.9717 |
| MQTT-DDoS-Publish_Flood | H2_energy_minus_msp_trace_moe   |                 -0.0537 |        -0.1933 |         0.0858 |        -0.2269 |         0.1302 |       -0.2272 |        0.1316 |        -0.2252 |         0.131  |
| MQTT-DDoS-Publish_Flood | H2_energy_minus_msp_transformer |                 -0.1811 |        -0.2938 |        -0.0681 |        -0.3686 |        -0.0485 |       -0.3632 |       -0.0489 |        -0.3647 |        -0.0486 |
| MQTT-DDoS-Publish_Flood | H2_energy_minus_msp_bilstm_dnn  |                 -0.1082 |        -0.2703 |         0.054  |        -0.2934 |         0.0791 |       -0.2925 |        0.08   |        -0.2929 |         0.081  |

| held_out_class          |   n_heldout_files |   n_known_files |   n_blocks_L1000 |   n_blocks_L250 |   n_blocks_L4000 |   n_heldout_windows_seed42 |
|:------------------------|------------------:|----------------:|-----------------:|----------------:|-----------------:|---------------------------:|
| ARP_Spoofing            |                 1 |              20 |                2 |               7 |                1 |                       1729 |
| Recon-VulScan           |                 1 |              20 |                2 |               5 |                1 |                       1016 |
| MQTT-Malformed_Data     |                 1 |              20 |                2 |               7 |                1 |                       1732 |
| TCP_IP-DoS-UDP          |                 1 |              20 |              138 |             551 |               35 |                     137538 |
| MQTT-DDoS-Publish_Flood |                 1 |              20 |                9 |              34 |                3 |                       8401 |
