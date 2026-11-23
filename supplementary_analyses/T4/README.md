# T4 PCRP variants (post-hoc exploratory)

Status: exploratory, post hoc; no change to H1-H5, their decision rules or verdicts. Saved artifacts only; no training.

Computation (run 2026-11-21 11:55-12:25): `scripts/p4_rev_pcrp_t4_u3.py`, SHA-256 1328fbd64f1091c3b7b50914515a2eebf525f773803cbea105f4c62b07623bae; inputs: twin profiles and prepared test windows; per-seed raw output in `supplementary_analyses/T4/`.
Summary: `scripts/p4_rev_t4_summarize.py` SHA-256 6a3d98bf32e7e63ab000a2f37ae7e2f68d8c69323e99cc54b18453dde0a1f9f2; Python 3.12.4, numpy 2.5.3, pandas 3.0.6; bootstrap 10,000 resamples over 5 seeds, rng 42; 2026-11-23 10:29:25.

Pass checks: estimator status ok for 50 of 50 (class, seed, estimator) rows; reference (original estimator, last timestep) AUROC reproduces the saved per-seed AUROC to <= 2.3e-16 in the 21 Nov log; mean reference AUROC versus twin_h5_verdict.csv max |difference| = 4.55e-07.

Variants: covariance = original (reference), Ledoit-Wolf, MinCovDet; aggregation of D_min over the last k timesteps: last (k=1), and the mean and the minimum over the last k = 4 and k = 16 timesteps. No variant is declared best.

| held_out_class          | estimator   | aggregation   |   k | metric   |   n_seeds |   mean |     lo |     hi |
|:------------------------|:------------|:--------------|----:|:---------|----------:|-------:|-------:|-------:|
| ARP_Spoofing            | reference   | last          |   1 | auroc    |         5 | 0.9215 | 0.9196 | 0.923  |
| ARP_Spoofing            | reference   | mean          |   4 | auroc    |         5 | 0.9479 | 0.9461 | 0.9493 |
| ARP_Spoofing            | reference   | min           |   4 | auroc    |         5 | 0.9184 | 0.9169 | 0.9198 |
| ARP_Spoofing            | reference   | mean          |  16 | auroc    |         5 | 0.9652 | 0.9639 | 0.9664 |
| ARP_Spoofing            | reference   | min           |  16 | auroc    |         5 | 0.8725 | 0.8712 | 0.8741 |
| ARP_Spoofing            | ledoit_wolf | last          |   1 | auroc    |         5 | 0.9282 | 0.928  | 0.9283 |
| ARP_Spoofing            | ledoit_wolf | mean          |   4 | auroc    |         5 | 0.9554 | 0.9553 | 0.9555 |
| ARP_Spoofing            | ledoit_wolf | min           |   4 | auroc    |         5 | 0.9134 | 0.9133 | 0.9134 |
| ARP_Spoofing            | ledoit_wolf | mean          |  16 | auroc    |         5 | 0.9716 | 0.9716 | 0.9717 |
| ARP_Spoofing            | ledoit_wolf | min           |  16 | auroc    |         5 | 0.8539 | 0.8538 | 0.8541 |
| ARP_Spoofing            | mincovdet   | last          |   1 | auroc    |         5 | 0.9215 | 0.9188 | 0.9241 |
| ARP_Spoofing            | mincovdet   | mean          |   4 | auroc    |         5 | 0.9601 | 0.9566 | 0.9633 |
| ARP_Spoofing            | mincovdet   | min           |   4 | auroc    |         5 | 0.8636 | 0.8593 | 0.8681 |
| ARP_Spoofing            | mincovdet   | mean          |  16 | auroc    |         5 | 0.9779 | 0.974  | 0.9813 |
| ARP_Spoofing            | mincovdet   | min           |  16 | auroc    |         5 | 0.7496 | 0.7419 | 0.7593 |
| Recon-VulScan           | reference   | last          |   1 | auroc    |         5 | 0.7656 | 0.7642 | 0.7674 |
| Recon-VulScan           | reference   | mean          |   4 | auroc    |         5 | 0.7887 | 0.7868 | 0.7908 |
| Recon-VulScan           | reference   | min           |   4 | auroc    |         5 | 0.7619 | 0.7592 | 0.7649 |
| Recon-VulScan           | reference   | mean          |  16 | auroc    |         5 | 0.8096 | 0.8076 | 0.8121 |
| Recon-VulScan           | reference   | min           |  16 | auroc    |         5 | 0.7589 | 0.7547 | 0.7631 |
| Recon-VulScan           | ledoit_wolf | last          |   1 | auroc    |         5 | 0.7452 | 0.7452 | 0.7453 |
| Recon-VulScan           | ledoit_wolf | mean          |   4 | auroc    |         5 | 0.7718 | 0.7715 | 0.772  |
| Recon-VulScan           | ledoit_wolf | min           |   4 | auroc    |         5 | 0.7327 | 0.7323 | 0.7333 |
| Recon-VulScan           | ledoit_wolf | mean          |  16 | auroc    |         5 | 0.8018 | 0.8016 | 0.802  |
| Recon-VulScan           | ledoit_wolf | min           |  16 | auroc    |         5 | 0.7046 | 0.7031 | 0.7071 |
| Recon-VulScan           | mincovdet   | last          |   1 | auroc    |         5 | 0.6813 | 0.6612 | 0.6963 |
| Recon-VulScan           | mincovdet   | mean          |   4 | auroc    |         5 | 0.7059 | 0.6692 | 0.7326 |
| Recon-VulScan           | mincovdet   | min           |   4 | auroc    |         5 | 0.6064 | 0.59   | 0.6174 |
| Recon-VulScan           | mincovdet   | mean          |  16 | auroc    |         5 | 0.7185 | 0.6633 | 0.7555 |
| Recon-VulScan           | mincovdet   | min           |  16 | auroc    |         5 | 0.5328 | 0.5266 | 0.5406 |
| MQTT-Malformed_Data     | reference   | last          |   1 | auroc    |         5 | 0.8879 | 0.8857 | 0.8896 |
| MQTT-Malformed_Data     | reference   | mean          |   4 | auroc    |         5 | 0.9056 | 0.9027 | 0.9076 |
| MQTT-Malformed_Data     | reference   | min           |   4 | auroc    |         5 | 0.8906 | 0.8871 | 0.8938 |
| MQTT-Malformed_Data     | reference   | mean          |  16 | auroc    |         5 | 0.9231 | 0.9186 | 0.9261 |
| MQTT-Malformed_Data     | reference   | min           |  16 | auroc    |         5 | 0.8612 | 0.8566 | 0.8665 |
| MQTT-Malformed_Data     | ledoit_wolf | last          |   1 | auroc    |         5 | 0.8836 | 0.8836 | 0.8837 |
| MQTT-Malformed_Data     | ledoit_wolf | mean          |   4 | auroc    |         5 | 0.9049 | 0.9048 | 0.905  |
| MQTT-Malformed_Data     | ledoit_wolf | min           |   4 | auroc    |         5 | 0.8804 | 0.88   | 0.8808 |
| MQTT-Malformed_Data     | ledoit_wolf | mean          |  16 | auroc    |         5 | 0.9253 | 0.9252 | 0.9253 |
| MQTT-Malformed_Data     | ledoit_wolf | min           |  16 | auroc    |         5 | 0.8435 | 0.8428 | 0.8442 |
| MQTT-Malformed_Data     | mincovdet   | last          |   1 | auroc    |         5 | 0.7814 | 0.7709 | 0.7919 |
| MQTT-Malformed_Data     | mincovdet   | mean          |   4 | auroc    |         5 | 0.8154 | 0.7942 | 0.8383 |
| MQTT-Malformed_Data     | mincovdet   | min           |   4 | auroc    |         5 | 0.713  | 0.7042 | 0.7198 |
| MQTT-Malformed_Data     | mincovdet   | mean          |  16 | auroc    |         5 | 0.8433 | 0.8147 | 0.872  |
| MQTT-Malformed_Data     | mincovdet   | min           |  16 | auroc    |         5 | 0.6539 | 0.6432 | 0.6618 |
| TCP_IP-DoS-UDP          | reference   | last          |   1 | auroc    |         5 | 0.9513 | 0.9314 | 0.9712 |
| TCP_IP-DoS-UDP          | reference   | mean          |   4 | auroc    |         5 | 0.9463 | 0.9167 | 0.9754 |
| TCP_IP-DoS-UDP          | reference   | min           |   4 | auroc    |         5 | 0.9624 | 0.9383 | 0.9864 |
| TCP_IP-DoS-UDP          | reference   | mean          |  16 | auroc    |         5 | 0.9362 | 0.8918 | 0.9785 |
| TCP_IP-DoS-UDP          | reference   | min           |  16 | auroc    |         5 | 0.9385 | 0.9103 | 0.9683 |
| TCP_IP-DoS-UDP          | ledoit_wolf | last          |   1 | auroc    |         5 | 0.7794 | 0.7793 | 0.7794 |
| TCP_IP-DoS-UDP          | ledoit_wolf | mean          |   4 | auroc    |         5 | 0.7774 | 0.7773 | 0.7774 |
| TCP_IP-DoS-UDP          | ledoit_wolf | min           |   4 | auroc    |         5 | 0.8158 | 0.8158 | 0.8159 |
| TCP_IP-DoS-UDP          | ledoit_wolf | mean          |  16 | auroc    |         5 | 0.7887 | 0.7886 | 0.7888 |
| TCP_IP-DoS-UDP          | ledoit_wolf | min           |  16 | auroc    |         5 | 0.8374 | 0.8373 | 0.8374 |
| TCP_IP-DoS-UDP          | mincovdet   | last          |   1 | auroc    |         5 | 0.9796 | 0.9734 | 0.9854 |
| TCP_IP-DoS-UDP          | mincovdet   | mean          |   4 | auroc    |         5 | 0.9806 | 0.975  | 0.9858 |
| TCP_IP-DoS-UDP          | mincovdet   | min           |   4 | auroc    |         5 | 0.9874 | 0.9836 | 0.991  |
| TCP_IP-DoS-UDP          | mincovdet   | mean          |  16 | auroc    |         5 | 0.9844 | 0.9799 | 0.9884 |
| TCP_IP-DoS-UDP          | mincovdet   | min           |  16 | auroc    |         5 | 0.989  | 0.9859 | 0.9921 |
| MQTT-DDoS-Publish_Flood | reference   | last          |   1 | auroc    |         5 | 0.9361 | 0.936  | 0.9363 |
| MQTT-DDoS-Publish_Flood | reference   | mean          |   4 | auroc    |         5 | 0.9443 | 0.9441 | 0.9444 |
| MQTT-DDoS-Publish_Flood | reference   | min           |   4 | auroc    |         5 | 0.9724 | 0.9723 | 0.9726 |
| MQTT-DDoS-Publish_Flood | reference   | mean          |  16 | auroc    |         5 | 0.9507 | 0.9505 | 0.9509 |
| MQTT-DDoS-Publish_Flood | reference   | min           |  16 | auroc    |         5 | 0.9831 | 0.9829 | 0.9833 |
| MQTT-DDoS-Publish_Flood | ledoit_wolf | last          |   1 | auroc    |         5 | 0.939  | 0.9389 | 0.9391 |
| MQTT-DDoS-Publish_Flood | ledoit_wolf | mean          |   4 | auroc    |         5 | 0.9503 | 0.9501 | 0.9504 |
| MQTT-DDoS-Publish_Flood | ledoit_wolf | min           |   4 | auroc    |         5 | 0.9736 | 0.9736 | 0.9737 |
| MQTT-DDoS-Publish_Flood | ledoit_wolf | mean          |  16 | auroc    |         5 | 0.9597 | 0.9596 | 0.9598 |
| MQTT-DDoS-Publish_Flood | ledoit_wolf | min           |  16 | auroc    |         5 | 0.9804 | 0.9804 | 0.9804 |
| MQTT-DDoS-Publish_Flood | mincovdet   | last          |   1 | auroc    |         5 | 0.9582 | 0.9416 | 0.9701 |
| MQTT-DDoS-Publish_Flood | mincovdet   | mean          |   4 | auroc    |         5 | 0.9673 | 0.9496 | 0.977  |
| MQTT-DDoS-Publish_Flood | mincovdet   | min           |   4 | auroc    |         5 | 0.9804 | 0.9735 | 0.9852 |
| MQTT-DDoS-Publish_Flood | mincovdet   | mean          |  16 | auroc    |         5 | 0.9693 | 0.9523 | 0.9784 |
| MQTT-DDoS-Publish_Flood | mincovdet   | min           |  16 | auroc    |         5 | 0.9822 | 0.9762 | 0.9866 |
