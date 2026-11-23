# T5: calibration and risk-coverage

**Status:** Exploratory, post hoc analysis; no change to H1-H5, their decision rules or verdicts. Test windows used for reporting only.

- Command: `python scripts/p4_rev_5_calibration_risk_coverage.py`
- Script: `p4_rev_5_calibration_risk_coverage.py`, SHA-256 `5fa676ff02a26022dda43df6c2e0574f7c6881fe21f038dc0fe9d232784eeab0`
- Inputs: score_export/*.npz; zero_day/<run>/predictions.npz
- Canonical runs read: 75
- rng seed: 42
- Versions: Python 3.12.4, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1
- Start 2026-11-20 11:33:58, end 2026-11-20 11:34:55, duration 57 s

## Result
Pass criterion: risk at coverage 1.0 equals (held-out + misclassified known) / total computed independently (max |difference| 0.0e+00); label order of predictions.npz equals score_export y19 for all runs.
ECE uses 15 equal-width bins (fixed now) on known test windows; AURC treats a held-out window and a misclassified known window as errors; lower is better.

Mean over classes and seeds:
```
             ece_19class_known  ece_binary_known  aurc_energy  aurc_msp
model                                                                  
bilstm_dnn              0.0160            0.0022       0.0259    0.0245
trace_moe               0.0079            0.0012       0.0336    0.0392
transformer             0.0071            0.0017       0.0293    0.0251
```

Figures: figures/reliability_19class.pdf, figures/risk_coverage.pdf.
