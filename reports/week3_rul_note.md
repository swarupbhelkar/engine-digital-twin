# Week 3: health index and RUL models (C-MAPSS FD001)

**Result:** physics health features from the Week 2 gas path analysis make a simple random forest 15 % more accurate and much safer. An LSTM matches that accuracy from sensors alone, but physics still cuts its late predictions.

## Health index
- Week 2 HPC health fit applied to the test engines too. Sensor noise weights and the failure-level damage come from train engines only.
- hi = 1 - average fraction of failure-level loss used (efficiency -1.49 %, flow -1.53 %). 1 = new, 0 = typical failure.
- Test engines, last cycle: Spearman rho(hi, true RUL) = 0.86. Mean hi 0.17 with under 25 cycles left, 0.90 with 100-150 left.

## Models
- Target: RUL capped at 125. Score: RMSE and NASA (PHM08) score at each test engine's last cycle.
- Random forest: 200 trees, min 10 cycles per leaf. 5-fold CV on train, split by engine.
- LSTM: 30-cycle window, 64 hidden units. 15 train engines held out for early stopping. 3 seeds averaged.

## Results (test, 100 engines)
| Model | Inputs | Test RMSE | NASA score |
|---|---|---|---|
| Random forest | 14 sensors | 19.6 | 1563 |
| Random forest | 14 sensors + 3 physics | 16.7 | 467 |
| Random forest | 3 physics only | 17.4 | 559 |
| LSTM | 14 sensors | 14.1 | 410 |
| LSTM | 14 sensors + 3 physics | 14.5 | 321 |

Single-seed LSTM spread: about +/- 0.6 (sensors) and +/- 0.2 (physics) RMSE.

## Engine interpretation
- Raw sensors mix wear with build-to-build differences. The gas path analysis compares each engine with its own healthy start and turns 7 correlated sensors into the two numbers that describe a worn HPC. A random forest, which sees one cycle at a time, benefits most: physics features take 77 % of its importance, HPC efficiency loss alone 43 %.
- Near failure (true RUL under 50), random-forest mean error falls from 10.9 to 6.1 cycles with physics.
- An LSTM watches 30 cycles of sensor drift and learns the wear trend itself, so physics adds no RMSE gain. It does reduce late predictions (NASA score 410 to 321), and it says which component is wearing, which the LSTM cannot.

## Limitations
- Early in life (true RUL over 100), physics features are slightly worse than raw sensors (random-forest error 14.5 vs 11.0): little damage yet, so the health signal is mostly noise.
- Health is measured against each engine's first 20 cycles, assuming it starts healthy. Short test engines (from 31 cycles) have a short baseline.
- 100 test engines: differences below about 1 RMSE are within noise. LSTM results vary slightly between machines and with the choice of validation engines.
- FD001 only: one operating condition, one fault mode.

## Files
src/features.py, src/rul_rf.py, src/rul_lstm.py, tests/test_features.py, tests/test_rul_rf.py, tests/test_rul_lstm.py, reports/rul_rf_results.csv, reports/rul_lstm_results.csv, reports/figures/07_health_index_train_test.png, reports/figures/08_rf_rul_test.png, reports/figures/09_lstm_rul_test.png
