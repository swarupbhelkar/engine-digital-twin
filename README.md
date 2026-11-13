# Aero Engine Digital Twin (HUMS)

A digital twin of a two-spool turbofan that tracks component health and predicts remaining useful life (RUL). It combines a thermodynamic cycle model, sensor data, and a CAD/CFD layer.

**Status:** Weeks 1-3 done: data exploration, 0D physics model, health index and RUL models.

## Headline result (FD001 test set, 100 engines, RUL capped at 125)
| Model | Test RMSE (cycles) | NASA score |
|---|---|---|
| Random forest, sensors only | 19.6 | 1563 |
| Random forest, sensors + physics health | 16.7 | 467 |
| LSTM, sensors only | 14.1 | 410 |
| LSTM, sensors + physics health | 14.5 | 321 |

Physics health features (HPC efficiency and flow-capacity loss from gas path analysis) cut random-forest error by 15 % and its NASA score by 70 %. With an LSTM, RMSE is tied within seed noise, but late predictions (the dangerous kind) drop: NASA score 410 to 321.

## Structure
    data/raw/        NASA C-MAPSS files (FD001)
    data/processed/  feature tables (generated, not in git)
    notebooks/       01_eda.ipynb: exploration
    src/             data_loader.py, features.py, rul_rf.py, rul_lstm.py
    src/physics/     0D cycle model and HPC health fit
    tests/           pytest checks
    reports/         notes, results CSVs, figures
    cad/ cfd/ app/   Weeks 5-6

## Setup (Windows)
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python -m pytest -q

## Reproduce the results
    python -m src.physics.health_fit   # Week 2: HPC health per engine
    python -m src.features             # Week 3: feature table, train + test
    python -m src.rul_rf               # random forest, about 1 minute
    python -m src.rul_lstm             # LSTM, a few minutes on CPU

## Data
NASA C-MAPSS turbofan engine degradation dataset (Saxena et al., 2008), from the NASA Prognostics Data Repository. FD001: 100 training engines run to failure, 100 test engines, one operating condition, HPC degradation fault.

## Week 1 findings
- Engine life ranges 128-362 cycles (median 199).
- 7 sensors are flat and dropped; the rest trend with degradation. See reports/week1_sensor_note.md.
- Strongest single signals: Ps30 and T50 (|rho| about 0.7).

## Week 2 findings: 0D cycle model
- Calibrated design point matches T24, T30, T50, P30, P15 and fuel flow exactly. HPC efficiency 0.852, T4 about 1680 K.
- Efficiency loss alone gets T24 wrong; efficiency plus flow-capacity loss explains all sensor directions.
- At failure: HPC efficiency -1.49 %, flow capacity -1.53 %.
- Physics health estimate tracks RUL better than any sensor (|rho| 0.90 vs 0.78 for Ps30). See reports/week2_physics_note.md.

## Week 3 findings: health index and RUL
- Physics health index (1 = new, 0 = typical failure damage) works on unseen test engines: Spearman rho 0.86 with true RUL.
- Three physics numbers beat all 14 raw sensors in a random forest (RMSE 17.4 vs 19.6), because each engine is compared with its own healthy start, removing build-to-build scatter.
- An LSTM reading 30-cycle windows learns the wear trend from sensors alone, so physics no longer lowers its RMSE, but it still makes errors safer.
- See reports/week3_rul_note.md.

## Roadmap
1. ~~Data & exploration~~
2. ~~0D cycle model (physics)~~
3. ~~Health index + RUL models (random forest, LSTM)~~
4. Twin logic: physics vs data, anomalies, efficiency loss
5. CAD cutaway + CFD blade case
6. Streamlit dashboard + report
