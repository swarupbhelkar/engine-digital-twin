# Aero Engine Digital Twin (HUMS)

A digital twin of a two-spool turbofan that tracks component health and predicts remaining useful life (RUL). It combines a thermodynamic cycle model, sensor data, and a CAD/CFD layer.

**Status:** Weeks 1-5 done: data exploration, 0D physics model, health index and RUL models, twin logic (sensor check, fault injection, per-cycle twin state), CAD cutaway coloured by health and a Fluent blade-cascade CFD case.

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
    src/             data_loader.py, features.py, rul_rf.py, rul_lstm.py,
                     sensor_check.py, fault_injection.py, twin_state.py
    src/physics/     0D cycle model and HPC health fit
    src/cad_cfd/     HPC stage design, blade, engine layout, cascade mesh, CFD results
    cad/             engine layout CSVs, Fusion 360 cutaway + colour scripts
    cfd/             cascade conditions, meshes, Fluent settings and results
    tests/           pytest checks
    reports/         notes, results CSVs, figures
    app/             Week 6

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
    python -m src.sensor_check         # Week 4: physics vs data sensor check
    python -m src.fault_injection      # injected sensor faults, about 1 minute
    python -m src.twin_state           # per-cycle twin state for the dashboard
    python -m src.cad_cfd.hpc_stage    # Week 5: HPC stage 5 mean-line design
    python -m src.cad_cfd.blade_geometry   # blade shapes + CFD inputs
    python -m src.cad_cfd.engine_layout    # cutaway sizes + health colours
    python -m src.cad_cfd.cascade_mesh     # Fluent meshes (needs gmsh)
    python -m src.cad_cfd.cascade_results  # loss and efficiency link from Fluent results

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

## Week 4 findings: twin logic
- Each sensor is checked against the physics prediction from the other 6. A learned model-mismatch correction stops false alarms near end of life (52 of 100 clean engines without it, 0 with it).
- Injected faults of 4 x sensor noise: the right sensor is named in 100 % of engines for 6 of 7 sensors (bias), within about 8 cycles. T24 is the weak spot (71 %).
- Trusting a faulty sensor shifts predicted RUL by 10-30 cycles; with the check, 5-7.
- Per-cycle twin state: health, wear rate, RUL with an 80 % range (76 % coverage on test), OK / WATCH / ACTION. No engine within 30 cycles of failure is marked OK.
- See reports/week4_twin_note.md.

## Week 5 findings: CAD and CFD
- Mid-stage HPC blade sized from the Week 2 design point and FD001 core speed: inlet Mach 0.75, chord 38 mm, 90 blades.
- Before CFD, the physics predicted that FD001's -1.49 % efficiency needs each blade row's loss to rise by about 0.008.
- Fluent 2D cascade at fixed mass flow: healthy loss 0.018; fouled (40 um roughness) 0.048, worth -5.5 % HPC efficiency, 3.7 x the FD001 failure level; leading-edge erosion (2 % chord) 0.020, only -0.4 %.
- Fouling, which lowers both efficiency and flow capacity as Week 2 found, is the mechanism that fits FD001.
- Fusion 360 cutaway with the HPC coloured by twin health for demo engine 52.
- See reports/week5_cad_cfd_note.md.

## Roadmap
1. ~~Data & exploration~~
2. ~~0D cycle model (physics)~~
3. ~~Health index + RUL models (random forest, LSTM)~~
4. ~~Twin logic: physics vs data, anomalies, efficiency loss~~
5. ~~CAD cutaway + CFD blade case~~
6. Streamlit dashboard + report
