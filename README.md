# Aero Engine Digital Twin (HUMS)

A digital twin of a two-spool turbofan that tracks component health and predicts remaining useful life (RUL). It combines a thermodynamic cycle model, sensor data, and a CAD/CFD layer.

**Status:** Week 1 done — data exploration and sensor selection.

## Structure
```
data/raw/        NASA C-MAPSS files (FD001)
notebooks/       01_eda.ipynb — exploration
src/             data_loader.py; physics/ and health/ come next
reports/         figures, sensor note, selected sensors
cad/  cfd/  app/ Weeks 5–6
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

## Data
NASA C-MAPSS turbofan engine degradation dataset (Saxena et al., 2008), from the NASA Prognostics Data Repository. FD001: 100 training engines run to failure, 100 test engines, one operating condition, HPC degradation fault.

## Week 1 findings
- Engine life ranges 128–362 cycles (median 199).
- 7 sensors are flat and dropped; 13 trend with degradation. See `reports/week1_sensor_note.md`.
- Strongest signals: Ps30 and T50 (|rho| ≈ 0.7).

## Roadmap
1. ~~Data & exploration~~
2. 0D cycle model (physics)
3. Health index + RUL models (random forest, LSTM)
4. Twin logic: physics vs data → anomalies, efficiency loss
5. CAD cutaway + CFD blade case
6. Streamlit dashboard + report
