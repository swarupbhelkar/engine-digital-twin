# Week 2: 0D cycle model and HPC health (C-MAPSS FD001)

**Result:** a physics model of the engine explains the FD001 sensor drift as combined HPC efficiency and flow-capacity loss, about 1.5 % each at failure. The resulting health estimate tracks remaining life better than any single sensor.

## Model
- Two-spool turbofan: fan, LPC, HPC, combustor, HPT, LPT, core and bypass nozzles. HPT and LPT coolant flows (W31, W32) taken from HPC exit and returned at each turbine inlet.
- Gas properties vary with temperature and fuel-air ratio (Walsh & Fletcher polynomials). Constant gamma moved HPC efficiency by up to 6 points, so this matters.
- Units as in C-MAPSS: degR, psia, lbm/s.

## Design point (new engine = fleet mean of first 20 cycles)
| Quantity | Value | Source |
|---|---|---|
| HPC efficiency | 0.852 | from T24, T30, P30 |
| LPC / HPC pressure ratio | 1.95 / 19.4 | sensors + assumed LPC efficiency 0.88 |
| Core flow / total airflow | 342 / 3,224 lbm/s | solved |
| T4 | 3024 degR (1680 K) | solved |
| HPT / LPT pressure ratio | 5.70 / 5.03 | derived |
| Coolant | 18.2 % of core flow | W31 + W32 |
| Gross thrust | about 88,500 lbf | indicative |

Matches T24, T30, T50, P30, P15 and fuel flow to 0.000 %.

## Off-design rules
Fan speed held (fixed total airflow and fan PR), fixed HPC corrected flow (scaled by flow-capacity health), choked HPT and LPT nozzles, fixed core nozzle area, power balance on each spool. Solver unknowns: core flow, LPC PR, HPC PR, T4.

## Sensor response to 1 % damage (% change)
| Sensor | Efficiency -1 % | Flow -1 % | Both | FD001 new to worn |
|---|---|---|---|---|
| T24 | -0.46 | +0.58 | +0.11 | +0.17 |
| T30 | +0.34 | +0.15 | +0.49 | +0.83 |
| T50 | +1.06 | -0.16 | +0.90 | +1.58 |
| P30 | -0.77 | +0.48 | -0.30 | -0.38 |
| BPR | +1.38 | -0.60 | +0.79 | +1.05 |
| W31, W32 | -1.22 | +0.54 | -0.70 | -1.07 |
| Fuel | +0.44 | +0.13 | +0.56 | +1.08 |

Efficiency loss alone gets T24 wrong. Both losses together get every direction right.

## Health fit (gas path analysis)
- Influence matrix from the model; weighted least squares (weights = 1 / sensor noise) on 7 sensors, each engine against its own first 20 cycles, 10-cycle smoothing.
- At failure: efficiency -1.49 % (std 0.27, range -2.05 to -1.08), flow capacity -1.53 % (std 0.27).
- Nonlinear model at fleet-mean damage reproduces all 7 sensors in direction; fuel, not used in the fit, is predicted +0.85 % vs +1.09 % measured.
- Spearman |rho| with RUL (last 150 cycles): efficiency estimate 0.90, flow estimate 0.90, Ps30 0.78.

## Limitations
- Fan, LPC, HPT, LPT efficiencies are textbook values; FD001 cannot pin them down separately.
- HPC modelled as fixed corrected flow, not a full map. Fine for 1-2 % changes.
- Fuel flow = phi x Ps30 read as lbm/h is an assumption. Core flow and thrust scale with it; temperatures and pressures do not.
- T50 response under-predicted by about 16 %, T30 by about 12 %.
- Efficiency and flow losses are 99 % correlated across engines, so FD001 cannot tell them apart in time, only in total. Supports a single combined health index in Week 3.

## Files
`src/physics/*.py`, `tests/`, `reports/design_point.json`, `reports/hpc_sensitivity.csv`, `reports/hpc_health_fd001.csv`, `reports/hpc_loss_per_engine_fd001.csv`, `reports/figures/05_*.png`, `reports/figures/06_*.png`
