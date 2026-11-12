# Week 1 — Sensor note (C-MAPSS FD001)

**Result:** 13 of 21 sensors track HPC degradation; 7 are flat and 1 is weak. The strongest signals are HPC outlet static pressure (Ps30) and LPT outlet temperature (T50).

## Data at a glance
- 100 training engines, run to failure; 100 test engines, cut off early
- Engine life: 128–362 cycles, median 199
- One operating condition, one fault mode (HPC degradation)

## Dropped
| Reason | Sensors |
|---|---|
| Flat (no information) | T2, P2, P15, epr, farB, Nf_dmd, PCNfR_dmd |
| Weak (\|rho\| < 0.3) | NRc |
| Operating settings | setting 1–2 are noise, setting 3 is constant |

## Kept (Spearman rho with RUL)
| Sensor | Meaning | rho |
|---|---|---|
| Ps30 | HPC outlet static pressure | -0.72 |
| T50 | LPT outlet temperature | -0.70 |
| phi | Fuel flow / Ps30 | +0.69 |
| P30 | HPC outlet total pressure | +0.68 |
| BPR | Bypass ratio | -0.67 |
| W32 | LPT coolant bleed | +0.66 |
| W31 | HPT coolant bleed | +0.65 |
| htBleed | Bleed enthalpy | -0.63 |
| T24 | LPC outlet temperature | -0.63 |
| T30 | HPC outlet temperature | -0.61 |
| Nf | Fan speed | -0.57 |
| NRf | Corrected fan speed | -0.57 |
| Nc | Core speed | -0.32 |

Negative rho = the sensor rises as the engine nears failure (RUL falls).

## What the physics says
As the HPC loses efficiency, it needs more work for the same pressure rise. So T30 rises, the turbines work harder and T50 rises, and Ps30 and bypass ratio climb. P30, phi (fuel flow / Ps30) and the coolant bleeds (W31, W32) fall. The data shows exactly this pattern, which is what the Week 2 cycle model must reproduce.

## Observations for modelling
- Sensors stay nearly flat early in life and drift in the last ~100–150 cycles. Cap the RUL target at about 125 cycles (piecewise-linear target).
- Nc and NRc move differently from engine to engine: some rise, some fall. Treat them as engine-specific, not a clean health signal.
- Sensor noise is large next to the trend. Smooth before use (10-cycle rolling mean works).
