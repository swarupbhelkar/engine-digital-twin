# Week 4: twin logic, physics vs data (C-MAPSS FD001)

**Result:** the twin checks every sensor against the physics model, catches injected sensor faults, keeps them out of the health estimate, and reports a per-cycle state (health, wear rate, RUL with a range, OK / WATCH / ACTION).

## 1. Sensor check (src/sensor_check.py)
- The HPC has 2 health numbers but 7 sensors are fitted, so the sensors are redundant. For each sensor, health is fitted from the other 6 and the physics predicts the 7th. Residual = measured - predicted.
- Leave-one-out matters: T24 is so quiet that an all-7 fit follows it almost exactly, so a bad T24 would hide inside the health numbers.
- Model mismatch grows with wear (0D model, Week 2): near failure T24 reads -0.17 % and T50 +0.32 % against the prediction. Learned from train engines as a curve in damage and subtracted. Without it, 52 of 100 clean train engines raise a false alarm near end of life.
- Alarm: the same sensor has the largest |z| above 4 for 5 cycles in a row. False alarms: 0 of 100 train engines, 1 of 100 test engines (engine 34, T24 drifting slowly in the last ~30 cycles, peak z about 4.3).
- After an alarm, health comes from the other 6 sensors.

## 2. Fault injection (src/fault_injection.py)
77 test engines (80+ cycles), one fault per engine, random start cycle. Sizes are multiples of each sensor's noise.

| Fault | Right sensor named, 6 of 7 sensors | T24 | Median delay |
|---|---|---|---|
| Bias 3 x noise | 99-100 % | 31 % | 9-11 cycles |
| Bias 4 x noise | 100 % | 71 % | 8-9 cycles |
| Drift 4 x noise (after 50 cycles) | 91-99 % | 49 % | 29-36 cycles |

- No alarm before any fault started.
- Effect on remaining life (physics-only random forest): trusting the faulty sensor shifts predicted RUL by 10-30 cycles on average (worst: P30 drift, 36 cycles). With the check: 5-7 cycles. About 4 cycles of that remains even when the fault is caught; it is the cost of running on 6 sensors.
- Weak spot: T24. 4 x noise is only about 1.2 degR, and the other 6 sensors cannot predict it that precisely. A missed T24 fault still shifts RUL by 13-16 cycles. A real HUMS would use a second T24 probe.

## 3. Twin state (src/twin_state.py)
One row per engine per cycle: HPC efficiency and flow loss, health index, wear rate (health lost per 100 cycles, last 30 cycles), predicted RUL with an 80 % range, sensor status, engine status. Written to data/processed/twin_state_test.csv for the dashboard.
- RUL model: random forest on the trusted physics health only (Week 3 model C). Test RMSE 17.4, NASA score 563. The LSTM is more accurate (14.1) but reads raw sensors, so it has no protection from sensor faults.
- 80 % range from 5-fold cross-validation errors on train, in 3 bands of predicted RUL. True RUL inside: 76 % of test engines. (The spread between trees was too narrow: 47 %.)
- Straight-line extrapolation of health to failure was tried and dropped (RMSE 29-36): HPC wear accelerates near the end.

Engine status at the last test cycle (ACTION <= 30 predicted cycles, WATCH <= 75):

| True remaining life | OK | WATCH | ACTION |
|---|---|---|---|
| <= 30 cycles | 0 | 8 | 17 |
| 31-75 | 3 | 14 | 1 |
| > 75 | 52 | 5 | 0 |

No engine within 30 cycles of failure is marked OK.

**Demo (figure 12):** test engine 52 (chosen because the clean twin is accurate there), T30 drifting low from cycle 90, which hides wear. Alarm at cycle 122. At the last cycle, true RUL 29: trusting T30 predicts 40 (WATCH, late); the twin predicts 30 (ACTION).

## Limitations
- C-MAPSS has no real sensor faults; all faults are injected and one at a time.
- Fault sizes are relative to sensor noise, not to real sensor accuracy specs.
- Status thresholds (30 / 75 cycles) are illustrative; an operator would set them.
- The 80 % range is empirical, slightly under target (76 %) on test.

## Figures
- reports/figures/10_sensor_residual_mismatch.png: residuals before and after the mismatch correction
- reports/figures/11_fault_injection.png: detection vs fault size, effect on RUL
- reports/figures/12_twin_demo.png: twin catching a T30 drift on engine 52
