# Week 5: CAD cutaway and CFD blade case (C-MAPSS FD001)

**Result:** a mid-stage HPC blade sized from this engine's own flows, run in Fluent healthy, fouled and eroded. Fouling raises the blade loss enough to explain the FD001 efficiency loss many times over; leading-edge erosion of the size tested does not. The CAD cutaway shows the HPC coloured by the twin's health index.

## 1. HPC stage design (src/cad_cfd/hpc_stage.py)
- HPC split into 10 equal-work stages with one polytropic efficiency (0.898) that reproduces the Week 2 pressure ratio (19.4). Stage 5 modelled.
- Velocity triangles from design choices: flow coefficient 0.55, work coefficient 0.35, reaction 0.5. Core speed Nc from FD001 sets the radius.
- Stage 5: PR 1.336, efficiency 0.894, blade speed 398 m/s, rotor flow turned 50.8 -> 30.6 deg, inlet relative Mach 0.75, de Haller 0.73, mean radius 419 mm, chord 38.2 mm, 90 blades, Reynolds 2.2 million.
- Prediction made before any CFD: to lose the Week 2 figure of 1.49 % HPC efficiency, each rotor and stator row needs its loss coefficient to rise by about 0.0082.

## 2. Blade and engine layout (blade_geometry.py, engine_layout.py)
- Blade: NACA 4-digit thickness (7 % chord) on a circular-arc camber line; Carter's rule for deviation (6.5 deg). Generic section, not a real HPC blade.
- Eroded blade: front 2 % of chord removed, blunt round nose.
- Layout: each station's flow area from the Week 2 mass flow, temperature and pressure. Fan tip 1.59 m, engine about 4.4 m long, HPC blades 178 mm tall at entry and 20 mm at exit.

## 3. CAD cutaway (cad/fusion_cutaway, cad/fusion_renders)
- Fusion 360 script builds 11 parts revolved 270 deg (open quarter facing the viewer). A second script gives each part a metal finish and paints the HPC by health.
- Demo engine 52: new (health 1.00, green), mid-wear (0.69, WATCH, light green), last cycle (0.38, ACTION, yellow). Figure 15. Casing and nacelle hidden so the HPC shows.
- The colour follows health; the status follows RUL. At the last cycle the HPC is yellow but the status is ACTION, because wear speeds up near the end.

## 4. Cascade CFD (src/cad_cfd/cascade_mesh.py, Fluent 2026 R1 Student)
- 2D periodic cascade, one blade passage. Mesh made in Gmsh and written in Fluent's mesh format: about 22,400 cells, 30 quad layers on the blade (first cell 1 um, y+ about 1-2), same settings for both blade shapes. Figure 16.
- Physics: steady, pressure-based coupled solver, ideal-gas air (cp 1050, Sutherland viscosity), k-omega SST. Inlet: relative total pressure 10.57 bar, 600 K, flow angle 50.8 deg. Outlet: target mass flow 30.15 kg/s per metre span, so all cases run at the same flow and inlet Mach (0.75).
- Converged to residuals below 1e-5. Results: cfd/results/cascade_fluent.csv, summary from src/cad_cfd/cascade_results.py, figure 17.

| Case | Loss coefficient | Rise | HPC efficiency if every row wore alike | vs FD001 failure (-1.49 %) |
|---|---|---|---|---|
| Healthy | 0.0182 | - | - | - |
| Fouled (40 um sand-grain roughness) | 0.0482 | +0.030 (+165 %) | -5.5 % | 3.7 x |
| Eroded (2 % chord, blunt nose) | 0.0203 | +0.002 (+11 %) | -0.4 % | 0.25 x |

## 5. What it means for the twin
- Fouling is the stronger loss mechanism: roughness thickens the boundary layer over the whole blade. The FD001 failure level corresponds to much lighter fouling than 40 um (rough estimate 10-15 um; one more run would pin it).
- Erosion of this depth hardly changes loss at the design point. Its known penalties (lower stall margin, more loss off-design) are not tested here.
- Week 2 found efficiency AND flow capacity falling together. Fouling does both; this supports reading FD001's HPC fault as fouling-like wear.
- Same order of magnitude between a 0D cycle model fitted to sensors and a CFD blade sized from the same engine: the twin's physics layers agree.

## Tried and dropped
- First mesh read had every face oriented the wrong way (Fluent's left/right cell rule); fixed in the writer.
- First CFD run left density constant (incompressible): mass flow 17 instead of 30. Rerun with ideal gas.
- Fixed outlet pressure gave inlet Mach 0.87 and a slow oscillation in loss (0.019-0.059). Target mass flow at the outlet fixed both.

## Limitations
- 2D blade section: no tip clearance, no endwall or secondary losses. Real stage losses are larger.
- One stage stands in for all 10; "every row wore alike" is an assumption.
- Stage loading, blade shape, roughness height and erosion depth are design choices, not measured values. The HPC comes out long (1.3 m) for its pressure ratio.
- Fluent's standard roughness model is calibrated on uniform sand grains; real deposits are patchy.
- Cutaway images are viewport captures, not photo-real renders.

## Reproduce
    python -m src.cad_cfd.hpc_stage
    python -m src.cad_cfd.blade_geometry
    python -m src.twin_state
    python -m src.cad_cfd.engine_layout
    python -m src.cad_cfd.cascade_mesh      # needs: pip install gmsh
    python -m src.cad_cfd.cascade_results
Fluent: read cfd/mesh/<case>.msh, /mesh/modify-zones/make-periodic (periodic-upper, periodic-lower, no, yes, yes), /file/read-settings cfd/fluent/setup.bc, hybrid initialise, run to 1e-5. Fouled: blade roughness height 4e-05 m, constant 0.5.
