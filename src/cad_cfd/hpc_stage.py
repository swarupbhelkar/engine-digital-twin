"""Week 5, step 1a: mean-line design of one HPC stage (the CFD blade comes from here).

Why: the CFD case needs a blade that belongs to THIS engine, at THIS engine's
flow conditions, not a textbook airfoil at made-up numbers.

Inputs from earlier weeks:
  - Week 2 design point: HPC inlet/exit temperature and pressure, core flow
  - FD001 sensor Nc (s9): physical core speed, which sets the blade radius

Method (standard compressor mean-line design, e.g. Saravanamuttoo, ch. 5):
  1. Split the HPC into N_STAGES stages with equal work (equal enthalpy rise).
  2. One polytropic efficiency for all stages, chosen so the stages multiply
     out to the measured HPC pressure ratio.
  3. Velocity triangles for one stage from three design choices:
       flow coefficient  phi = Cx / U      (axial speed / blade speed)
       work coefficient  psi = dh / U^2    (sets the blade speed U)
       reaction          0.5               (rotor and stator share the diffusion)
  4. Checks: de Haller W2/W1 >= 0.72 (not too much diffusion in one row).
     Lieblein diffusion factor DF sets the solidity (chord / pitch).

These are DESIGN CHOICES, not data. A real HPC is not public. The numbers are
typical of a modern high-bypass HPC (about 10 stages, PR about 20).

Run from the repo root:
    python -m src.cad_cfd.hpc_stage
"""
import math
from dataclasses import dataclass

from src.data_loader import load_train
from src.physics import gas
from src.physics.design_point import run_engine
from src.physics.off_design import load_design

# --- Design choices (change here, then rerun) -------------------------------
N_STAGES = 10        # HPC stages
STAGE = 5            # the stage we model: mid-compressor
PHI = 0.55           # flow coefficient Cx/U
PSI = 0.35           # work coefficient dh/U^2
REACTION = 0.5
DIFFUSION = 0.42     # Lieblein diffusion factor target (0.45 is the usual limit)
ASPECT = 1.5         # blade height / chord

# --- Unit conversions and air properties (SI) -------------------------------
R_SI = 287.05                 # J/(kg K)
BTU_LBM_TO_J_KG = 2326.0
PSI_TO_PA = 6894.76
LBM_TO_KG = 0.453592


def sutherland_mu(t_k: float) -> float:
    """Dynamic viscosity of air, Pa s (Sutherland's law)."""
    return 1.716e-5 * (t_k / 273.15) ** 1.5 * (273.15 + 110.4) / (t_k + 110.4)


@dataclass
class Stage:
    """Everything the blade and the CFD case need, in SI units, angles in degrees."""
    number: int
    t0_in: float        # stage inlet total temperature, K
    p0_in: float        # stage inlet total pressure, Pa
    pr: float           # stage total pressure ratio
    eta: float          # stage isentropic efficiency
    eta_poly: float     # polytropic efficiency (same for all stages)
    dh: float           # stage enthalpy rise, J/kg
    u: float            # mean blade speed, m/s
    cx: float           # axial velocity, m/s
    beta1: float        # rotor inlet relative flow angle (from axial)
    beta2: float        # rotor exit relative flow angle
    alpha1: float       # rotor inlet absolute flow angle
    t1: float           # rotor inlet static temperature, K
    p1: float           # rotor inlet static pressure, Pa
    rho1: float
    w1: float           # inlet relative velocity, m/s
    w2: float           # exit relative velocity, m/s
    mach_w1: float      # inlet relative Mach number
    t0_rel: float       # relative total temperature (rotor frame), K
    p0_rel: float       # relative total pressure at rotor inlet, Pa
    gamma: float
    cp: float           # J/(kg K)
    r_mean: float       # m
    height: float       # blade height, m
    chord: float        # m
    solidity: float     # chord / pitch
    pitch: float        # m
    blades: int
    reynolds: float     # based on chord and W1
    de_haller: float    # W2 / W1


def design_stage(stage: int = STAGE) -> Stage:
    d = load_design()
    st = run_engine(d)["stations"]
    t24, t30, p24, p30 = st["24"].t, st["30"].t, st["24"].p, st["30"].p

    # 1) Equal work per stage
    h24, h30 = gas.enthalpy(t24), gas.enthalpy(t30)
    dh_btu = (h30 - h24) / N_STAGES

    # 2) Polytropic efficiency: R ln(PR) = eta_p * (phi(T30) - phi(T24)),
    #    where phi = integral of cp dT / T (variable cp, same as Week 2)
    eta_p = gas.R_GAS * math.log(p30 / p24) / (gas.entropy_fn(t30) - gas.entropy_fn(t24))

    def total_state(k):
        """Total T (degR) and P (psia) after k stages."""
        t = float(gas.temp_from_enthalpy(h24 + k * dh_btu))
        p = p24 * math.exp(eta_p * (gas.entropy_fn(t) - gas.entropy_fn(t24)) / gas.R_GAS)
        return t, float(p)

    t_in, p_in = total_state(stage - 1)
    t_out, p_out = total_state(stage)
    pr = p_out / p_in
    eta = float(gas.compressor_efficiency(t_in, t_out, pr))

    # 3) Velocity triangles (repeating stage, C3 = C1)
    dh = dh_btu * BTU_LBM_TO_J_KG
    u = math.sqrt(dh / PSI)
    cx = PHI * u
    tan_b1 = (REACTION + PSI / 2) / PHI
    tan_b2 = (REACTION - PSI / 2) / PHI
    tan_a1 = 1 / PHI - tan_b1
    b1, b2, a1 = (math.degrees(math.atan(x)) for x in (tan_b1, tan_b2, tan_a1))

    # Gas properties at stage inlet, then static conditions at rotor inlet
    t0 = t_in / 1.8
    p0 = p_in * PSI_TO_PA
    cp = float(gas.cp(t_in)) / gas.KJKGK_TO_BTULBMR * 1000.0
    gam = float(gas.gamma(t_in))
    c1 = cx / math.cos(math.radians(a1))
    t1 = t0 - c1 ** 2 / (2 * cp)
    p1 = p0 * (t1 / t0) ** (gam / (gam - 1))
    rho1 = p1 / (R_SI * t1)
    w1 = cx / math.cos(math.radians(b1))
    w2 = cx / math.cos(math.radians(b2))
    t0_rel = t1 + w1 ** 2 / (2 * cp)
    p0_rel = p1 * (t0_rel / t1) ** (gam / (gam - 1))

    # 4) Size: radius from core speed, height from mass flow, chord from aspect ratio
    nc_rpm = load_train()["s9"].mean()
    r_mean = u / (nc_rpm * 2 * math.pi / 60)
    area = d.w_core * LBM_TO_KG / (rho1 * cx)
    height = area / (2 * math.pi * r_mean)
    chord = height / ASPECT

    # Lieblein: DF = 1 - W2/W1 + (W1t - W2t) / (2 sigma W1)  ->  solve for sigma
    dwt = cx * (tan_b1 - tan_b2)
    solidity = dwt / (2 * w1 * (DIFFUSION - 1 + w2 / w1))
    blades = round(2 * math.pi * r_mean * solidity / chord)
    pitch = 2 * math.pi * r_mean / blades          # whole number of blades
    solidity = chord / pitch

    return Stage(number=stage, t0_in=t0, p0_in=p0, pr=pr, eta=eta, eta_poly=eta_p,
                 dh=dh, u=u, cx=cx, beta1=b1, beta2=b2, alpha1=a1,
                 t1=t1, p1=p1, rho1=rho1, w1=w1, w2=w2,
                 mach_w1=w1 / math.sqrt(gam * R_SI * t1),
                 t0_rel=t0_rel, p0_rel=p0_rel, gamma=gam, cp=cp,
                 r_mean=r_mean, height=height, chord=chord, solidity=solidity,
                 pitch=pitch, blades=blades,
                 reynolds=rho1 * w1 * chord / sutherland_mu(t1), de_haller=w2 / w1)


def efficiency_drop_from_loss(s: Stage, d_omega: float, rows: int = 2) -> float:
    """Stage efficiency change (fraction) if each of `rows` blade rows gains
    d_omega of total-pressure loss coefficient omega = (P01 - P02) / (P01 - p1).

    Lost work = T * ds, and ds = R * dP0/P0 for a small pressure loss, so
        d_eta = - rows * T * R * d_omega * (P01 - p1) / P01 / dh
    rows = 2: rotor and stator degrade alike (50 % reaction makes them similar).
    If every stage degrades alike, the whole-HPC change is about the same.
    """
    dyn = (s.p0_rel - s.p1) / s.p0_rel
    return -rows * s.t0_in * R_SI * d_omega * dyn / s.dh


def main() -> None:
    s = design_stage()
    print(f"HPC stage {s.number} of {N_STAGES} (mean line)\n")
    print(f"  inlet total      {s.t0_in:7.1f} K   {s.p0_in / 1e5:6.2f} bar")
    print(f"  stage PR         {s.pr:7.3f}     efficiency {s.eta:.4f} "
          f"(polytropic {s.eta_poly:.4f})")
    print(f"  blade speed U    {s.u:7.1f} m/s   axial Cx {s.cx:.1f} m/s")
    print(f"  rotor angles     beta1 {s.beta1:.1f} deg -> beta2 {s.beta2:.1f} deg "
          f"(turning {s.beta1 - s.beta2:.1f})")
    print(f"  inlet rel. Mach  {s.mach_w1:7.3f}     de Haller W2/W1 {s.de_haller:.3f}")
    print(f"  mean radius      {1000 * s.r_mean:7.1f} mm  blade height {1000 * s.height:.1f} mm")
    print(f"  chord            {1000 * s.chord:7.1f} mm  pitch {1000 * s.pitch:.1f} mm  "
          f"solidity {s.solidity:.2f}  blades {s.blades}")
    print(f"  Reynolds (chord) {s.reynolds:9.3g}")

    target = 0.0149                       # Week 2: HPC efficiency loss at failure
    d_omega = target / -efficiency_drop_from_loss(s, 1.0)
    print("\nPrediction to test with CFD (step 5):")
    print(f"  to lose {100 * target:.2f} % efficiency, each rotor and stator row needs a")
    print(f"  loss coefficient rise of about {d_omega:.4f} (a clean row is about 0.03-0.05)")


if __name__ == "__main__":
    main()
