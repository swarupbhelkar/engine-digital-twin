"""Gas properties and the basic compressor / turbine equations.

Units follow the C-MAPSS data: temperature in degR, pressure in psia,
energy in BTU per lbm.

All temperatures and pressures are TOTAL (stagnation) values unless a
name says "static".

Step 2 upgrade: specific heat cp now changes with temperature and with
fuel-air ratio (far). Air at 1600 degR has a cp about 10% higher than at
room temperature, which matters for efficiency numbers.

Instead of cp*dT we use two functions of temperature:
  h(T)    enthalpy, the energy content of the gas        dh   = cp dT
  phi(T)  entropy function, used for ideal compression   dphi = cp dT / T
An ideal (isentropic) process from P1 to P2 obeys:
  phi(T2) - phi(T1) = R * ln(P2 / P1)
"""
import numpy as np
from numpy.polynomial import Polynomial as Poly

# --- Property polynomials ---------------------------------------------------
# cp in kJ/(kg K) as a polynomial in z = T[K] / 1000.
# Source: Walsh & Fletcher, "Gas Turbine Performance", 2nd ed., ch. 3.
# AIR: dry air. FUEL: correction for burnt kerosene, scaled by far/(1+far).
_CP_AIR_SI = Poly([0.992313, 0.236688, -1.852148, 6.083152, -8.893933,
                   7.097112, -3.234725, 0.794571, -0.081873])
_CP_FUEL_SI = Poly([-0.718874, 8.747481, -15.863157, 17.254096, -10.233795,
                    3.081778, -0.361112, -0.003919])

KJKG_TO_BTULBM = 0.429923        # 1 kJ/kg   = 0.4299 BTU/lbm
KJKGK_TO_BTULBMR = 0.238846      # 1 kJ/kg K = 0.2388 BTU/(lbm degR)
R_GAS = 0.068557                 # gas constant of air, BTU/(lbm degR)
T_REF = 518.67                   # reference temperature (59 degF), h = 0 here
LHV = 18_400.0                   # jet fuel lower heating value, BTU/lbm

# Same polynomials, rescaled so cp comes out in BTU/(lbm degR)
_CP_AIR = _CP_AIR_SI * KJKGK_TO_BTULBMR
_CP_FUEL = _CP_FUEL_SI * KJKGK_TO_BTULBMR

# Integrals, done once. With z = T/1800 (T in degR): dT = 1800 dz.
_H_AIR = _CP_AIR.integ() * 1800.0
_H_FUEL = _CP_FUEL.integ() * 1800.0
_Z_REF = T_REF / 1800.0


def _z(t):
    return np.asarray(t, dtype=float) / 1800.0


def _blend(far):
    """Weight of the fuel correction term."""
    return far / (1.0 + far)


def cp(t, far=0.0):
    """Specific heat, BTU/(lbm degR), at temperature t (degR)."""
    return _CP_AIR(_z(t)) + _blend(far) * _CP_FUEL(_z(t))


def enthalpy(t, far=0.0):
    """Enthalpy per lbm of gas, BTU/lbm, measured from T_REF."""
    z = _z(t)
    h_air = _H_AIR(z) - _H_AIR(_Z_REF)
    h_fuel = _H_FUEL(z) - _H_FUEL(_Z_REF)
    return h_air + _blend(far) * h_fuel


def entropy_fn(t, far=0.0):
    """phi(T) = integral of cp/T dT, BTU/(lbm degR). Only differences matter."""
    z = _z(t)

    def phi(poly):
        c = poly.coef
        # integral of (c0 + c1 z + c2 z^2 ...) / z dz = c0 ln z + c1 z + c2 z^2/2 ...
        rest = Poly(np.concatenate([[0.0], c[1:] / np.arange(1, len(c))]))
        return c[0] * np.log(z) + rest(z)

    return phi(_CP_AIR) + _blend(far) * phi(_CP_FUEL)


def gamma(t, far=0.0):
    """Ratio of specific heats cp/cv at temperature t."""
    c = cp(t, far)
    return c / (c - R_GAS)


# --- Inverting the property functions ---------------------------------------
# Newton's method: T_new = T - (f(T) - target) / f'(T). Works on arrays too.

def temp_from_enthalpy(h, far=0.0):
    """Temperature (degR) that has enthalpy h."""
    t = np.full_like(np.asarray(h, dtype=float), 1000.0)
    for _ in range(30):
        t = t - (enthalpy(t, far) - h) / cp(t, far)
    return t


def temp_from_entropy_fn(phi, far=0.0):
    """Temperature (degR) that has entropy function phi."""
    t = np.full_like(np.asarray(phi, dtype=float), 1000.0)
    for _ in range(30):
        t = t - (entropy_fn(t, far) - phi) * t / cp(t, far)
    return t


def isentropic_temp(t_in, pr, far=0.0):
    """Exit temperature of an ideal process through pressure ratio pr = P_out/P_in."""
    return temp_from_entropy_fn(entropy_fn(t_in, far) + R_GAS * np.log(pr), far)


# --- Compressor -------------------------------------------------------------
# Isentropic efficiency = ideal enthalpy rise / real enthalpy rise.
# Below 1 means extra temperature rise (wasted work) for the same pressure
# rise. HPC degradation in FD001 is exactly this.

def compressor_exit_temp(t_in, pr, eta, far=0.0):
    """Exit total temperature for a given pressure ratio and efficiency."""
    h_in = enthalpy(t_in, far)
    dh_ideal = enthalpy(isentropic_temp(t_in, pr, far), far) - h_in
    return temp_from_enthalpy(h_in + dh_ideal / eta, far)


def compressor_efficiency(t_in, t_out, pr, far=0.0):
    """Isentropic efficiency from measured inlet/exit temperatures and pressure ratio."""
    h_in = enthalpy(t_in, far)
    dh_ideal = enthalpy(isentropic_temp(t_in, pr, far), far) - h_in
    return dh_ideal / (enthalpy(t_out, far) - h_in)


def compressor_pressure_ratio(t_in, t_out, eta, far=0.0):
    """Pressure ratio that gives a measured temperature rise at a given efficiency."""
    h_in = enthalpy(t_in, far)
    h_ideal = h_in + eta * (enthalpy(t_out, far) - h_in)
    t_ideal = temp_from_enthalpy(h_ideal, far)
    return np.exp((entropy_fn(t_ideal, far) - entropy_fn(t_in, far)) / R_GAS)


# --- Turbine ----------------------------------------------------------------
# Efficiency = real enthalpy drop / ideal enthalpy drop (the inverse of a
# compressor). pr = P_in / P_out > 1.

def turbine_exit_temp(t_in, pr, eta, far=0.0):
    """Exit total temperature after expanding through pressure ratio pr."""
    h_in = enthalpy(t_in, far)
    dh_ideal = h_in - enthalpy(isentropic_temp(t_in, 1.0 / pr, far), far)
    return temp_from_enthalpy(h_in - eta * dh_ideal, far)


def turbine_pressure_ratio(t_in, t_out, eta, far=0.0):
    """Pressure ratio P_in/P_out that gives a required temperature drop."""
    h_in = enthalpy(t_in, far)
    h_ideal = h_in - (h_in - enthalpy(t_out, far)) / eta
    t_ideal = temp_from_enthalpy(h_ideal, far)
    return np.exp((entropy_fn(t_in, far) - entropy_fn(t_ideal, far)) / R_GAS)
