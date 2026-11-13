"""Gas properties and the basic compressor / turbine equations.

Units follow the C-MAPSS data: temperature in degR, pressure in psia,
energy in BTU per lbm of gas.

All temperatures and pressures here are TOTAL (stagnation) values,
the ones a probe facing the flow measures.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Gas:
    """A gas with constant specific heat (a 'calorically perfect' gas)."""

    name: str
    cp: float     # specific heat at constant pressure, BTU/(lbm*degR)
    gamma: float  # ratio of specific heats, cp/cv

    @property
    def k(self) -> float:
        """Exponent (gamma - 1) / gamma in the isentropic relation T2/T1 = (P2/P1)**k."""
        return (self.gamma - 1.0) / self.gamma


# Cold section (fan, compressors). gamma = 1.37 is a mean value for air
# between about 520 and 1600 degR; 1.40 is only right near room temperature.
AIR = Gas("air", cp=0.245, gamma=1.37)

# Hot section (combustor exit, turbines): air mixed with burnt fuel.
HOT_GAS = Gas("combustion gas", cp=0.276, gamma=1.33)


# --- Compressor -------------------------------------------------------------
# Ideal (isentropic) exit temperature: T_ideal = T_in * PR**k
# Isentropic efficiency:  eta = (T_ideal - T_in) / (T_out - T_in)
# An efficiency below 1 means extra temperature rise (wasted work) for the
# same pressure rise. HPC degradation in FD001 is exactly this.

def compressor_exit_temp(t_in: float, pr: float, eta: float, gas: Gas = AIR) -> float:
    """Exit total temperature for a given pressure ratio and efficiency."""
    t_ideal = t_in * pr ** gas.k
    return t_in + (t_ideal - t_in) / eta


def compressor_efficiency(t_in: float, t_out: float, pr: float, gas: Gas = AIR) -> float:
    """Isentropic efficiency from measured inlet/exit temperatures and pressure ratio."""
    return t_in * (pr ** gas.k - 1.0) / (t_out - t_in)


def compressor_pressure_ratio(t_in: float, t_out: float, eta: float, gas: Gas = AIR) -> float:
    """Pressure ratio that gives a measured temperature rise at a given efficiency."""
    return (1.0 + eta * (t_out / t_in - 1.0)) ** (1.0 / gas.k)


# --- Turbine ----------------------------------------------------------------
# Expansion through pressure ratio PR = P_in / P_out (> 1).
# eta = (T_in - T_out) / (T_in - T_ideal), with T_ideal = T_in / PR**k

def turbine_exit_temp(t_in: float, pr: float, eta: float, gas: Gas = HOT_GAS) -> float:
    """Exit total temperature after expanding through pressure ratio pr."""
    return t_in * (1.0 - eta * (1.0 - pr ** -gas.k))


# --- Work -------------------------------------------------------------------

def specific_work(t_in: float, t_out: float, gas: Gas) -> float:
    """Work per lbm of gas, BTU/lbm. Positive when temperature rises (compressor)."""
    return gas.cp * (t_out - t_in)
