"""Load NASA C-MAPSS turbofan degradation data (FD001-FD004).

Each row is one engine cycle: unit number, cycle, 3 operating settings, 21 sensors.
Training engines run until failure; test engines stop early, and RUL_FD00x.txt
gives the true remaining useful life at the last test cycle.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

SETTINGS = ["setting_1", "setting_2", "setting_3"]

# Sensor meanings from Saxena et al. (2008), "Damage propagation modeling
# for aircraft engine run-to-failure simulation".
SENSORS = {
    "s1": ("T2", "Fan inlet total temperature", "degR"),
    "s2": ("T24", "LPC outlet total temperature", "degR"),
    "s3": ("T30", "HPC outlet total temperature", "degR"),
    "s4": ("T50", "LPT outlet total temperature", "degR"),
    "s5": ("P2", "Fan inlet pressure", "psia"),
    "s6": ("P15", "Bypass-duct total pressure", "psia"),
    "s7": ("P30", "HPC outlet total pressure", "psia"),
    "s8": ("Nf", "Physical fan speed", "rpm"),
    "s9": ("Nc", "Physical core speed", "rpm"),
    "s10": ("epr", "Engine pressure ratio (P50/P2)", "-"),
    "s11": ("Ps30", "HPC outlet static pressure", "psia"),
    "s12": ("phi", "Fuel flow / Ps30", "pps/psi"),
    "s13": ("NRf", "Corrected fan speed", "rpm"),
    "s14": ("NRc", "Corrected core speed", "rpm"),
    "s15": ("BPR", "Bypass ratio", "-"),
    "s16": ("farB", "Burner fuel-air ratio", "-"),
    "s17": ("htBleed", "Bleed enthalpy", "-"),
    "s18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "s20": ("W31", "HPT coolant bleed", "lbm/s"),
    "s21": ("W32", "LPT coolant bleed", "lbm/s"),
}

COLUMNS = ["unit", "cycle"] + SETTINGS + list(SENSORS)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    return df


def load_train(subset: str = "FD001") -> pd.DataFrame:
    """Training data with an RUL column (cycles left until failure)."""
    df = _read(DATA_DIR / f"train_{subset}.txt")
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = last - df["cycle"]
    return df


def load_test(subset: str = "FD001") -> pd.DataFrame:
    """Test data with an RUL column, using the true RUL at each unit's last cycle."""
    df = _read(DATA_DIR / f"test_{subset}.txt")
    true_rul = pd.read_csv(DATA_DIR / f"RUL_{subset}.txt", header=None).iloc[:, 0]
    true_rul.index = range(1, len(true_rul) + 1)
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = df["unit"].map(true_rul) + (last - df["cycle"])
    return df


def label(sensor: str) -> str:
    """Readable axis label, e.g. 's3' -> 'T30 (degR)'."""
    tag, _, unit = SENSORS[sensor]
    return f"{tag} ({unit})" if unit != "-" else tag


if __name__ == "__main__":
    tr, te = load_train(), load_test()
    print(f"train: {tr.unit.nunique()} engines, {len(tr)} rows")
    print(f"test:  {te.unit.nunique()} engines, {len(te)} rows")
