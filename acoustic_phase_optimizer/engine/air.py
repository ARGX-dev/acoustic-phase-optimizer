"""Properties of air: speed of sound and atmospheric absorption (ISO 9613-1)."""

from __future__ import annotations

import numpy as np

P_REF_KPA = 101.325
T0_K = 293.15  # reference temperature, 20 degC
T01_K = 273.16  # triple point of water
R_GAS = 8.314462  # J/(mol K)
M_DRY = 0.0289647  # kg/mol
M_WATER = 0.0180153  # kg/mol


def saturation_pressure_ratio(temperature_k: float) -> float:
    """Saturation vapour pressure of water relative to 101.325 kPa (ISO 9613-1)."""
    return float(10.0 ** (-6.8346 * (T01_K / temperature_k) ** 1.261 + 4.6151))


def speed_of_sound(temperature_c: float, humidity_pct: float = 50.0, pressure_kpa: float = P_REF_KPA) -> float:
    """Speed of sound in humid air, m/s (ideal-gas mixture of dry air and water vapour)."""
    T = float(temperature_c) + 273.15
    xw = humidity_pct / 100.0 * saturation_pressure_ratio(T) * P_REF_KPA / pressure_kpa
    xw = float(np.clip(xw, 0.0, 0.2))  # mole fraction of water vapour
    molar_mass = M_DRY - (M_DRY - M_WATER) * xw
    gamma = (7.0 + xw) / (5.0 + xw)
    return float(np.sqrt(gamma * R_GAS * T / molar_mass))


def air_absorption_db_per_m(
    freqs: np.ndarray,
    temperature_c: float = 20.0,
    humidity_pct: float = 50.0,
    pressure_kpa: float = P_REF_KPA,
) -> np.ndarray:
    """Pure-tone atmospheric attenuation coefficient in dB/m (ISO 9613-1)."""
    f = np.asarray(freqs, dtype=float)
    T = temperature_c + 273.15
    pa_pr = pressure_kpa / P_REF_KPA
    h = max(humidity_pct, 0.5) * saturation_pressure_ratio(T) / pa_pr  # molar concentration, %
    tr = T / T0_K
    fr_o = pa_pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pa_pr * tr ** -0.5 * (9.0 + 280.0 * h * np.exp(-4.170 * (tr ** (-1.0 / 3.0) - 1.0)))
    f2 = f * f
    return 8.686 * f2 * (
        1.84e-11 / pa_pr * tr ** 0.5
        + tr ** -2.5
        * (
            0.01275 * np.exp(-2239.1 / T) / (fr_o + f2 / fr_o)
            + 0.1068 * np.exp(-3352.0 / T) / (fr_n + f2 / fr_n)
        )
    )
