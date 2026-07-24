from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from acoustic_phase_optimizer.weather.models import YearlyAverages


@dataclass
class AcousticWeatherParams:
    temperature_c: float = 20.0
    humidity_percent: float = 50.0
    pressure_hpa: float = 1013.25
    speed_of_sound: float = 343.0


def temperature_to_speed_of_sound(temp_c: float) -> float:
    return 331.3 * np.sqrt(1.0 + temp_c / 273.15)


def air_absorption_coefficient(
    freq_hz: float,
    temp_c: float,
    humidity_percent: float,
    pressure_hpa: float,
) -> float:
    T = temp_c + 273.15
    T0 = 293.15
    T01 = 273.16
    pr = pressure_hpa / 1013.25

    h = humidity_percent / 100.0
    psat = 610.78 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    h_mol = h * psat / pressure_hpa * 100.0

    frO = pressure_hpa / 1013.25 * (24.0 + 4.04e4 * h_mol * (0.02 + h_mol) / (0.391 + h_mol))
    frN = pressure_hpa / 1013.25 * np.sqrt(T / T0) * (
        9.0 + 280.0 * h_mol * np.exp(-4.17 * ((T / T0) ** (-1.0 / 3.0) - 1.0))
    )

    z = 2.0 * np.pi * freq_hz / frN
    y = 2.0 * np.pi * freq_hz / frO

    alpha = (
        1.6e-10 * (T / T01) ** (1.0 / 2.0) * (freq_hz**2) / pr
        + 1.59e-8 * (T / T01) ** (-5.0 / 2.0) * (freq_hz**2 * np.exp(-2239.1 / T)) / (frN + freq_hz**2 / frN)
        + 5.74e-8 * (T / T01) ** (-5.0 / 2.0) * (freq_hz**2 * np.exp(-3352.0 / T)) / (frO + freq_hz**2 / frO)
        + 1.59e-8 * (T / T01) ** (-5.0 / 2.0) * (freq_hz**2 * np.exp(-2239.1 / T)) / (frN + y)
        + 5.74e-8 * (T / T01) ** (-5.0 / 2.0) * (freq_hz**2 * np.exp(-3352.0 / T)) / (frO + z)
    )
    return float(max(alpha, 0.0))


def weather_to_acoustic_params(
    averages: YearlyAverages,
) -> AcousticWeatherParams:
    c = temperature_to_speed_of_sound(averages.temperature_mean)
    return AcousticWeatherParams(
        temperature_c=averages.temperature_mean,
        humidity_percent=averages.humidity_mean,
        pressure_hpa=averages.pressure_mean,
        speed_of_sound=c,
    )
