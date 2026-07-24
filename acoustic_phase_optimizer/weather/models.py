from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Location:
    name: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    country: str = ""


@dataclass
class WeatherData:
    date: str = ""
    temperature_mean: float = 20.0
    humidity_mean: float = 50.0
    pressure_mean: float = 1013.25


@dataclass
class YearlyAverages:
    location: Location = field(default_factory=Location)
    temperature_mean: float = 20.0
    humidity_mean: float = 50.0
    pressure_mean: float = 1013.25
    sample_count: int = 0
    daily_data: list[WeatherData] = field(default_factory=list)
