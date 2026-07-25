"""DriveRack VENU360 integration.

Reality check: dbx/Harman does not publish a control protocol for the
VENU360. The only official control path is the free "DriveRack VENU360
Control" app — no SDK, no telnet spec, no documented API. Any code anywhere
that sends command strings to a VENU360 over TCP should be treated as
fiction until proven otherwise.

This module gives you three honest things instead of a fake protocol:

  1. DSPProcessor -- an abstract interface that any real backend
     (Qu-16, future Venue360 driver, etc.) can implement, so the optimizer
     never depends on fake bytes.

  2. Venue360ConfigExporter -- works today. Writes the computed delay/gain/
     EQ/zone settings to JSON or CSV that an engineer reads off and enters
     by hand, or that a future real driver consumes.

  3. capture_to_candidate_commands() -- scaffolding for deriving the real
     protocol from your OWN VENU360 unit via Wireshark capture of the
     official app's traffic. This is standard interoperability reverse-
     engineering of hardware you own; there is no shortcut.

The old Venus360 stub class is retained for backward compatibility with
DSPInterface.create() and existing tests.
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from acoustic_phase_optimizer.dsp.interface import DSPInterface
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# New honest API (no fake protocol bytes)
# ---------------------------------------------------------------------------

@dataclass
class ZoneSettings:
    zone_name: str
    delay_ms: float
    gain_db: float
    polarity_invert: bool
    crossover_hz: Optional[float] = None
    parametric_eq: Optional[list] = None  # list of dicts: {freq, gain_db, q}


class DSPProcessor(ABC):
    """Backend-agnostic interface. Implement for each real device."""

    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def apply_zone_settings(self, settings: ZoneSettings) -> None:
        """Push one zone's computed settings to the hardware."""

    @abstractmethod
    def supports_live_control(self) -> bool:
        """False for backends that can only export, not push live."""


class Venue360ConfigExporter(DSPProcessor):
    """
    Not a live link to the hardware -- generates a settings sheet the
    engineer applies by hand in the official VENU360 Control app.

    Fully functional right now for the optimizer's output path.
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self._pending: list[ZoneSettings] = []

    def connect(self) -> None:
        pass

    def close(self) -> None:
        self._flush()

    def supports_live_control(self) -> bool:
        return False

    def apply_zone_settings(self, settings: ZoneSettings) -> None:
        self._pending.append(settings)

    def _flush(self) -> None:
        if not self._pending:
            return
        data = [asdict(z) for z in self._pending]
        if self.output_path.suffix.lower() == ".json":
            self.output_path.write_text(json.dumps(data, indent=2))
        else:
            with open(self.output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "zone_name", "delay_ms", "gain_db",
                        "polarity_invert", "crossover_hz", "parametric_eq",
                    ],
                )
                writer.writeheader()
                writer.writerows(data)


def capture_to_candidate_commands(
    pcap_path: str | Path,
    venue360_ip: str,
) -> list[dict]:
    """
    Parse a Wireshark/tshark capture taken while YOU operate the official
    VENU360 Control app against your own unit, and pull out the raw TCP
    payload bytes exchanged with the unit's IP.

    This does not interpret the bytes -- it just extracts (direction, hex
    payload, timestamp, ports) so you can correlate app actions with
    changing byte patterns.

    Requires: pip install pyshark  (tshark/Wireshark must also be installed).
    """
    import pyshark

    cap = pyshark.FileCapture(
        str(pcap_path),
        display_filter=f"ip.addr == {venue360_ip} && tcp",
    )
    records: list[dict] = []
    for pkt in cap:
        if not hasattr(pkt, "tcp") or not hasattr(pkt.tcp, "payload"):
            continue
        direction = "to_unit" if pkt.ip.dst == venue360_ip else "from_unit"
        raw_hex = pkt.tcp.payload.replace(":", "")
        records.append({
            "timestamp": float(pkt.sniff_timestamp),
            "direction": direction,
            "src_port": pkt.tcp.srcport,
            "dst_port": pkt.tcp.dstport,
            "payload_hex": raw_hex,
        })
    cap.close()
    return records


# ---------------------------------------------------------------------------
# Legacy stub retained for backward compatibility
# ---------------------------------------------------------------------------

class Venus360(DSPInterface):
    """Interface stub for dbx DriveRack Venue360 (legacy, no real protocol).

    Note: there is no published control protocol for the VENU360. This stub
    stores values in memory and does not communicate with any hardware.
    Use Venue360ConfigExporter for production workflows.
    """

    MAX_CHANNELS = 4
    MAX_FIR_TAPS = 512
    MAX_EQ_BANDS = 10

    def __init__(self, address: str = "", port: int = 23, sample_rate: int = 48000):
        super().__init__(address, port, sample_rate)
        self._config: dict[int, dict[str, Any]] = {}
        self._initialize_config()

    def _initialize_config(self) -> None:
        for ch in range(1, self.MAX_CHANNELS + 1):
            self._config[ch] = {
                "delay_ms": None,
                "gain_db": None,
                "polarity_inverted": False,
                "crossover_freq": None,
                "crossover_slope": 24.0,
                "eq": [],
                "fir": None,
                "muted": False,
                "name": f"Channel {ch}",
            }

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def set_delay(self, channel: int, delay_ms: float) -> bool:
        if channel not in range(1, self.MAX_CHANNELS + 1):
            return False
        self._config[channel]["delay_ms"] = max(0.0, min(delay_ms, 1000.0))
        return True

    def set_gain(self, channel: int, gain_db: float) -> bool:
        if channel not in range(1, self.MAX_CHANNELS + 1):
            return False
        self._config[channel]["gain_db"] = max(-60.0, min(gain_db, 20.0))
        return True

    def set_polarity(self, channel: int, inverted: bool) -> bool:
        if channel not in range(1, self.MAX_CHANNELS + 1):
            return False
        self._config[channel]["polarity_inverted"] = inverted
        return True

    def set_crossover(self, channel: int, frequency_hz: float, slope_db_per_octave: float = 24.0) -> bool:
        if channel not in range(1, self.MAX_CHANNELS + 1):
            return False
        self._config[channel]["crossover_freq"] = frequency_hz
        self._config[channel]["crossover_slope"] = slope_db_per_octave
        return True

    def set_eq_parametric(self, channel: int, frequency_hz: float, gain_db: float, q: float) -> bool:
        if channel not in range(1, self.MAX_CHANNELS + 1):
            return False
        bands = self._config[channel]["eq"]
        if len(bands) >= self.MAX_EQ_BANDS:
            return False
        bands.append({"freq": frequency_hz, "gain": gain_db, "q": q})
        return True

    def set_fir_coefficients(self, channel: int, coefficients: NDArray[np.float64]) -> bool:
        if channel not in range(1, self.MAX_CHANNELS + 1):
            return False
        self._config[channel]["fir"] = coefficients[: self.MAX_FIR_TAPS]
        return True

    def get_delay(self, channel: int) -> Optional[float]:
        return self._config.get(channel, {}).get("delay_ms")

    def get_gain(self, channel: int) -> Optional[float]:
        return self._config.get(channel, {}).get("gain_db")

    def get_polarity(self, channel: int) -> Optional[bool]:
        return self._config.get(channel, {}).get("polarity_inverted")

    def get_crossover(self, channel: int) -> Optional[tuple[float, float]]:
        cfg = self._config.get(channel, {})
        freq = cfg.get("crossover_freq")
        if freq is None:
            return None
        return (freq, cfg.get("crossover_slope", 24.0))

    def mute_channel(self, channel: int, muted: bool) -> bool:
        if channel not in range(1, self.MAX_CHANNELS + 1):
            return False
        self._config[channel]["muted"] = muted
        return True

    def apply_configuration(self, config: dict) -> bool:
        for ch_str, settings in config.items():
            ch = int(ch_str)
            if "delay_ms" in settings:
                self.set_delay(ch, settings["delay_ms"])
            if "gain_db" in settings:
                self.set_gain(ch, settings["gain_db"])
            if "polarity_inverted" in settings:
                self.set_polarity(ch, settings["polarity_inverted"])
        return True

    def read_configuration(self) -> dict:
        return {
            str(ch): {
                "delay_ms": cfg["delay_ms"],
                "gain_db": cfg["gain_db"],
                "polarity_inverted": cfg["polarity_inverted"],
                "crossover_freq": cfg["crossover_freq"],
                "crossover_slope": cfg["crossover_slope"],
                "eq_count": len(cfg["eq"]),
                "fir_length": len(cfg["fir"]) if cfg["fir"] is not None else 0,
                "muted": cfg["muted"],
            }
            for ch, cfg in self._config.items()
        }

    def reset_to_defaults(self) -> bool:
        self._initialize_config()
        return True
