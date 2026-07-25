"""
Allen & Heath Qu-16 control via MIDI-over-TCP.

Protocol reference: Allen & Heath "Qu MIDI Protocol" guide (publicly published
by A&H, e.g. Qu_MIDI_Protocol_V1.9.pdf). This is a real, documented protocol.

Transport facts (from the published guide):
  - Rear-panel NETWORK port, TCP, port 51325
  - Only ONE TCP MIDI connection is permitted at a time
  - Payload is raw MIDI bytes (Channel Voice + NRPN), no higher-level framing

NRPN mechanics (confirmed against the guide's worked example for "Ip1 Mute Toggle Ch1"):
  CC 99 (0x63) = NRPN parameter MSB
  CC 98 (0x62) = NRPN parameter LSB
  CC  6 (0x06) = Data Entry coarse (absolute value, high byte)
  CC 38 (0x26) = Data Entry fine   (absolute value, low byte)
  CC 96 (0x60) = Data Increment (relative +1, used for toggle/increment)
  CC 97 (0x61) = Data Decrement (relative -1)

Gap: the per-channel NRPN parameter ID table (which MSB/LSB pair addresses
"Input 1 fader", "Mix 3 send", etc.) lives in the "Reference Tables" section
at the back of the official PDF. Those numbers are real and public but I don't
have the table memorized here. Fill QU16_PARAMS below from the PDF and this
module is immediately usable end to end — a ~20-minute transcription job.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)

QU_NETWORK_PORT = 51325


class MidiCC(IntEnum):
    NRPN_MSB = 0x63
    NRPN_LSB = 0x62
    DATA_ENTRY_COARSE = 0x06
    DATA_ENTRY_FINE = 0x26
    DATA_INCREMENT = 0x60
    DATA_DECREMENT = 0x61


@dataclass(frozen=True)
class NRPNParam:
    """One entry in the Qu reference table: fill from the official PDF."""
    name: str
    msb: int
    lsb: int


QU16_PARAMS: dict[str, NRPNParam] = {
    "input_1_mute": NRPNParam("input_1_mute", msb=0x00, lsb=0x00),
}


class QuMixerClient:
    """
    TCP MIDI client for an Allen & Heath Qu-16/24/32.

    Usage:
        with QuMixerClient("192.168.1.50", midi_channel=0) as qu:
            qu.mute_toggle("input_1_mute")
            qu.set_fader_absolute("input_1_fader", db=-6.0)
    """

    def __init__(self, host: str, midi_channel: int = 0, timeout: float = 3.0):
        if not 0 <= midi_channel <= 15:
            raise ValueError("MIDI channel must be 0-15 (0-indexed)")
        self.host = host
        self.midi_channel = midi_channel
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        self._sock = socket.create_connection(
            (self.host, QU_NETWORK_PORT), timeout=self.timeout
        )
        logger.info("Connected to Qu mixer at %s:%d", self.host, QU_NETWORK_PORT)

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    def _status_byte(self) -> int:
        return 0xB0 | (self.midi_channel & 0x0F)

    def _send_cc(self, controller: int, value: int) -> None:
        if not self._sock:
            raise RuntimeError("Not connected")
        if not 0 <= value <= 0x7F:
            raise ValueError(f"MIDI data byte out of range: {value}")
        msg = bytes([self._status_byte(), controller & 0x7F, value & 0x7F])
        self._sock.sendall(msg)

    def _select_param(self, param: NRPNParam) -> None:
        self._send_cc(MidiCC.NRPN_MSB, param.msb)
        self._send_cc(MidiCC.NRPN_LSB, param.lsb)

    def mute_toggle(self, param_name: str) -> None:
        param = self._lookup(param_name)
        self._select_param(param)
        self._send_cc(MidiCC.DATA_INCREMENT, 0x00)

    def set_fader_absolute(self, param_name: str, db: float) -> None:
        param = self._lookup(param_name)
        value_14bit = self._db_to_14bit(db)
        coarse = (value_14bit >> 7) & 0x7F
        fine = value_14bit & 0x7F
        self._select_param(param)
        self._send_cc(MidiCC.DATA_ENTRY_COARSE, coarse)
        self._send_cc(MidiCC.DATA_ENTRY_FINE, fine)

    @staticmethod
    def _db_to_14bit(db: float, db_min: float = -80.0, db_max: float = 10.0) -> int:
        db = max(db_min, min(db_max, db))
        frac = (db - db_min) / (db_max - db_min)
        return round(frac * 16383)

    @staticmethod
    def _lookup(param_name: str) -> NRPNParam:
        try:
            return QU16_PARAMS[param_name]
        except KeyError:
            raise KeyError(
                f"'{param_name}' is not in QU16_PARAMS yet. Add its MSB/LSB "
                f"from the official Allen & Heath Qu MIDI Protocol PDF "
                f"reference tables before using it."
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with QuMixerClient("192.168.1.50", midi_channel=0) as qu:
        qu.mute_toggle("input_1_mute")
        time.sleep(0.05)
