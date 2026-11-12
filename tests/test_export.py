"""Processor exports: the VENU360 sheet, REW / Equalizer APO filter text and miniDSP biquads."""

from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pytest

from acoustic_phase_optimizer import export
from acoustic_phase_optimizer.engine import presets
from acoustic_phase_optimizer.engine.eq import biquad_coeffs, biquad_response, peq_response
from acoustic_phase_optimizer.engine.field import box_response
from acoustic_phase_optimizer.engine.model import PEQBand, Project


def project(n: int = 2) -> Project:
    p = presets.VENUES["conference"][1]()
    s = p.speakers[0]
    s.name = "Main L/R"
    s.hp_hz, s.lp_hz = 80.0, 0.0
    s.delay_ms = 12.5
    s.peq = [PEQBand(type="bell", freq=250, gain_db=-4.5, q=2.0), PEQBand(type="low_shelf", freq=120, gain_db=3.0, q=0.7),
             PEQBand(type="high_shelf", freq=8000, gain_db=-2.0, q=0.7), PEQBand(type="highpass", freq=40, q=0.9),
             PEQBand(type="bell", freq=1000, gain_db=0.0, q=1.0)]  # a flat bell is not a filter
    p.speakers[1].name = "Main L/R"  # the same name twice must not collide in a zip
    p.speakers = p.speakers[:n]
    return p


def minidsp_sections(text: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read miniDSP text back as (b, a) in the usual convention (denominator 1 + a1 z^-1 + a2 z^-2)."""
    out = []
    for blk in re.findall(r"biquad\d+,\s*b0=([^,]+),\s*b1=([^,]+),\s*b2=([^,]+),\s*a1=([^,]+),\s*a2=([^,]+),", text):
        b0, b1, b2, a1, a2 = map(float, blk)
        out.append((np.array([b0, b1, b2]), np.array([1.0, -a1, -a2])))
    return out


def test_minidsp_sign_convention_matches_published_coefficients():
    # Filter 1 of the REW biquad export the open-source `minidsp` crate tests against: a bell, so its gain at DC is 1.
    # That holds only with a1 and a2 negated (y = b0 x + b1 x1 + b2 x2 + a1 y1 + a2 y2).
    b0, b1, b2, a1, a2 = 0.9991949, -1.9973354, 0.9981886, 1.9973354, -0.99738353
    assert (b0 + b1 + b2) / (1 - a1 - a2) == pytest.approx(1.0, abs=0.01)
    assert abs((b0 + b1 + b2) / (1 + a1 + a2)) < 0.01  # the other sign gives nonsense
    # and ours round-trips through the same reading
    b, a = biquad_coeffs("bell", 100, -6, 4, 48000)
    (rb, ra), = minidsp_sections(export.minidsp_text([(b, a)]))
    assert np.allclose(rb, b) and np.allclose(ra, a)


def test_minidsp_biquads_reproduce_the_model():
    p = project()
    s = p.speakers[0]
    z = zipfile.ZipFile(io.BytesIO(export.to_minidsp(p)))
    names = z.namelist()
    assert "README.txt" in names and "48 kHz/peq/01 Main L_R.txt" in names and "96 kHz/crossover/01 Main L_R.txt" in names
    assert len(set(names)) == len(names) and all("/" not in n.split(" kHz/")[-1].split("/", 1)[-1] for n in names)
    f = np.geomspace(20, 16000, 200)
    for fs in (48000, 96000):
        peq = minidsp_sections(z.read(f"{fs // 1000} kHz/peq/01 Main L_R.txt").decode())
        assert len(peq) == 4  # the flat bell left out
        h = np.prod([biquad_response(b, a, f, fs) for b, a in peq], axis=0)
        ref = peq_response(s.peq, f)  # the engine's own (designed at 96 kHz)
        band = f < fs / 6  # above that the bilinear warp of a 48 kHz design differs from the 96 kHz model
        assert np.max(np.abs(20 * np.log10(np.abs(h[band]) / np.abs(ref[band])))) < 0.05
        xo = minidsp_sections(z.read(f"{fs // 1000} kHz/crossover/01 Main L_R.txt").decode())
        assert len(xo) == 2  # LR24 high-pass: two Butterworth sections
        hx = np.prod([biquad_response(b, a, f, fs) for b, a in xo], axis=0)
        assert np.max(np.abs(20 * np.log10(np.abs(hx[band]) / np.abs(box_response(s, f)[band])))) < 0.05
    readme = z.read("README.txt").decode()
    assert "delay 12.50 ms" in readme and "a1 and a2 are negated" in readme


def test_rew_apo_lines():
    p = project()
    z = zipfile.ZipFile(io.BytesIO(export.to_rew_apo(p)))
    assert sorted(z.namelist()) == ["01 Main L_R.txt", "02 Main L_R.txt"]
    text = z.read("01 Main L_R.txt").decode()
    lines = [x for x in text.splitlines() if x.startswith("Filter")]
    assert lines == ["Filter 1: ON HPQ Fc 80 Hz Q 0.7071", "Filter 2: ON HPQ Fc 80 Hz Q 0.7071",
                     "Filter 3: ON PK Fc 250 Hz Gain -4.5 dB Q 2.0000", "Filter 4: ON LSC Fc 120 Hz Gain 3.0 dB Q 0.7000",
                     "Filter 5: ON HSC Fc 8000 Hz Gain -2.0 dB Q 0.7000", "Filter 6: ON HPQ Fc 40 Hz Q 0.9000"]
    assert "Delay: 12.50 ms" in text and "Preamp:" in text and "\r\n" in text


def test_venu360_sheet_and_limits():
    p = project()
    txt = export.to_venu360(p)
    assert "OUTPUT 1  \"Main L/R\"" in txt and "High Pass 80 Hz, Linkwitz-Riley 24 dB/oct" in txt
    assert "Alignment Delay  12.50 ms" in txt and "Low Shelf" in txt and "by hand" in txt
    assert "PEQ band 4 is a 2nd-order high pass" in txt
    clean = project()
    clean.speakers[0].peq = clean.speakers[0].peq[:3]
    assert "Within the VENU360's published limits." in export.to_venu360(clean)
    p = project(2)
    p.speakers = [p.speakers[0].model_copy(update={"id": f"s{i}", "name": f"S{i}"}) for i in range(8)]
    p.speakers[3].delay_ms = 1200
    p.speakers[2].peq = [PEQBand(type="bell", freq=60, gain_db=-6, q=20)]
    checks = export.venu360_checks(p)
    assert any("8 outputs" in c for c in checks)
    assert any("Output 4" in c and "1000 ms" in c for c in checks)
    assert any("Output 3" in c and "Q 20" in c for c in checks)
    assert any("Output 1" in c and "high pass" in c for c in checks)  # the PEQ high-pass band
    assert "OUTPUT 2 (unit 2)" in export.to_venu360(p)
