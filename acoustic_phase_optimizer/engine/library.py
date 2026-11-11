"""Real loudspeakers, from manufacturer datasheets.

Each entry holds what the datasheet says (coverage, max SPL, low-frequency limit and the level it is quoted
at) plus where it came from. Figures were checked against the manufacturer's page or spec sheet in
November 2026; where a figure could only be found on a reseller page, or the sources disagree, the entry
says so in `note` and is flagged `approximate`.

The simulator needs a few numbers datasheets don't give directly. These are derived by fixed rules
(see `to_speaker_type`) and shown to the user as estimates:

- high-pass corner: the quoted low limit moved up to an approximate -3 dB point
- pattern-control frequency: from woofer size and horizontal coverage (a wavelength-vs-aperture rule)
- line-array elements become a hang of 8 boxes: level + 10·log10(8) (a power sum: a curved hang does not
  add fully in phase), vertical coverage 8 × 2.5°
"""

from __future__ import annotations

import math

C = 343.0
HANG = 8  # boxes in a hang made from a line-array element

# id, brand, model, kind, woofer_in, coverage_h, coverage_v, max_spl, lf_hz, lf_db, hf_hz, source, note
# kind: point | element (line array box) | column | sub. lf_db: the level the low limit is quoted at.
_ROWS = [
    # ------------------------------------------------------------ portable / powered point source
    ("qsc_k12_2", "QSC", "K12.2", "point", 12, 75, 75, 132, 45, -10, None,
     "https://www.qscaudio.com/resource-files/productresources/spk/k.2/q_spk_k2_specs.pdf", ""),
    ("jbl_prx915", "JBL", "PRX915", "point", 15, 90, 50, 133, 48, -10, None,
     "https://jblpro.com/en/products/prx915", ""),
    ("jbl_eon715", "JBL", "EON715", "point", 15, 90, 60, 128, 45, -10, None,
     "https://jblpro.com/en/products/jbl-eon715", ""),
    ("yamaha_dxr12mk2", "Yamaha", "DXR12mkII", "point", 12, 90, 60, 134, 52, -10, None,
     "https://usa.yamaha.com/products/proaudio/speakers/dxr/specs.html", ""),
    ("yamaha_dzr12", "Yamaha", "DZR12", "point", 12, 90, 60, 139, 39, -10, None,
     "https://ro.yamaha.com/files/download/brochure/4/1174324/dzr_czr_brochure.pdf", ""),
    ("ev_elx200_12p", "Electro-Voice", "ELX200-12P", "point", 12, 90, 60, 130, 51, -10, None,
     "https://products.electrovoice.com/na/en/elx200-12p/", "An older data sheet lists 128 dB (calculated); the current page says 130 dB."),
    ("rcf_art912a", "RCF", "ART 912-A", "point", 12, 100, 60, 130, 50, None, None,
     "https://images.static-thomann.de/pics/atg/atgdata/document/specs/522521_specsheet_eng_art_912_a.pdf",
     "RCF quotes 50 Hz - 20 kHz without saying at what level; treated as -10 dB."),
    # ------------------------------------------------------------ installation / touring point source
    ("lacoustics_x12", "L-Acoustics", "X12", "point", 12, 90, 60, 136, 59, -10, None,
     "https://l-acoustics.com/products/x12",
     "L-Acoustics sources disagree on whether the 90° axis is horizontal or vertical; the horn is rotatable."),
    ("lacoustics_x8", "L-Acoustics", "X8", "point", 8, 100, 100, 129, 60, -10, None,
     "https://www.l-acoustics.com/products/x8/", ""),
    ("lacoustics_5xt", "L-Acoustics", "5XT", "point", 5, 110, 110, 121, 95, -10, None,
     "https://l-acoustics.com/products/5XT", ""),
    ("db_e12", "d&b audiotechnik", "E12", "point", 12, 80, 50, 134, 60, None, None,
     "https://www.dbaudio.com/assets/products/downloads/architectural-specifications/dbaudio-arch-specs-e12-en.txt",
     "Low-frequency limit not confirmed; 60 Hz is an estimate."),
    ("meyer_upj1p", "Meyer Sound", "UPJ-1P", "point", 10, 80, 50, 128, 55, None, None,
     "https://fullcompass.com/common/files/48041-UPJ1datasheet.pdf",
     "Peak SPL from a reseller listing, not confirmed on Meyer's datasheet. 55 Hz is Meyer's operating range."),
    # ------------------------------------------------------------ line array elements
    ("lacoustics_k2", "L-Acoustics", "K2", "element", 12, 70, 10, 145, 35, -10, None,
     "https://l-acoustics.com/documentation/LOUDSPEAKERS/K2/EN/Owner's%20Manual/K2_OM_EN.pdf",
     "Panflex: 70°, 110° or 90° asymmetric horizontally. Max SPL is for the 70° setting."),
    ("lacoustics_kara2", "L-Acoustics", "Kara II", "element", 8, 70, 10, 142, 55, -10, None,
     "https://l-acoustics.com/products/kara-ii",
     "Panflex: 70°, 110° or 90° asymmetric. The 110° setting is about 2 dB quieter."),
    ("db_ksl8", "d&b audiotechnik", "KSL8", "element", 10, 80, 10, 145, 54, -5, None,
     "https://www.dbaudio.com/global/en/products/series/sl-series/ksl8/", ""),
    ("meyer_leopard", "Meyer Sound", "LEOPARD", "element", 9, 110, 10, 142, 55, None, None,
     "https://docs.meyersound.com/products/en/datasheet---leopard---leopard-m80.html",
     "Meyer also quotes 133.5 dB linear peak (M-noise); 142 dB is its maximum SPL figure."),
    # ------------------------------------------------------------ columns
    ("bose_l1pro32", "Bose", "L1 Pro32 + Sub1", "column", 2, 180, 10, 123, 32, -10, None,
     "https://assets.boseprofessional.com/m/13c094483942315f/original/tds_L1Pro32_Sub1.pdf",
     "System figures with the Sub1 bass module (calculated peak). Bose calls the vertical pattern 0°; 10° is used here."),
    # ------------------------------------------------------------ subwoofers
    ("qsc_ks118", "QSC", "KS118", "sub", 18, 360, 360, 136, 35, -10, 111,
     "https://qsc.com/live-sound/products/loudspeakers/active-loudspeakers/ks-series/ks118", ""),
    ("jbl_srx828sp", "JBL", "SRX828SP", "sub", 18, 360, 360, 141, 29, -10, 150,
     "https://jblpro.com/en/products/srx828sp", ""),
    ("yamaha_dxs18xlf", "Yamaha", "DXS18XLF", "sub", 18, 360, 360, 136, 30, -10, 150,
     "https://uk.yamaha.com/en/business/audio/products/speakers/dxs-xlf/", ""),
    ("lacoustics_ks28", "L-Acoustics", "KS28", "sub", 18, 360, 360, 143, 25, -10, None,
     "https://www.l-acoustics.com/products/KS28", "Two 18\" drivers."),
    ("db_sl_sub", "d&b audiotechnik", "SL-SUB", "sub", 21, 360, 360, 144, 33, -5, 84,
     "https://www.dbaudio.com/global/en/products/series/sl-series/sl-sub/", "Cardioid by design; modelled here as omnidirectional."),
    ("db_b22_sub", "d&b audiotechnik", "B22-SUB", "sub", 21, 360, 360, 143, 37, -5, 90,
     "https://www.dbaudio.com/global/en/products/subwoofers/b22-sub/", ""),
    ("meyer_1100lfc", "Meyer Sound", "1100-LFC", "sub", 18, 360, 360, 140, 28, None, 100,
     "https://docs.meyersound.com/pdf/1100-lfc_ds_b3.pdf", "Two 18\" drivers. Linear peak SPL (pink noise); 28-100 Hz operating range."),
]

# How far above a quoted low limit the -3 dB point roughly sits, for a typical vented box.
_TO_3DB = {-10: 1.3, -6: 1.15, -5: 1.12, -3: 1.0, None: 1.2}


def _pattern_control_hz(woofer_in: float, coverage_h: float, kind: str) -> float:
    if kind == "sub":
        return 10000.0
    if kind == "column":
        return 500.0
    if kind == "element":
        return 400.0
    aperture = woofer_in * 0.0254
    f = 0.6 * C / (aperture * math.sin(math.radians(min(coverage_h, 170)) / 2))
    return float(round(min(3000.0, max(300.0, f)), -1))


def to_speaker_type(row: tuple) -> dict:
    sid, brand, model, kind, woofer, cov_h, cov_v, spl, lf, lf_db, hf, source, note = row
    hp = round(lf * _TO_3DB[lf_db])
    level = spl
    label = f"{brand} {model}"
    if kind == "element":
        level = round(spl + 10 * math.log10(HANG), 1)
        cov_v = HANG * 2.5
        label = f"{brand} {model}, hang of {HANG}"
    return {
        "id": sid, "label": label, "brand": brand, "group": brand,
        "coverage_h": cov_h, "coverage_v": cov_v, "level_db": min(160, level), "hp_hz": hp,
        "lp_hz": round(hf / _TO_3DB[lf_db]) if hf else (100 if kind == "sub" else 0),
        "directivity_hz": _pattern_control_hz(woofer, cov_h, kind),
        "role": "sub" if kind == "sub" else "main",
        "spec": {
            "kind": kind, "coverage": f"{cov_h:g}° × {row[6]:g}°" if kind != "sub" else "omnidirectional",
            "max_spl": spl, "low_limit": lf, "low_limit_db": lf_db, "high_limit": hf,
            "source": source, "note": note,
            "approximate": bool(note) and any(w in note for w in ("not confirmed", "estimate", "reseller")),
            "estimated": ["pattern-control frequency", "high-pass corner"] + (
                [f"hang of {HANG}: level +{10 * math.log10(HANG):.0f} dB, vertical {HANG * 2.5:g}°"] if kind == "element" else []),
        },
    }


LIBRARY: list[dict] = [to_speaker_type(r) for r in _ROWS]
