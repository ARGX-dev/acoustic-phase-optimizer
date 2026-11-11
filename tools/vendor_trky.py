"""Copy the .trky code out of the converter page (trky.html) into web/vendor/trky, byte for byte.

The page holds three self-contained scripts (core, audio coder, FLAC coder). They are cut at their <script> tags and
written unchanged, so the vendored files can be checked against the page at any time:

    python tools/vendor_trky.py path/to/trky.html          # write the files
    python tools/vendor_trky.py path/to/trky.html --check  # only compare
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "acoustic_phase_optimizer" / "web" / "vendor" / "trky"
# each script is recognised by its opening comment, so the cut does not depend on line numbers
PARTS = {
    "core.js": b"/* TRKY core:",
    "audio.js": b"/* TRKY audio module:",
    "flac.js": b"/* TRKY FLAC module:",
}


def scripts(page: bytes) -> dict[str, bytes]:
    found = {}
    for m in re.finditer(rb"<script>\r?\n(.*?)</script>", page, re.S):
        body = m.group(1)
        for name, head in PARTS.items():
            if body.startswith(head):
                if name in found:
                    raise SystemExit(f"{name}: found twice in the page")
                found[name] = body
    missing = sorted(set(PARTS) - set(found))
    if missing:
        raise SystemExit(f"not found in the page: {', '.join(missing)}")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("page", type=Path)
    ap.add_argument("--check", action="store_true", help="compare only; exit 1 on any difference")
    a = ap.parse_args()
    page = a.page.read_bytes()
    print(f"source {a.page.name}: sha256 {hashlib.sha256(page).hexdigest()}")
    bad = 0
    for name, body in scripts(page).items():
        dest = OUT / name
        same = dest.exists() and dest.read_bytes() == body
        print(f"{name}: {len(body):,} bytes, sha256 {hashlib.sha256(body).hexdigest()}{' (unchanged)' if same else ''}")
        if a.check:
            bad += not same
        elif not same:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
