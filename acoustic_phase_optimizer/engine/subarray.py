"""Subwoofer array builder: the standard ways of steering low frequencies.

- line:     boxes in a row across the room. With a spread angle the boxes are delayed as if bent onto an
            arc (an "arc delay"), which widens the otherwise narrow centre beam of a long sub line.
- cardioid: gradient pairs. A second box sits `depth` behind each front box, polarity inverted and
            delayed by depth / c, so the two cancel behind the array and still add in front.
- endfire:  boxes in a column pointing into the audience, each delayed by its distance from the rearmost
            one so they arrive together in front; behind, the arrivals are spread out and partly cancel.

Boxes share an `array_group`, so the optimizer moves them as one unit, and keep the pattern itself in
`array_delay_ms` / `array_invert`, which nothing else changes.
"""

from __future__ import annotations

import math
import secrets
from typing import Literal

from .air import speed_of_sound
from .model import Project, Speaker
from .presets import make_speaker

Kind = Literal["line", "cardioid", "endfire"]


def _ms(metres: float, c: float) -> float:
    return round(metres / c * 1000.0, 3)



def _both(m: float) -> str:
    """A length for a message read in either unit system."""
    return f"{m:.1f} m ({m / 0.3048:.1f} ft)"

def build(project: Project, kind: Kind, model: str, count: int, spacing: float, x: float, y: float,
          depth: float = 1.0, spread_deg: float = 0.0, height: float = 0.4) -> list[Speaker]:
    """New speakers for one sub array. `x`, `y` place the array centre (front row for cardioid/end-fire)."""
    env, room = project.environment, project.room
    c = speed_of_sound(env.temperature_c, env.humidity_pct, env.pressure_kpa)
    group = "arr_" + secrets.token_hex(4)
    count = max(1, min(int(count), 24))
    spacing = max(0.3, float(spacing))
    base_color = 4  # subs share one colour slot; their names tell them apart
    out: list[Speaker] = []

    def box(name: str, bx: float, by: float, delay: float = 0.0, invert: bool = False) -> Speaker:
        # The pattern delays are computed from these exact positions, so a box that does not fit is refused
        # rather than moved (a moved box would keep a delay for a spacing that is no longer there).
        if not 0.1 - 1e-6 <= by <= room.length - 0.1 + 1e-6:
            need = 0.1 - by if by < 0.1 else by - (room.length - 0.1)
            raise ValueError(f"{name} would be {_both(abs(need))} outside the room front-to-back. "
                             f"Move the array {'further from' if by < 0.1 else 'closer to'} the stage or use fewer boxes.")
        if not 0.1 - 1e-6 <= bx <= room.width - 0.1 + 1e-6:
            raise ValueError(f"{name} would be outside the room across. Use a smaller spacing or fewer boxes "
                             f"(this array is {_both((count - 1) * spacing)} wide; the room is {_both(room.width)}).")
        s = make_speaker(model, name, round(bx, 2), round(by, 2), height, yaw=0, pitch=0, role="sub")
        s.array_group, s.array_delay_ms, s.array_invert, s.color = group, round(delay, 3), invert, base_color
        return s

    if kind == "line":
        xs = [x + (i - (count - 1) / 2) * spacing for i in range(count)]
        half_width = (count - 1) * spacing / 2
        if spread_deg > 0 and count > 1:
            # Delay each box by how far behind the line it would sit on an arc whose ends span `spread_deg`.
            r = half_width / math.sin(math.radians(min(spread_deg, 170.0)) / 2)
            delays = [_ms(r - math.sqrt(max(r * r - (bx - x) ** 2, 0.0)), c) for bx in xs]
        else:
            delays = [0.0] * count
        for i, (bx, d) in enumerate(zip(xs, delays)):
            out.append(box(f"Sub {i + 1}", bx, y, d))
    elif kind == "cardioid":
        depth = max(0.3, float(depth))
        xs = [x + (i - (count - 1) / 2) * spacing for i in range(count)]
        for i, bx in enumerate(xs):
            out.append(box(f"Sub {i + 1}", bx, y))
            out.append(box(f"Sub {i + 1} rear", bx, y - depth, _ms(depth, c), True))
    elif kind == "endfire":
        rear = y - (count - 1) * spacing
        for i in range(count):
            by = rear + i * spacing  # rearmost first
            out.append(box(f"Sub {i + 1}", x, by, _ms(by - rear, c)))
    else:
        raise ValueError(f"unknown array kind {kind!r}")
    return out
