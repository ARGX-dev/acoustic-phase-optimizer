"""Preset demo venues for presentations and testing."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from acoustic_phase_optimizer.acoustic.room_model import RoomModel, RoomDimensions
from acoustic_phase_optimizer.acoustic.speaker import Speaker, SpeakerType


def gymnasium() -> Tuple[RoomModel, List[Speaker]]:
    """Large rectangular gym: 30m x 20m x 8m, 4 delay towers + 2 mains."""
    room = RoomModel(RoomDimensions(length=30.0, width=20.0, height=8.0))
    room.speed_of_sound = 343.0

    speakers = [
        Speaker("Main L", SpeakerType.MAIN_LEFT, np.array([-10.0, -7.0, 3.0])),
        Speaker("Main R", SpeakerType.MAIN_RIGHT, np.array([-10.0, 7.0, 3.0])),
        Speaker("Delay L", SpeakerType.DELAY, np.array([4.0, -6.0, 4.0])),
        Speaker("Delay R", SpeakerType.DELAY, np.array([4.0, 6.0, 4.0])),
        Speaker("Sub L", SpeakerType.SUBWOOFER, np.array([-8.0, -4.0, 0.5])),
        Speaker("Sub R", SpeakerType.SUBWOOFER, np.array([-8.0, 4.0, 0.5])),
    ]
    # Set initial bad alignment for dramatic before/after
    speakers[2].delay_ms = 0.0   # Delay L — should be ~35ms
    speakers[3].delay_ms = 0.0   # Delay R
    return room, speakers


def theater() -> Tuple[RoomModel, List[Speaker]]:
    """Narrow theater: 22m x 12m x 6m, LCR + surrounds."""
    room = RoomModel(RoomDimensions(length=22.0, width=12.0, height=6.0))
    room.speed_of_sound = 343.0

    speakers = [
        Speaker("Center", SpeakerType.CENTER, np.array([-8.0, 0.0, 2.5])),
        Speaker("Left", SpeakerType.MAIN_LEFT, np.array([-8.0, -4.0, 2.5])),
        Speaker("Right", SpeakerType.MAIN_RIGHT, np.array([-8.0, 4.0, 2.5])),
        Speaker("Surround L", SpeakerType.DELAY, np.array([5.0, -5.0, 3.0])),
        Speaker("Surround R", SpeakerType.DELAY, np.array([5.0, 5.0, 3.0])),
        Speaker("Fill", SpeakerType.FRONT_FILL, np.array([0.0, 0.0, 1.5])),
    ]
    return room, speakers


def conference_room() -> Tuple[RoomModel, List[Speaker]]:
    """Flat conference room: 12m x 8m x 3m, simple stereo."""
    room = RoomModel(RoomDimensions(length=12.0, width=8.0, height=3.0))
    room.speed_of_sound = 343.0

    speakers = [
        Speaker("Left", SpeakerType.MAIN_LEFT, np.array([-4.0, -2.5, 1.8])),
        Speaker("Right", SpeakerType.MAIN_RIGHT, np.array([-4.0, 2.5, 1.8])),
        Speaker("Center", SpeakerType.CENTER, np.array([1.0, 0.0, 1.2])),
    ]
    return room, speakers
