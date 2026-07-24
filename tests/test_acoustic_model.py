"""Tests for the acoustic modelling module."""

import numpy as np
import pytest
from acoustic_phase_optimizer.acoustic.room_model import RoomModel, RoomDimensions, Surface
from acoustic_phase_optimizer.acoustic.speaker import Speaker, SpeakerType, SpeakerPolarity
from acoustic_phase_optimizer.acoustic.microphone import Microphone
from acoustic_phase_optimizer.acoustic.reflection import ReflectionEngine
from acoustic_phase_optimizer.acoustic.comb_filter import CombFilterDetector


class TestRoomModel:
    def test_default_room(self):
        room = RoomModel()
        assert len(room.surfaces) == 6

    def test_dimensions(self):
        room = RoomModel(RoomDimensions(20, 15, 6))
        L, W, H = room.get_dimensions_array()
        assert L == 20
        assert W == 15
        assert H == 6

    def test_set_dimensions(self):
        room = RoomModel()
        room.set_dimensions(40, 30, 10)
        L, W, H = room.get_dimensions_array()
        assert L == 40
        assert W == 30
        assert H == 10

    def test_add_surface(self):
        room = RoomModel()
        n = len(room.surfaces)
        surface = Surface(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))
        room.add_surface(surface)
        assert len(room.surfaces) == n + 1

    def test_volume(self):
        room = RoomModel(RoomDimensions(10, 10, 5))
        volume = room.get_volume()
        assert volume == 500

    def test_surface_area(self):
        room = RoomModel(RoomDimensions(10, 10, 5))
        area = room.get_surface_area()
        assert area > 0

    def test_average_absorption(self):
        room = RoomModel()
        alpha = room.average_absorption()
        assert 0 < alpha < 1

    def test_rt60_sabine(self):
        room = RoomModel(RoomDimensions(20, 15, 8))
        rt60 = room.estimate_rt60_sabine()
        assert rt60 > 0

    def test_rt60_eyring(self):
        room = RoomModel(RoomDimensions(20, 15, 8))
        rt60 = room.estimate_rt60_eyring()
        assert rt60 > 0

    def test_schroeder_frequency(self):
        room = RoomModel(RoomDimensions(20, 15, 8))
        f = room.schroeder_frequency()
        assert f > 0

    def test_ray_intersect(self):
        room = RoomModel(RoomDimensions(10, 10, 5))
        origin = np.array([0.0, 0.0, 2.5])
        direction = np.array([1.0, 0.0, 0.0])
        hits = room.ray_intersect(origin, direction)
        assert len(hits) >= 0

    def test_surface_normal(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        surface = Surface(verts)
        assert surface.normal is not None
        assert abs(np.linalg.norm(surface.normal) - 1.0) < 1e-10

    def test_triangle_intersect(self):
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([0.0, 0.0, -1.0])
        v0 = np.array([-1.0, -1.0, 0.0])
        v1 = np.array([1.0, -1.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        point, distance = RoomModel._ray_triangle_intersect(origin, direction, v0, v1, v2)
        assert point is not None
        assert distance is not None
        assert abs(distance - 1.0) < 1e-10


class TestSpeaker:
    def test_speaker_creation(self):
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 1.0, 2.0]))
        assert spk.name == "Test"
        assert spk.x == 0.0
        assert spk.y == 1.0
        assert spk.z == 2.0

    def test_distance_to(self):
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 0.0, 0.0]))
        d = spk.distance_to(np.array([3.0, 4.0, 0.0]))
        assert d == 5.0

    def test_delay_from_distance(self):
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 0.0, 0.0]))
        delay = spk.delay_from_distance(np.array([34.3, 0.0, 0.0]), 343.0)
        assert abs(delay - 100.0) < 1e-6

    def test_spl_at_distance(self):
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 0.0, 0.0]))
        spl = spk.spl_at_distance(np.array([1.0, 0.0, 0.0]))
        assert spl == spk.sensitivity_db

    def test_polarity_default(self):
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 0.0, 0.0]))
        assert spk.polarity == SpeakerPolarity.NORMAL

    def test_to_dict(self):
        spk = Speaker("Test", SpeakerType.SUBWOOFER, np.array([0.0, 2.0, 0.0]))
        d = spk.to_dict()
        assert d["name"] == "Test"
        assert d["type"] == "subwoofer"

    def test_from_dict(self):
        data = {
            "name": "Test",
            "type": "left_main",
            "position": [0.0, 1.0, 2.0],
        }
        spk = Speaker.from_dict(data)
        assert spk.name == "Test"
        assert spk.speaker_type == SpeakerType.MAIN_LEFT

    def test_speaker_enabled(self):
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 0.0, 0.0]))
        assert spk.enabled
        spk.enabled = False
        assert not spk.enabled

    def test_list_conversion(self):
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, [1.0, 2.0, 3.0])
        assert isinstance(spk.position, np.ndarray)
        assert spk.x == 1.0


class TestMicrophone:
    def test_mic_creation(self):
        mic = Microphone("Test", np.array([0.0, 1.0, 1.5]))
        assert mic.name == "Test"
        assert mic.x == 0.0
        assert mic.y == 1.0
        assert mic.z == 1.5

    def test_mic_to_dict(self):
        mic = Microphone("Test", [1.0, 2.0, 0.0], zone="foh")
        d = mic.to_dict()
        assert d["name"] == "Test"
        assert d["zone"] == "foh"

    def test_mic_from_dict(self):
        data = {"name": "Test", "position": [0.0, 0.0, 0.0], "zone": "center"}
        mic = Microphone.from_dict(data)
        assert mic.name == "Test"
        assert mic.zone == "center"

    def test_list_conversion(self):
        mic = Microphone("Test", [1.0, 2.0, 3.0])
        assert isinstance(mic.position, np.ndarray)


class TestReflectionEngine:
    def test_image_sources(self):
        room = RoomModel(RoomDimensions(10, 10, 5))
        engine = ReflectionEngine(room, max_order=2)
        sources = engine.compute_image_sources(np.array([0.0, 0.0, 2.0]))
        assert len(sources) > 0
        for src in sources:
            assert src["order"] >= 1

    def test_compute_reflections(self):
        room = RoomModel(RoomDimensions(20, 20, 8))
        engine = ReflectionEngine(room, max_order=2)
        refs = engine.compute_reflections(
            np.array([0.0, 0.0, 2.0]),
            np.array([0.0, 10.0, 1.2]),
        )
        assert len(refs) > 0
        assert refs[0]["type"] == "direct"

    def test_compute_impulse_response(self):
        room = RoomModel(RoomDimensions(10, 10, 5))
        engine = ReflectionEngine(room, max_order=1)
        ir, times = engine.compute_impulse_response(
            np.array([0.0, 0.0, 2.0]),
            np.array([0.0, 5.0, 1.2]),
            sample_rate=1000,
            max_delay_s=1.0,
        )
        assert len(ir) == 1000
        assert len(times) == 1000

    def test_reverb_time_estimate(self):
        room = RoomModel(RoomDimensions(20, 20, 8))
        engine = ReflectionEngine(room, max_order=3)
        rt60 = engine.compute_reverb_time_estimate(
            np.array([0.0, 0.0, 2.0]),
            np.array([0.0, 10.0, 1.2]),
        )
        assert rt60 >= 0


class TestCombFilterDetector:
    def test_detect_comb_filtering(self):
        detector = CombFilterDetector(48000)
        np.random.seed(42)
        freqs = np.linspace(20, 20000, 1000)
        mag = np.sin(freqs / 100) * 6.0
        result = detector.detect_comb_filtering(mag, freqs)
        assert isinstance(result.detected, bool)

    def test_detect_phase_cancellation(self):
        detector = CombFilterDetector(48000)
        ir1 = np.zeros(1024)
        ir1[100] = 1.0
        ir2 = np.zeros(1024)
        ir2[150] = 1.0
        freqs, cancellation, phase = detector.detect_phase_cancellation(ir1, ir2)
        assert len(freqs) > 0
        assert len(cancellation) == len(freqs)
        assert len(phase) == len(freqs)

    def test_map_cancellation_zones(self):
        detector = CombFilterDetector()
        X, Y, Z = detector.map_cancellation_zones(
            np.array([-5.0, 0.0, 2.0]),
            np.array([5.0, 0.0, 2.0]),
            100.0,
            (-10, 10, -10, 10),
            resolution=20,
        )
        assert X.shape == (20, 20)
        assert np.all(np.isfinite(Z))

    def test_smooth_spectrum(self):
        detector = CombFilterDetector()
        mag = np.random.randn(100) * 3
        freqs = np.logspace(1, 4, 100)
        smoothed = detector._smooth_spectrum(mag, freqs)
        assert len(smoothed) == len(mag)
        assert np.all(np.isfinite(smoothed))

    def test_find_peaks(self):
        detector = CombFilterDetector()
        data = np.array([0, 1, 2, 5, 2, 1, 0, 1, 3, 1, 0])
        freqs = np.arange(len(data))
        peaks = detector._find_peaks(data, freqs, min_distance=2)
        assert len(peaks) > 0

    def test_estimate_fundamental_empty(self):
        detector = CombFilterDetector()
        f = detector._estimate_fundamental(np.array([]))
        assert f == 0.0
