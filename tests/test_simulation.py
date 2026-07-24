"""Tests for the simulation module."""

import numpy as np
import pytest
from acoustic_phase_optimizer.simulation.virtual_room import VirtualRoom
from acoustic_phase_optimizer.simulation.virtual_dsp import VirtualDSP
from acoustic_phase_optimizer.acoustic.room_model import RoomModel, RoomDimensions
from acoustic_phase_optimizer.acoustic.speaker import Speaker, SpeakerType
from acoustic_phase_optimizer.acoustic.microphone import Microphone


class TestVirtualRoom:
    def test_virtual_room_creation(self):
        room = RoomModel(RoomDimensions(20, 15, 8))
        vr = VirtualRoom(room)
        assert len(vr.speakers) == 0
        assert len(vr.microphones) == 0

    def test_add_speaker(self):
        vr = VirtualRoom()
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 0.0, 2.0]))
        vr.add_speaker(spk)
        assert len(vr.speakers) == 1

    def test_add_microphone(self):
        vr = VirtualRoom()
        mic = Microphone("Test", np.array([0.0, 10.0, 1.2]))
        vr.add_microphone(mic)
        assert len(vr.microphones) == 1

    def test_compute_transfer_function(self):
        vr = VirtualRoom()
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 0.0, 2.0]))
        mic = Microphone("Test", np.array([0.0, 10.0, 1.2]))
        vr.add_speaker(spk)
        vr.add_microphone(mic)
        freqs, mag_db = vr.compute_transfer_function(spk, mic)
        assert len(freqs) > 0
        assert len(mag_db) == len(freqs)
        assert np.all(np.isfinite(mag_db))

    def test_compute_impulse_response(self):
        vr = VirtualRoom()
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 0.0, 2.0]))
        mic = Microphone("Test", np.array([0.0, 5.0, 1.2]))
        vr.add_speaker(spk)
        vr.add_microphone(mic)
        ir = vr.compute_impulse_response(spk, mic)
        assert len(ir) > 0
        assert np.all(np.isfinite(ir))

    def test_simulate_measurement(self):
        vr = VirtualRoom()
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 0.0, 2.0]))
        mic = Microphone("Test", np.array([0.0, 5.0, 1.2]))
        vr.add_speaker(spk)
        vr.add_microphone(mic)
        signal = np.sin(np.linspace(0, 2 * np.pi * 440, 4800))
        result = vr.simulate_measurement(signal, spk, mic, add_noise=False)
        assert len(result) == len(signal)

    def test_multiple_transfer_functions(self):
        vr = VirtualRoom()
        spk1 = Speaker("L", SpeakerType.MAIN_LEFT, np.array([-5.0, 0.0, 2.0]))
        spk2 = Speaker("R", SpeakerType.MAIN_RIGHT, np.array([5.0, 0.0, 2.0]))
        mic1 = Microphone("M1", np.array([0.0, 5.0, 1.2]))
        mic2 = Microphone("M2", np.array([0.0, 10.0, 1.2]))
        vr.add_speaker(spk1)
        vr.add_speaker(spk2)
        vr.add_microphone(mic1)
        vr.add_microphone(mic2)
        results = vr.compute_multiple_transfer_functions()
        assert "M1" in results
        assert "M2" in results
        assert "L" in results["M1"]
        assert "R" in results["M1"]

    def test_listening_area_mesh(self):
        vr = VirtualRoom()
        X, Y = vr.get_listening_area_mesh(20)
        assert X.shape == (20, 20)
        assert Y.shape == (20, 20)

    def test_transfer_with_reflections(self):
        vr = VirtualRoom()
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 0.0, 2.0]))
        mic = Microphone("Test", np.array([0.0, 3.0, 1.2]))
        vr.add_speaker(spk)
        vr.add_microphone(mic)
        freqs_no_ref, mag_no_ref = vr.compute_transfer_function(spk, mic, include_reflections=False)
        freqs_ref, mag_ref = vr.compute_transfer_function(spk, mic, include_reflections=True)
        assert len(freqs_no_ref) == len(freqs_ref)

    def test_simulate_with_noise(self):
        vr = VirtualRoom()
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([0.0, 0.0, 2.0]))
        mic = Microphone("Test", np.array([0.0, 5.0, 1.2]))
        vr.add_speaker(spk)
        vr.add_microphone(mic)
        signal = np.ones(1000)
        clean = vr.simulate_measurement(signal, spk, mic, add_noise=False)
        noisy = vr.simulate_measurement(signal, spk, mic, add_noise=True, noise_level_db=-20)
        assert not np.array_equal(clean, noisy)


class TestVirtualDSP:
    def test_virtual_dsp_connect(self):
        dsp = VirtualDSP()
        assert dsp.connect()
        assert dsp.is_connected

    def test_virtual_dsp_delay(self):
        dsp = VirtualDSP()
        dsp.connect()
        dsp.set_delay(1, 10.0)
        assert dsp.get_delay(1) == 10.0

    def test_virtual_dsp_gain(self):
        dsp = VirtualDSP()
        dsp.connect()
        dsp.set_gain(1, -6.0)
        assert dsp.get_gain(1) == -6.0

    def test_virtual_dsp_polarity(self):
        dsp = VirtualDSP()
        dsp.connect()
        dsp.set_polarity(1, True)
        assert dsp.get_polarity(1) == True

    def test_virtual_dsp_crossover(self):
        dsp = VirtualDSP()
        dsp.connect()
        dsp.set_crossover(1, 1000, 24)
        result = dsp.get_crossover(1)
        assert result is not None
        assert result[0] == 1000

    def test_virtual_dsp_eq(self):
        dsp = VirtualDSP()
        dsp.connect()
        dsp.set_eq_parametric(1, 1000, 3.0, 1.0)
        config = dsp.read_configuration()
        assert len(config["1"]["eq_filters"]) == 1

    def test_virtual_dsp_fir(self):
        dsp = VirtualDSP()
        dsp.connect()
        taps = np.ones(32) / 32
        dsp.set_fir_coefficients(1, taps)
        config = dsp.read_configuration()
        assert len(config["1"]["fir"]) == 32

    def test_virtual_dsp_process(self):
        dsp = VirtualDSP()
        dsp.connect()
        signal = np.sin(np.linspace(0, 2 * np.pi * 440, 4800))
        dsp.set_gain(1, -6.0)
        processed = dsp.process_signal(1, signal)
        assert len(processed) == len(signal)
        assert np.max(np.abs(processed)) < np.max(np.abs(signal))

    def test_virtual_dsp_mute(self):
        dsp = VirtualDSP()
        dsp.connect()
        signal = np.ones(100)
        dsp.mute_channel(1, True)
        processed = dsp.process_signal(1, signal)
        assert np.all(processed == 0)

    def test_virtual_dsp_delay_processing(self):
        dsp = VirtualDSP(sample_rate=1000)
        dsp.connect()
        signal = np.zeros(100)
        signal[0] = 1.0
        dsp.set_delay(1, 5.0)
        processed = dsp.process_signal(1, signal)
        assert np.argmax(np.abs(processed)) > 0

    def set_sample_rate(self, rate: int) -> None:
        self.sample_rate = rate

    def test_virtual_dsp_reset(self):
        dsp = VirtualDSP()
        dsp.connect()
        dsp.set_delay(1, 50.0)
        dsp.reset_to_defaults()
        assert dsp.get_delay(1) is None

    def test_virtual_dsp_config_apply(self):
        dsp = VirtualDSP()
        dsp.connect()
        config = {"1": {"delay_ms": 25.0, "gain_db": -3.0}}
        dsp.apply_configuration(config)
        assert dsp.get_delay(1) == 25.0
        assert dsp.get_gain(1) == -3.0

    def test_virtual_dsp_polarity_inversion(self):
        dsp = VirtualDSP()
        dsp.connect()
        signal = np.ones(100)
        dsp.set_polarity(1, True)
        processed = dsp.process_signal(1, signal)
        assert np.all(processed < 0)
