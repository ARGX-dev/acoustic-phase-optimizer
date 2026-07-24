"""Tests for the measurement module."""

import numpy as np
import pytest
from acoustic_phase_optimizer.measurement.signal_generator import SignalGenerator
from acoustic_phase_optimizer.measurement.impulse_response import ImpulseResponse
from acoustic_phase_optimizer.measurement.rt60 import RT60Estimator
from acoustic_phase_optimizer.measurement.room_analysis import RoomAnalysis


class TestSignalGenerator:
    def test_log_sweep_shape(self):
        gen = SignalGenerator(48000)
        signal = gen.log_sweep(1.0, 20, 20000)
        assert len(signal) == 48000
        assert signal.dtype == np.float64

    def test_log_sweep_values(self):
        gen = SignalGenerator(48000)
        signal = gen.log_sweep(0.1, 20, 20000, fade_samples=10)
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) <= 1.0

    def test_inverse_sweep_shape(self):
        gen = SignalGenerator(48000)
        signal = gen.inverse_sweep(1.0, 20, 20000)
        assert len(signal) == 48000

    def test_mls_sequence(self):
        gen = SignalGenerator(48000)
        seq = gen.mls_sequence(order=5, repetitions=1)
        assert len(seq) == 31
        assert set(np.unique(seq)) == {-1.0, 1.0}

    def test_mls_repetitions(self):
        gen = SignalGenerator(48000)
        seq = gen.mls_sequence(order=5, repetitions=3)
        assert len(seq) == 93

    def test_mls_invalid_order(self):
        gen = SignalGenerator(48000)
        with pytest.raises(ValueError):
            gen.mls_sequence(order=1)

    def test_generate_log(self):
        gen = SignalGenerator(48000)
        signal = gen.generate("log", duration=0.5)
        assert len(signal) == 24000

    def test_generate_mls(self):
        gen = SignalGenerator(48000)
        signal = gen.generate("mls", mls_order=7)
        assert len(signal) == 254

    def test_generate_invalid(self):
        gen = SignalGenerator(48000)
        with pytest.raises(ValueError):
            gen.generate("invalid")

    def test_test_signal(self):
        gen = SignalGenerator(48000)
        signal = gen.generate_test_signal(1000, 0.1)
        assert len(signal) == 4800

    def test_mls_taps_all_orders(self):
        gen = SignalGenerator(48000)
        for order in range(2, 21):
            seq = gen.mls_sequence(order=order, repetitions=1)
            assert len(seq) == 2 ** order - 1

    def test_fade_applied(self):
        gen = SignalGenerator(48000)
        signal = gen.log_sweep(1.0, 20, 20000, fade_samples=100)
        assert signal[0] < 0.01
        assert signal[-1] < 0.01


class TestImpulseResponse:
    def test_extract_from_sweep_shape(self):
        ir = ImpulseResponse(48000)
        sweep = np.sin(np.linspace(0, 2 * np.pi * 1000, 48000))
        recorded = np.roll(sweep, 100)
        result = ir.extract_from_sweep(recorded, sweep)
        assert len(result) == 48000
        assert result.dtype == np.float64

    def test_extract_phase(self):
        ir = ImpulseResponse(48000)
        test_ir = np.zeros(1023)
        test_ir[100] = 1.0
        freqs, phase = ir.extract_phase(test_ir)
        n_expected = len(test_ir) // 2 + 1
        assert len(freqs) == n_expected
        assert len(phase) == n_expected

    def test_extract_magnitude(self):
        ir = ImpulseResponse(48000)
        test_ir = np.zeros(1023)
        test_ir[100] = 1.0
        freqs, mag = ir.extract_magnitude(test_ir)
        n_expected = len(test_ir) // 2 + 1
        assert len(freqs) == n_expected
        assert len(mag) == n_expected

    def test_group_delay(self):
        ir = ImpulseResponse(48000)
        test_ir = np.zeros(2047)
        test_ir[500] = 1.0
        freqs, gd = ir.compute_group_delay(test_ir)
        n_expected = len(test_ir) // 2 + 1
        assert len(freqs) == n_expected
        assert len(gd) == n_expected

    def test_window_ir(self):
        ir = ImpulseResponse(48000)
        test_ir = np.ones(1024)
        windowed = ir.window_ir(test_ir, 100, 500)
        assert windowed[50] == 0.0
        assert windowed[300] > 0.0
        assert windowed[600] == 0.0

    def test_minimum_phase(self):
        ir = ImpulseResponse(48000)
        test_ir = np.zeros(256)
        test_ir[10] = 1.0
        min_phase = ir.minimum_phase(test_ir)
        assert len(min_phase) == 256
        assert np.all(np.isfinite(min_phase))

    def test_extract_from_mls(self):
        ir = ImpulseResponse(48000)
        mls = np.array([1, -1, 1, 1, -1, -1, 1, -1], dtype=np.float64)
        recorded = np.roll(mls, 2)
        result = ir.extract_from_mls(recorded, mls)
        assert len(result) == len(mls)

    def test_deconvolution_identity(self):
        ir = ImpulseResponse(48000)
        sweep = np.sin(np.linspace(0, 2 * np.pi * 500, 4800))
        result = ir.extract_from_sweep(sweep, sweep)
        assert np.argmax(np.abs(result)) < 10


class TestRT60Estimator:
    def test_estimate_rt60(self):
        rt60 = RT60Estimator(48000)
        ir = np.zeros(48000)
        decay = np.exp(-np.arange(48000) / 4800.0)
        ir[:len(decay)] = decay
        results = rt60.estimate_rt60(ir)
        assert "rt60_broadband" in results
        assert results["rt60_broadband"] >= 0

    def test_edt(self):
        rt60 = RT60Estimator(48000)
        ir = np.exp(-np.arange(48000) / 2400.0)
        edt = rt60.estimate_edt(ir)
        assert edt >= 0

    def test_clarity(self):
        rt60 = RT60Estimator(48000)
        ir = np.zeros(48000)
        ir[0] = 1.0
        ir[5000] = 0.5
        clarity = rt60.clarity(ir)
        assert "c50" in clarity
        assert "c80" in clarity

    def test_definition(self):
        rt60 = RT60Estimator(48000)
        ir = np.zeros(48000)
        ir[0] = 1.0
        definition = rt60.definition(ir)
        assert "d50" in definition
        assert 0 <= definition["d50"] <= 1

    def test_center_time(self):
        rt60 = RT60Estimator(48000)
        ir = np.zeros(48000)
        ir[100] = 1.0
        ts = rt60.center_time(ir)
        assert ts > 0

    def test_bandpass_filter(self):
        rt60 = RT60Estimator(48000)
        ir = np.sin(np.linspace(0, 2 * np.pi * 1000, 48000))
        filtered = rt60._bandpass_filter(ir, 1000)
        assert len(filtered) == len(ir)
        assert np.all(np.isfinite(filtered))

    def test_schroeder_rt60(self):
        rt60 = RT60Estimator(48000)
        ir = np.exp(-np.arange(4800) / 2400.0)
        result = rt60._schroeder_rt60(ir)
        assert result >= 0


class TestRoomAnalysis:
    def test_analyze_impulse_response(self):
        analysis = RoomAnalysis(48000)
        ir = np.zeros(48000)
        ir[100] = 1.0
        result = analysis.analyze_impulse_response(ir)
        assert "magnitude_db" in result
        assert "phase" in result
        assert "rt60" in result
        assert "clarity_50" in result

    def test_detect_reflections(self):
        analysis = RoomAnalysis(48000)
        ir = np.zeros(48000)
        ir[100] = 1.0
        ir[500] = 0.5
        ir[1000] = 0.3
        reflections = analysis.detect_reflections(ir, threshold_db=-30)
        assert len(reflections) > 0
        for ref in reflections:
            assert "sample" in ref
            assert "time_ms" in ref
            assert "amplitude" in ref

    def test_energy_decay_curve(self):
        analysis = RoomAnalysis(48000)
        ir = np.exp(-np.arange(4800) / 2400.0)
        decay = analysis.compute_energy_decay_curve(ir)
        assert len(decay) == len(ir)
        assert decay[0] == 0.0

    def test_noise_floor(self):
        analysis = RoomAnalysis(48000)
        ir = np.random.normal(0, 0.001, 48000)
        nf = analysis.estimate_noise_floor(ir)
        assert nf < 0

    def test_empty_ir(self):
        analysis = RoomAnalysis(48000)
        ir = np.zeros(48000)
        result = analysis.analyze_impulse_response(ir)
        assert np.all(np.isfinite(result["magnitude_db"]))
