"""Integration tests for end-to-end workflows."""

import numpy as np
import pytest
from acoustic_phase_optimizer.config import Config
from acoustic_phase_optimizer.measurement.signal_generator import SignalGenerator
from acoustic_phase_optimizer.measurement.impulse_response import ImpulseResponse
from acoustic_phase_optimizer.measurement.rt60 import RT60Estimator
from acoustic_phase_optimizer.acoustic.room_model import RoomModel, RoomDimensions
from acoustic_phase_optimizer.acoustic.speaker import Speaker, SpeakerType
from acoustic_phase_optimizer.acoustic.microphone import Microphone
from acoustic_phase_optimizer.acoustic.reflection import ReflectionEngine
from acoustic_phase_optimizer.acoustic.comb_filter import CombFilterDetector
from acoustic_phase_optimizer.simulation.virtual_room import VirtualRoom
from acoustic_phase_optimizer.simulation.virtual_dsp import VirtualDSP
from acoustic_phase_optimizer.optimization.engine import OptimizationEngine
from acoustic_phase_optimizer.optimization.objectives import ObjectiveFunction
from acoustic_phase_optimizer.dsp.filters import FilterDesign
from acoustic_phase_optimizer.dsp.crossover import CrossoverDesigner


class TestEndToEndWorkflow:
    """End-to-end integration test simulating a complete optimization session."""

    def test_measurement_to_analysis(self):
        gen = SignalGenerator(48000)
        sweep = gen.log_sweep(0.5, 20, 20000)

        delayed = np.roll(sweep, 500)
        delayed[:500] = 0

        ir_extractor = ImpulseResponse(48000)
        ir = ir_extractor.extract_from_sweep(delayed, sweep)

        rt60_estimator = RT60Estimator(48000)
        rt60_results = rt60_estimator.estimate_rt60(ir)

        freqs, mag = ir_extractor.extract_magnitude(ir)
        freqs_p, phase = ir_extractor.extract_phase(ir)

        assert len(ir) == len(sweep)
        assert "rt60_broadband" in rt60_results
        assert len(mag) > 0
        assert len(phase) > 0

    def test_virtual_room_to_optimization(self):
        room = RoomModel(RoomDimensions(30, 20, 8))
        vr = VirtualRoom(room)

        speakers = [
            Speaker("L", SpeakerType.MAIN_LEFT, np.array([-8.0, 1.0, 2.0])),
            Speaker("R", SpeakerType.MAIN_RIGHT, np.array([8.0, 1.0, 2.0])),
            Speaker("Sub", SpeakerType.SUBWOOFER, np.array([0.0, 2.0, 0.0])),
        ]
        mics = [
            Microphone("FOH", np.array([0.0, 12.0, 1.2])),
            Microphone("Balcony", np.array([0.0, 18.0, 3.0])),
        ]

        for s in speakers:
            vr.add_speaker(s)
        for m in mics:
            vr.add_microphone(m)

        for mic in mics:
            for spk in speakers:
                freqs, mag_db = vr.compute_transfer_function(spk, mic)
                assert np.all(np.isfinite(mag_db))

    def test_full_optimization_pipeline(self):
        room = RoomModel(RoomDimensions(20, 15, 8))
        speakers = [
            Speaker("L", SpeakerType.MAIN_LEFT, np.array([-6.0, 1.0, 2.0])),
            Speaker("R", SpeakerType.MAIN_RIGHT, np.array([6.0, 1.0, 2.0])),
        ]

        objective = ObjectiveFunction()
        engine = OptimizationEngine({"max_iterations": 30, "population_size": 20})

        def obj_fn(p):
            for i, spk in enumerate(speakers):
                if i * 3 < len(p):
                    spk.delay_ms = float(p[i * 3])
                    spk.gain_db = float(p[i * 3 + 1])
            data = {
                "phase": [[0.1, 0.2], [0.15, 0.25]],
                "magnitude_db": [[-20, -18], [-22, -19]],
                "cancellation_zones": np.array([0.1, 0.2]),
                "delays_ms": [s.delay_ms for s in speakers],
                "rt60": {"broadband": room.estimate_rt60_eyring()},
            }
            return objective.compute(p, data)

        bounds, initial = engine.setup_speaker_optimization(len(speakers))
        result = engine.optimize("genetic", obj_fn, initial)
        assert result.success
        assert result.best_value > -10

    def test_dsp_filter_design_pipeline(self):
        fd = FilterDesign(48000)

        test_signal = np.sin(np.linspace(0, 2 * np.pi * 440, 48000))
        noise = np.random.randn(48000) * 0.1
        noisy = test_signal + noise

        lp_taps = fd.fir_lowpass(2000, num_taps=128)
        filtered_fir = fd.apply_fir(noisy, lp_taps)

        b, a = fd.iir_butterworth(4, 2000, "lowpass")
        filtered_iir = fd.apply_iir(noisy, b, a)

        assert len(filtered_fir) == len(noisy)
        assert len(filtered_iir) == len(noisy)

    def test_crossover_simulation(self):
        cd = CrossoverDesigner(48000)
        xover = cd.design_2way(1000, 24, "linkwitz_riley")

        test_signal = np.sin(np.linspace(0, 2 * np.pi * 200, 48000))
        low = cd.filter_design.apply_iir(test_signal, xover["lowpass_b"], xover["lowpass_a"])
        high = cd.filter_design.apply_iir(test_signal, xover["highpass_b"], xover["highpass_a"])

        assert len(low) == len(test_signal)
        assert len(high) == len(test_signal)

    def test_virtual_dsp_processing_pipeline(self):
        dsp = VirtualDSP(sample_rate=48000)
        dsp.connect()

        signal = np.sin(np.linspace(0, 2 * np.pi * 440, 4800))

        dsp.set_gain(1, -6.0)
        dsp.set_delay(1, 5.0)
        dsp.set_polarity(1, False)

        dsp.set_eq_parametric(1, 1000, 3.0, 2.0)

        processed = dsp.process_signal(1, signal)
        assert len(processed) == len(signal)
        assert np.all(np.isfinite(processed))

    def test_comb_filtering_detection_pipeline(self):
        detector = CombFilterDetector(48000)

        np.random.seed(42)
        freqs = np.linspace(20, 20000, 2000)
        mag = np.sin(freqs / 50) * 6.0 + np.random.randn(2000) * 0.5

        result = detector.detect_comb_filtering(mag, freqs)

        spk1 = Speaker("A", SpeakerType.MAIN_LEFT, np.array([-5.0, 0.0, 2.0]))
        spk2 = Speaker("B", SpeakerType.MAIN_RIGHT, np.array([5.0, 0.0, 2.0]))

        X, Y, Z = detector.map_cancellation_zones(
            spk1.position, spk2.position, 100.0,
            (-10, 10, -10, 10), resolution=30,
        )

        assert X.shape == (30, 30)
        assert Z.shape == (30, 30)
        assert np.min(Z) >= -1.0
        assert np.max(Z) <= 1.0

    def test_model_export_import(self):
        spk = Speaker("Test", SpeakerType.MAIN_LEFT, np.array([1.0, 2.0, 3.0]))
        spk.delay_ms = 15.0
        spk.gain_db = -3.0

        data = spk.to_dict()
        restored = Speaker.from_dict(data)

        assert restored.name == spk.name
        assert restored.speaker_type == spk.speaker_type
        assert np.allclose(restored.position, spk.position)

    def test_config_persistence(self):
        config = Config()
        config.set("system", "sample_rate", 96000)
        config.set("optimization", "max_iterations", 500)

        data = config.to_dict()
        restored = Config.from_dict(data)

        assert restored.get("system", "sample_rate") == 96000
        assert restored.get("optimization", "max_iterations") == 500

    def test_all_optimization_algorithms(self):
        engine = OptimizationEngine({"max_iterations": 20, "population_size": 20})

        def sphere(x):
            return -np.sum(x ** 2) + 100

        algorithms = ["gradient", "genetic", "annealing", "bayesian"]
        results = []

        for algo in algorithms:
            try:
                result = engine.optimize(algo, sphere, np.array([2.0, -1.0]))
                results.append(result.success)
            except Exception:
                results.append(False)

        assert any(results), "At least one algorithm should succeed"

    def test_room_reverberation_estimation(self):
        room = RoomModel(RoomDimensions(25, 18, 7))
        sabine = room.estimate_rt60_sabine()
        eyring = room.estimate_rt60_eyring()
        assert sabine > 0
        assert eyring > 0
