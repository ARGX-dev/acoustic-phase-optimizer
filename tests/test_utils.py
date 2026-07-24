"""Tests for the utility modules."""

import numpy as np
import pytest
import tempfile
import os
from acoustic_phase_optimizer.utils.math_utils import MathUtils
from acoustic_phase_optimizer.utils.audio_utils import AudioUtils
from acoustic_phase_optimizer.utils.logging import setup_logging, get_logger


class TestMathUtils:
    def test_db_linear_conversion(self):
        db = np.array([0.0, -6.0, -20.0])
        linear = MathUtils.db_to_linear(db)
        db_back = MathUtils.linear_to_db(linear)
        assert np.allclose(db, db_back, atol=0.1)

    def test_magnitude_to_db(self):
        mag = np.array([1.0, 0.5, 0.1])
        db = MathUtils.magnitude_to_db(mag)
        assert db[0] == 0.0
        assert db[1] < 0
        assert db[2] < db[1]

    def test_phase_unwrap(self):
        phase = np.array([0.0, np.pi, 2 * np.pi, 3 * np.pi])
        unwrapped = MathUtils.phase_unwrap(phase)
        assert np.all(np.diff(unwrapped) >= 0)

    def test_phase_to_samples(self):
        phase = np.array([-np.pi, 0.0, np.pi])
        freqs = np.array([100.0, 100.0, 100.0])
        samples = MathUtils.phase_to_samples(phase, freqs)
        assert len(samples) == 3

    def test_group_delay(self):
        freq_response = np.exp(-1j * np.linspace(0, np.pi, 100))
        freqs = np.linspace(0, 24000, 100)
        gd = MathUtils.group_delay(freq_response, freqs)
        assert len(gd) == 100

    def test_next_power_of_two(self):
        assert MathUtils.next_power_of_two(3) == 4
        assert MathUtils.next_power_of_two(4) == 4
        assert MathUtils.next_power_of_two(5) == 8

    def test_pad_to_power_of_two(self):
        data = np.ones(100)
        padded = MathUtils.pad_to_power_of_two(data)
        assert len(padded) == 128

    def test_smooth_frequency_response(self):
        mag = np.random.randn(200) * 5
        freqs = np.logspace(1, 4, 200)
        smoothed = MathUtils.smooth_frequency_response(mag, freqs)
        assert len(smoothed) == len(mag)

    def test_interpolate_ir(self):
        ir = np.ones(100)
        resampled = MathUtils.interpolate_ir(ir, 100, 200)
        assert len(resampled) == 200

    def test_cosine_window(self):
        w = MathUtils.cosine_window(100)
        assert len(w) == 100
        assert np.allclose(w[0], 0.0, atol=0.01)

    def test_apply_window(self):
        data = np.ones(100)
        windowed = MathUtils.apply_window(data, "hann")
        assert windowed[0] < 0.1
        assert abs(windowed[49] - 1.0) < 0.01

    def test_apply_window_none(self):
        data = np.ones(100)
        result = MathUtils.apply_window(data, None)
        assert np.all(result == 1.0)

    def test_apply_window_invalid(self):
        with pytest.raises(ValueError):
            MathUtils.apply_window(np.ones(10), "invalid")

    def test_find_nearest_frequency(self):
        freqs = np.array([100, 200, 300, 400, 500])
        idx, freq = MathUtils.find_nearest_frequency(freqs, 350)
        assert freq == 300

    def test_compute_coherence(self):
        np.random.seed(42)
        x = np.random.randn(10000)
        y = np.roll(x, 10)
        f, cxy = MathUtils.compute_coherence(x, y, nperseg=1024, fs=1000)
        assert len(f) > 0
        assert len(cxy) == len(f)

    def test_ms_to_samples(self):
        assert MathUtils.ms_to_samples(10, 48000) == 480

    def test_samples_to_ms(self):
        assert abs(MathUtils.samples_to_ms(480, 48000) - 10.0) < 0.01

    def test_freq_to_bark(self):
        bark = MathUtils.freq_to_bark(np.array([1000]))
        assert 8 < bark[0] < 9

    def test_freq_to_mel(self):
        mel = MathUtils.freq_to_mel(np.array([1000]))
        assert mel[0] > 0


class TestAudioUtils:
    def test_normalize_signal(self):
        signal = np.array([0.5, -1.0, 0.3])
        normalized = AudioUtils.normalize_signal(signal, 0.95)
        assert np.max(np.abs(normalized)) == 0.95

    def test_apply_fade(self):
        signal = np.ones(1000)
        faded = AudioUtils.apply_fade(signal, fade_samples=100)
        assert faded[0] < 0.01
        assert faded[-1] < 0.01
        assert faded[500] == 1.0

    def test_compute_rms(self):
        signal = np.ones(1000) * 0.5
        rms = AudioUtils.compute_rms(signal)
        assert abs(rms - 0.5) < 0.01

    def test_compute_peak(self):
        signal = np.array([0.1, -0.5, 0.8, -0.2])
        peak = AudioUtils.compute_peak(signal)
        assert peak == 0.8

    def test_silence_detection(self):
        signal = np.zeros(100)
        assert AudioUtils.silence_detection(signal)
        signal_loud = np.ones(100) * 0.5
        assert not AudioUtils.silence_detection(signal_loud)

    def test_read_write_wav(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            data = np.sin(np.linspace(0, 2 * np.pi * 440, 4800)).reshape(-1, 1)
            AudioUtils.write_wav(path, data, 48000)
            read_data, sr = AudioUtils.read_wav(path)
            assert sr == 48000
            assert len(read_data) == len(data)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_list_devices(self):
        devices = AudioUtils.list_devices()
        assert isinstance(devices, list)

    def test_list_input_devices(self):
        devices = AudioUtils.list_input_devices()
        assert isinstance(devices, list)


class TestLogging:
    def test_setup_logging(self):
        setup_logging(level="DEBUG", verbose=True)
        logger = get_logger("test")
        assert logger is not None

    def test_get_logger(self):
        logger = get_logger("test_logger")
        assert logger.name == "acoustic_phase_optimizer.test_logger"
