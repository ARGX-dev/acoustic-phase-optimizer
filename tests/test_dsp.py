"""Tests for the DSP module."""

import numpy as np
import pytest
from acoustic_phase_optimizer.dsp.filters import FilterDesign
from acoustic_phase_optimizer.dsp.crossover import CrossoverDesigner
from acoustic_phase_optimizer.dsp.interface import DSPInterface
from acoustic_phase_optimizer.dsp.generic import GenericDSP
from acoustic_phase_optimizer.dsp.venus360 import Venus360


class TestFilterDesign:
    def test_fir_lowpass(self):
        fd = FilterDesign(48000)
        taps = fd.fir_lowpass(1000, num_taps=128)
        assert len(taps) == 128
        assert taps.dtype == np.float64

    def test_fir_highpass(self):
        fd = FilterDesign(48000)
        taps = fd.fir_highpass(1000, num_taps=128)
        assert len(taps) == 129

    def test_fir_bandpass(self):
        fd = FilterDesign(48000)
        taps = fd.fir_bandpass(500, 2000, num_taps=256)
        assert len(taps) == 256

    def test_fir_bandstop(self):
        fd = FilterDesign(48000)
        taps = fd.fir_bandstop(500, 2000, num_taps=256)
        assert len(taps) == 257

    def test_fir_from_target(self):
        fd = FilterDesign(48000)
        freqs = np.linspace(0, 24000, 100)
        target = np.ones(100)
        taps = fd.fir_from_target(target, freqs, num_taps=63)
        assert len(taps) == 63

    def test_iir_parametric_eq(self):
        fd = FilterDesign(48000)
        b, a = fd.iir_parametric_eq(1000, 3.0, 1.0, "peaking")
        assert len(b) == 3
        assert len(a) == 3

    def test_iir_parametric_eq_invalid(self):
        fd = FilterDesign(48000)
        with pytest.raises(ValueError):
            fd.iir_parametric_eq(1000, 3.0, 1.0, "invalid")

    def test_iir_butterworth(self):
        fd = FilterDesign(48000)
        b, a = fd.iir_butterworth(4, 1000, "lowpass")
        assert len(b) > 0
        assert len(a) > 0

    def test_apply_fir(self):
        fd = FilterDesign(48000)
        signal = np.sin(np.linspace(0, 2 * np.pi * 440, 4800))
        taps = fd.fir_lowpass(1000, num_taps=32)
        result = fd.apply_fir(signal, taps)
        assert len(result) == len(signal)

    def test_apply_iir(self):
        fd = FilterDesign(48000)
        signal = np.sin(np.linspace(0, 2 * np.pi * 440, 4800))
        b, a = fd.iir_parametric_eq(1000, 3.0, 1.0, "peaking")
        result = fd.apply_iir(signal, b, a)
        assert len(result) == len(signal)

    def test_frequency_response(self):
        fd = FilterDesign(48000)
        b, a = fd.iir_butterworth(2, 1000, "lowpass")
        freqs, mag, phase = fd.frequency_response(b, a)
        assert len(freqs) == 512

    def test_design_corrective_eq(self):
        fd = FilterDesign(48000)
        freqs = np.logspace(1, 4, 100)
        measured = np.ones(100) * -6
        eq = fd.design_corrective_eq(measured, freqs)
        assert isinstance(eq, list)

    def test_fir_minimum_phase(self):
        fd = FilterDesign(48000)
        taps = np.array([0.5, 1.0, 0.5])
        min_phase = fd.fir_minimum_phase(taps)
        assert len(min_phase) == len(taps)

    def test_fir_frequency_response(self):
        fd = FilterDesign(48000)
        taps = fd.fir_lowpass(1000, num_taps=64)
        freqs, mag, phase = fd.fir_frequency_response(taps)
        assert len(freqs) == 512

    def test_sos_filter(self):
        fd = FilterDesign(48000)
        from scipy import signal
        sos = signal.butter(4, 1000 / 24000, btype="lowpass", output="sos")
        x = np.random.randn(1000)
        y = fd.apply_sos(x, sos)
        assert len(y) == len(x)

    def test_lowshelf_eq(self):
        fd = FilterDesign(48000)
        b, a = fd.iir_parametric_eq(100, 6.0, 0.7, "lowshelf")
        assert len(b) == 3

    def test_highshelf_eq(self):
        fd = FilterDesign(48000)
        b, a = fd.iir_parametric_eq(10000, -6.0, 0.7, "highshelf")
        assert len(b) == 3


class TestCrossoverDesigner:
    def test_linkwitz_riley(self):
        cd = CrossoverDesigner(48000)
        b, a = cd.linkwitz_riley(1000, 4, "lowpass")
        assert len(b) > 0

    def test_butterworth(self):
        cd = CrossoverDesigner(48000)
        b, a = cd.butterworth(1000, 4, "highpass")
        assert len(b) > 0

    def test_bessel(self):
        cd = CrossoverDesigner(48000)
        b, a = cd.bessel(1000, 4, "lowpass")
        assert len(b) > 0

    def test_fir_crossover(self):
        cd = CrossoverDesigner(48000)
        taps = cd.fir_crossover(1000, num_taps=128, filter_type="lowpass")
        assert len(taps) == 128

    def test_design_2way(self):
        cd = CrossoverDesigner(48000)
        result = cd.design_2way(1000, 24, "linkwitz_riley")
        assert result["type"] == "2_way"
        assert "lowpass_b" in result
        assert "highpass_b" in result

    def test_design_3way(self):
        cd = CrossoverDesigner(48000)
        result = cd.design_3way(300, 3000, 24)
        assert result["type"] == "3_way"

    def test_subwoofer_crossover(self):
        cd = CrossoverDesigner(48000)
        result = cd.subwoofer_crossover(80)
        assert result["type"] == "2_way"

    def test_optimize_crossover_freq(self):
        cd = CrossoverDesigner(48000)
        freqs = np.logspace(2, 4, 100)
        woofer = -10 * np.log10(1 + (freqs / 1000) ** 4)
        tweeter = -10 * np.log10(1 + (3000 / freqs) ** 4)
        freq = cd.optimize_crossover_freq(woofer, tweeter, freqs)
        assert 500 <= freq <= 4000


class TestDSPInterface:
    def test_generic_dsp(self):
        dsp = GenericDSP()
        assert dsp.connect()
        assert dsp.is_connected
        assert dsp.set_delay(1, 10.0)
        assert dsp.set_gain(1, -3.0)
        assert dsp.set_polarity(1, True)
        assert dsp.get_delay(1) == 10.0
        assert dsp.get_gain(1) == -3.0
        assert dsp.get_polarity(1) == True
        assert dsp.disconnect()

    def test_generic_dsp_config(self):
        dsp = GenericDSP()
        dsp.connect()
        config = {"1": {"delay_ms": 20.0, "gain_db": -6.0}}
        assert dsp.apply_configuration(config)
        read = dsp.read_configuration()
        assert "1" in read
        assert read["1"]["delay_ms"] == 20.0

    def test_generic_dsp_reset(self):
        dsp = GenericDSP()
        dsp.connect()
        dsp.set_delay(1, 50.0)
        dsp.reset_to_defaults()
        assert dsp.get_delay(1) is None

    def test_venus360(self):
        dsp = Venus360("127.0.0.1", 23)
        assert dsp.connect()
        assert dsp.set_delay(1, 5.0)
        assert dsp.set_gain(2, -12.0)
        assert dsp.set_polarity(1, True)
        assert dsp.get_delay(1) == 5.0
        assert dsp.get_gain(2) == -12.0
        assert dsp.get_polarity(1) == True

    def test_venus360_channel_limits(self):
        dsp = Venus360("127.0.0.1", 23)
        dsp.connect()
        assert not dsp.set_delay(5, 10.0)
        assert not dsp.set_gain(0, 10.0)

    def test_venus360_eq_bands(self):
        dsp = Venus360("127.0.0.1", 23)
        dsp.connect()
        for i in range(10):
            assert dsp.set_eq_parametric(1, 100 * (i + 1), 0.0, 1.0)
        assert not dsp.set_eq_parametric(1, 1000, 0.0, 1.0)

    def test_venus360_config_read(self):
        dsp = Venus360("127.0.0.1", 23)
        dsp.connect()
        dsp.set_delay(1, 15.0)
        dsp.set_gain(2, -3.0)
        config = dsp.read_configuration()
        assert "1" in config
        assert "2" in config
        assert config["1"]["delay_ms"] == 15.0

    def test_interface_factory(self):
        iface = DSPInterface.create({"type": "generic"})
        assert isinstance(iface, GenericDSP)

    def test_interface_factory_venus(self):
        iface = DSPInterface.create({"type": "dbx_venus360"})
        assert isinstance(iface, Venus360)

    def test_interface_factory_unknown(self):
        iface = DSPInterface.create({"type": "unknown"})
        assert isinstance(iface, GenericDSP)

    def test_validate_sample_rate(self):
        iface = GenericDSP()
        assert iface.validate_sample_rate(48000)
        assert iface.validate_sample_rate(96000)
        assert not iface.validate_sample_rate(12345)

    def test_generic_export_yaml(self):
        dsp = GenericDSP()
        dsp.connect()
        dsp.set_delay(1, 10.0)
        yaml_str = dsp.export_as_yaml()
        assert "delay_ms" in yaml_str

    def test_venus360_mute(self):
        dsp = Venus360("127.0.0.1", 23)
        dsp.connect()
        assert dsp.mute_channel(1, True)
        assert not dsp.mute_channel(99, True)

    def test_venus360_crossover(self):
        dsp = Venus360("127.0.0.1", 23)
        dsp.connect()
        assert dsp.set_crossover(1, 1000, 24)
        result = dsp.get_crossover(1)
        assert result is not None
        assert result[0] == 1000
