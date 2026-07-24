"""Example: Generate measurement signals and analyze.

This example demonstrates:
1. Generating log sweep and MLS test signals
2. Simulating a room measurement
3. Extracting impulse response and acoustic parameters

Usage:
    python examples/run_measurement.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from acoustic_phase_optimizer.measurement.signal_generator import SignalGenerator
from acoustic_phase_optimizer.measurement.impulse_response import ImpulseResponse
from acoustic_phase_optimizer.measurement.room_analysis import RoomAnalysis
from acoustic_phase_optimizer.utils.logging import setup_logging


def main():
    setup_logging(level="INFO")

    print("=" * 60)
    print("Acoustic Phase Optimizer - Measurement Example")
    print("=" * 60)

    gen = SignalGenerator(sample_rate=48000)

    print("\nGenerating log sweep...")
    sweep = gen.log_sweep(duration=2.0, start_freq=20, end_freq=20000)
    print(f"  Length: {len(sweep)} samples ({len(sweep)/48000:.2f}s)")

    inv_sweep = gen.inverse_sweep(duration=2.0, start_freq=20, end_freq=20000)

    print("\nGenerating MLS sequence...")
    mls = gen.mls_sequence(order=10, repetitions=2)
    print(f"  Length: {len(mls)} samples ({len(mls)/48000:.2f}s)")
    print(f"  Unique values: {np.unique(mls)}")

    print("\nSimulating room measurement...")
    delay_samples = 500
    recorded = np.roll(sweep, delay_samples)
    recorded[:delay_samples] = 0
    noise = np.random.normal(0, 0.001, len(recorded))
    recorded += noise

    ir_extractor = ImpulseResponse(sample_rate=48000)

    print("\nExtracting impulse response...")
    ir = ir_extractor.extract_from_sweep(recorded, sweep)
    print(f"  IR length: {len(ir)} samples")
    peak_idx = np.argmax(np.abs(ir))
    print(f"  Peak at sample: {peak_idx} ({peak_idx/48000*1000:.2f}ms)")

    print("\nAnalyzing frequency response...")
    freqs, magnitude_db = ir_extractor.extract_magnitude(ir)
    print(f"  Frequency points: {len(freqs)}")
    print(f"  Magnitude range: {np.min(magnitude_db):.1f} to {np.max(magnitude_db):.1f} dB")

    freqs_p, phase = ir_extractor.extract_phase(ir)
    print(f"  Phase range: {np.min(phase):.2f} to {np.max(phase):.2f} rad")

    print("\nComputing group delay...")
    freqs_gd, group_delay = ir_extractor.compute_group_delay(ir)
    print(f"  Mean group delay: {np.mean(group_delay)*1000:.2f} ms")

    print("\nPerforming room acoustic analysis...")
    analysis = RoomAnalysis(sample_rate=48000)
    results = analysis.analyze_impulse_response(ir)

    print(f"  RT60 (broadband): {results['rt60'].get('rt60_broadband', 0):.2f}s")
    print(f"  EDT: {results['edt']:.2f}s")
    print(f"  Clarity C50: {results['clarity_50']:.1f} dB")
    print(f"  Clarity C80: {results['clarity_80']:.1f} dB")
    print(f"  Definition D50: {results['definition_50']:.2f}")
    print(f"  Center time: {results['center_time']*1000:.1f} ms")

    print("\nDetecting reflections...")
    reflections = analysis.detect_reflections(ir, threshold_db=-30)
    print(f"  Detected reflections: {len(reflections)}")
    for ref in reflections[:5]:
        print(f"    {ref['time_ms']:.1f}ms, {ref['amplitude_db']:.1f}dB")

    print(f"\nNoise floor: {analysis.estimate_noise_floor(ir):.1f} dB")
    print("\nMeasurement analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
