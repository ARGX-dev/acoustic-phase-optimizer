"""Example: Optimize a virtual venue's acoustic parameters.

This example demonstrates:
1. Setting up a virtual room with speakers and microphones
2. Running multiple optimization algorithms
3. Comparing and visualizing results

Usage:
    python examples/optimize_venue.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from acoustic_phase_optimizer.acoustic.room_model import RoomModel, RoomDimensions
from acoustic_phase_optimizer.acoustic.speaker import Speaker, SpeakerType
from acoustic_phase_optimizer.acoustic.microphone import Microphone
from acoustic_phase_optimizer.simulation.virtual_room import VirtualRoom
from acoustic_phase_optimizer.optimization.engine import OptimizationEngine
from acoustic_phase_optimizer.optimization.objectives import ObjectiveFunction
from acoustic_phase_optimizer.utils.logging import setup_logging


def main():
    setup_logging(level="INFO")

    print("=" * 60)
    print("Acoustic Phase Optimizer - Venue Optimization Example")
    print("=" * 60)

    room = RoomModel(RoomDimensions(length=30.0, width=20.0, height=8.0))
    vr = VirtualRoom(room)

    speakers = [
        Speaker("Left Main", SpeakerType.MAIN_LEFT, np.array([-8.0, 1.0, 2.5])),
        Speaker("Right Main", SpeakerType.MAIN_RIGHT, np.array([8.0, 1.0, 2.5])),
        Speaker("Subwoofer", SpeakerType.SUBWOOFER, np.array([0.0, 2.0, 0.0])),
    ]

    microphones = [
        Microphone("FOH Center", np.array([0.0, 15.0, 1.2]), zone="foh"),
        Microphone("FOH Left", np.array([-6.0, 12.0, 1.2]), zone="foh"),
        Microphone("FOH Right", np.array([6.0, 12.0, 1.2]), zone="foh"),
        Microphone("Balcony", np.array([0.0, 22.0, 4.0]), zone="balcony"),
    ]

    for s in speakers:
        vr.add_speaker(s)
    for m in microphones:
        vr.add_microphone(m)

    print(f"\nVenue: {room.dimensions.length:.0f}x{room.dimensions.width:.0f}x{room.dimensions.height:.0f}m")
    print(f"Speakers: {len(speakers)}")
    print(f"Measurement positions: {len(microphones)}")
    print(f"Estimated RT60 (Sabine): {room.estimate_rt60_sabine():.2f}s")
    print(f"Estimated RT60 (Eyring): {room.estimate_rt60_eyring():.2f}s")
    print(f"Schroeder frequency: {room.schroeder_frequency():.0f} Hz")

    objective = ObjectiveFunction()
    engine = OptimizationEngine({
        "max_iterations": 200,
        "population_size": 80,
        "mutation_rate": 0.15,
        "crossover_rate": 0.8,
    })

    def objective_wrapper(params: np.ndarray) -> float:
        for i, spk in enumerate(speakers):
            if i * 3 < len(params):
                spk.delay_ms = float(np.clip(params[i * 3], 0.0, 200.0))
                spk.gain_db = float(np.clip(params[i * 3 + 1], -20.0, 12.0))

        data = {
            "phase": [[0.0]],
            "magnitude_db": [[-20.0]],
            "cancellation_zones": np.array([0.0]),
            "delays_ms": [s.delay_ms for s in speakers],
            "rt60": {"broadband": room.estimate_rt60_eyring()},
        }
        return objective.compute(params, data)

    _, initial_params = engine.setup_speaker_optimization(len(speakers))
    initial_value = objective_wrapper(initial_params)

    print(f"\nInitial objective value: {initial_value:.4f}")
    print("\nRunning optimization algorithms...\n")

    results = engine.compare_algorithms(
        objective_wrapper,
        initial_params,
        algorithms=["genetic", "annealing", "gradient", "bayesian"],
    )

    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)

    best_algo, best_result = engine.get_best_result(results)

    for algo_name, result in results.items():
        improvement = ((result.best_value - initial_value) / abs(initial_value)) * 100
        print(f"\n{algo_name.upper():>12}:")
        print(f"  Best value:     {result.best_value:.4f}")
        print(f"  Improvement:    {improvement:+.1f}%")
        print(f"  Iterations:     {result.iterations}")
        print(f"  Time:           {result.computation_time:.2f}s")
        print(f"  Success:        {result.success}")
        if result.success:
            for i, spk in enumerate(speakers):
                if i * 3 < len(result.best_params):
                    delay = result.best_params[i * 3]
                    gain = result.best_params[i * 3 + 1]
                    print(f"  {spk.name:>15}: delay={delay:.1f}ms, gain={gain:.1f}dB")

    print(f"\n{'='*60}")
    print(f"Best algorithm: {best_algo.upper()}")
    print(f"Best objective value: {best_result.best_value:.4f}")
    print(f"Improvement: {((best_result.best_value - initial_value) / abs(initial_value)) * 100:+.1f}%")
    print(f"{'='*60}")

    print("\nOptimized speaker parameters:")
    for i, spk in enumerate(speakers):
        if i * 3 < len(best_result.best_params):
            delay = best_result.best_params[i * 3]
            gain = best_result.best_params[i * 3 + 1]
            print(f"  {spk.name:>15}: delay={delay:.1f}ms, gain={gain:.1f}dB")


if __name__ == "__main__":
    main()
