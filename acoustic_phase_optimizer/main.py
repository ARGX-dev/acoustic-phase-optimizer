"""Main entry point for the Acoustic Phase Optimizer."""

from __future__ import annotations

import argparse
import sys
from typing import Optional
from acoustic_phase_optimizer.config import Config
from acoustic_phase_optimizer.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


class AcousticPhaseOptimizer:
    """Main application orchestrator for the Acoustic Phase Optimizer."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._initialized = False

    def initialize(self) -> bool:
        logger.info("Initializing Acoustic Phase Optimizer...")
        logger.info(f"Sample rate: {self.config.get('system', 'sample_rate')} Hz")
        logger.info(f"Speed of sound: {self.config.get('acoustic', 'speed_of_sound')} m/s")
        self._initialized = True
        return True

    def run_gui(self) -> None:
        if not self._initialized:
            self.initialize()
        logger.info("Launching GUI...")
        try:
            from acoustic_phase_optimizer.visualization.app import launch_gui
            launch_gui(self.config)
        except ImportError as e:
            logger.error(f"Cannot launch GUI: {e}")
            logger.error("Install PyQt6 and pyqtgraph: pip install -e .[gui]")
            sys.exit(1)

    def run_headless(self, config_path: Optional[str] = None) -> dict:
        if not self._initialized:
            self.initialize()

        logger.info("Running in headless mode...")

        from acoustic_phase_optimizer.simulation.virtual_room import VirtualRoom
        from acoustic_phase_optimizer.acoustic.room_model import RoomModel
        from acoustic_phase_optimizer.acoustic.speaker import Speaker, SpeakerType
        from acoustic_phase_optimizer.acoustic.microphone import Microphone
        from acoustic_phase_optimizer.optimization.engine import OptimizationEngine
        from acoustic_phase_optimizer.optimization.objectives import ObjectiveFunction
        import numpy as np

        room = RoomModel()
        speakers = [
            Speaker("Left Main", SpeakerType.MAIN_LEFT, np.array([-8.0, 1.0, 2.0])),
            Speaker("Right Main", SpeakerType.MAIN_RIGHT, np.array([8.0, 1.0, 2.0])),
            Speaker("Sub", SpeakerType.SUBWOOFER, np.array([0.0, 2.5, 0.0])),
        ]
        microphones = [
            Microphone("FOH", np.array([0.0, 12.0, 1.2])),
            Microphone("Left", np.array([-5.0, 8.0, 1.2])),
            Microphone("Right", np.array([5.0, 8.0, 1.2])),
        ]

        virtual_room = VirtualRoom(room)
        for s in speakers:
            virtual_room.add_speaker(s)
        for m in microphones:
            virtual_room.add_microphone(m)

        engine = OptimizationEngine(self.config.data.get("optimization", {}))
        objective = ObjectiveFunction()

        def obj_fn(p: np.ndarray) -> float:
            for i, spk in enumerate(speakers):
                if i * 3 < len(p):
                    spk.delay_ms = float(p[i * 3])
                    spk.gain_db = float(p[i * 3 + 1])
            data = {
                "phase": [],
                "magnitude_db": [],
                "cancellation_zones": [],
                "delays_ms": [s.delay_ms for s in speakers],
                "rt60": {"broadband": room.estimate_rt60_eyring()},
            }
            return objective.compute(p, data)

        algorithm = self.config.get("optimization", "default_algorithm", default="genetic")
        bounds, initial = engine.setup_speaker_optimization(len(speakers))

        if algorithm == "compare_all":
            results = engine.compare_algorithms(obj_fn, initial)
            best_algo, best_result = engine.get_best_result(results)
            result_data = {
                "best_algorithm": best_algo,
                "best_value": best_result.best_value,
                "best_params": best_result.best_params.tolist(),
                "results": {
                    name: {
                        "best_value": r.best_value,
                        "iterations": r.iterations,
                        "time": r.computation_time,
                    }
                    for name, r in results.items()
                },
            }
        else:
            result = engine.optimize(algorithm, obj_fn, initial)
            result_data = {
                "algorithm": algorithm,
                "best_value": result.best_value,
                "best_params": result.best_params.tolist(),
                "iterations": result.iterations,
                "time": result.computation_time,
                "success": result.success,
            }

        logger.info(f"Optimization complete: {result_data}")
        return result_data

    def run_measurement(self, config_path: Optional[str] = None) -> dict:
        if not self._initialized:
            self.initialize()

        logger.info("Running measurement...")

        from acoustic_phase_optimizer.measurement.signal_generator import SignalGenerator
        import numpy as np

        gen = SignalGenerator(self.config.get("system", "sample_rate", default=48000))
        params = self.config.data.get("measurement", {})

        signal = gen.generate(
            sweep_type=params.get("sweep_type", "log"),
            duration=params.get("sweep_duration", 5.0),
            start_freq=params.get("sweep_start_freq", 20.0),
            end_freq=params.get("sweep_end_freq", 20000.0),
            mls_order=params.get("mls_order", 15),
            averaging=params.get("averaging", 3),
        )

        logger.info(f"Generated test signal: {len(signal)} samples")

        from acoustic_phase_optimizer.utils.audio_utils import AudioUtils
        AudioUtils.write_wav("measurement_signal.wav", signal)

        return {
            "signal_length": len(signal),
            "duration": len(signal) / self.config.get("system", "sample_rate", default=48000),
            "output_file": "measurement_signal.wav",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acoustic Phase Optimizer - Live Sound DSP Optimization Platform",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "-g", "--gui",
        action="store_true",
        help="Launch graphical user interface",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run optimization in headless mode",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Run measurement signal generation",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file",
    )

    args = parser.parse_args()

    setup_logging(
        level="DEBUG" if args.verbose else "INFO",
        log_file=args.log_file,
        verbose=args.verbose,
    )

    config = Config(args.config) if args.config else Config()

    app = AcousticPhaseOptimizer(config)
    app.initialize()

    if args.gui:
        app.run_gui()
    elif args.headless:
        result = app.run_headless()
        import json
        print(json.dumps(result, indent=2))
    elif args.measure:
        result = app.run_measurement()
        import json
        print(json.dumps(result, indent=2))
    else:
        app.run_gui()


if __name__ == "__main__":
    main()
