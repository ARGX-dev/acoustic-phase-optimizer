from __future__ import annotations

import time
import threading
from typing import Optional

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LevelMeter(QWidget):
    def __init__(self, parent=None, label: str = ""):
        super().__init__(parent)
        self._level = -120.0
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 0, 2, 0)
        self.label = QLabel(label)
        self.label.setFixedWidth(60)
        self.label.setStyleSheet("color: #8b93a1; font-size: 10px;")
        layout.addWidget(self.label)
        self.bar = QLabel()
        self.bar.setFixedHeight(12)
        self.bar.setStyleSheet(self._bar_style(-120.0))
        layout.addWidget(self.bar, 1)
        self.value_label = QLabel("-inf dB")
        self.value_label.setFixedWidth(70)
        self.value_label.setStyleSheet("color: #e7ebf0; font-size: 10px; font-family: monospace;")
        layout.addWidget(self.value_label)
        self.setLayout(layout)

    def set_level(self, level_db: float) -> None:
        self._level = level_db
        self.bar.setStyleSheet(self._bar_style(level_db))
        self.value_label.setText(f"{level_db:.1f} dB")

    def _bar_style(self, db: float) -> str:
        pct = max(0, min(100, (db + 60) / 70 * 100))
        if db > -6:
            color = "#ff5462"
        elif db > -20:
            color = "#ffb020"
        else:
            color = "#34d6c0"
        return (
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 {color}, stop:{pct/100:.3f} {color}, "
            f"stop:{pct/100:.3f} #1e1e1e, stop:1 #1e1e1e); "
            f"border: 1px solid #262e39; border-radius: 3px;"
        )


class LiveFFTMonitor(FigureCanvas):
    def __init__(self, parent=None, width: int = 6, height: int = 4, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(20, 20000)
        self.ax.set_ylim(-80, 10)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.set_title("Live FFT Monitor")
        self.ax.grid(True, alpha=0.3, which="both")
        self.fig.tight_layout()

        self._line, = self.ax.semilogx([20, 20000], [-80, -80], linewidth=1.0, color="#34d6c0")
        self._hold_line, = self.ax.semilogx([20, 20000], [-80, -80], linewidth=0.6, color="#ffb020", alpha=0.6)
        self._peak_hold = np.zeros(500)
        self._spectrum = np.zeros(500)

    def update_spectrum(self, freqs: np.ndarray, magnitude_db: np.ndarray) -> None:
        self._spectrum = magnitude_db
        self._peak_hold = np.maximum(self._peak_hold, magnitude_db - 3)
        valid = (freqs >= 20) & np.isfinite(magnitude_db)
        if not np.any(valid):
            return
        self._line.set_data(freqs[valid], magnitude_db[valid])
        self._hold_line.set_data(freqs[valid], self._peak_hold[:len(valid)])
        self.ax.relim()
        self.ax.autoscale_view(scaley=False)
        self.draw_idle()

    def reset_hold(self) -> None:
        self._peak_hold = np.zeros_like(self._peak_hold)

    def clear_spectrum(self) -> None:
        self._spectrum = np.zeros(500)
        self._line.set_data([20, 20000], [-80, -80])
        self._hold_line.set_data([20, 20000], [-80, -80])
        self.draw_idle()


class FFTAcquirer(QObject):
    spectrum_ready = pyqtSignal(object, object)
    level_ready = pyqtSignal(float, float)
    finished = pyqtSignal()

    def __init__(self, sample_rate: int = 48000, device: int | None = None, channels: int = 1):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        self.channels = channels
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _acquire_loop(self) -> None:
        import sounddevice as sd

        block_size = 2048

        def callback(indata, frames, time_info, status):
            if not self._running:
                raise sd.CallbackStop
            if indata.ndim > 1:
                channel_data = indata[:, 0]
            else:
                channel_data = indata

            n_fft = len(channel_data)
            spec = np.fft.rfft(channel_data * np.hanning(n_fft))
            mag = np.abs(spec)
            mag_db = 20.0 * np.log10(mag + 1e-12)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

            peak = float(np.max(np.abs(channel_data)))
            rms = float(np.sqrt(np.mean(channel_data ** 2)))
            peak_db = 20.0 * np.log10(peak + 1e-12)
            rms_db = 20.0 * np.log10(rms + 1e-12)

            self.spectrum_ready.emit(freqs, mag_db)
            self.level_ready.emit(peak_db, rms_db)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                device=self.device,
                channels=self.channels,
                blocksize=block_size,
                callback=callback,
            ):
                while self._running:
                    time.sleep(0.1)
        except Exception as e:
            pass
        finally:
            self.finished.emit()
