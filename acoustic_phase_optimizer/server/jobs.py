"""Background jobs with progress and cancellation (used for the deep optimizer)."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable


class Job:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.state = "running"  # running | done | cancelled | error
        self.progress: dict = {}
        self.result: dict | None = None
        self.error: str | None = None
        self.started = time.time()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def view(self) -> dict:
        return {"id": self.id, "state": self.state, "progress": self.progress, "result": self.result,
                "error": self.error, "elapsed": round(time.time() - self.started, 2)}


class JobManager:
    def __init__(self, keep: int = 16) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._keep = keep

    def start(self, fn: Callable[[Job], dict]) -> Job:
        job = Job()
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._keep:
                oldest = next(iter(self._jobs.values()))
                if oldest.state == "running":
                    break
                self._jobs.popitem(last=False)

        def run() -> None:
            try:
                result = fn(job)
                if job.cancelled():
                    job.state = "cancelled"
                else:
                    job.result, job.state = result, "done"
            except Exception as exc:  # noqa: BLE001 - reported to the client
                traceback.print_exc()
                job.error, job.state = f"{type(exc).__name__}: {exc}", "error"

        threading.Thread(target=run, name=f"job-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)
