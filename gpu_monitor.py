"""GPU hours and utilization monitoring for distributed workloads.

Tracks wall-clock GPU hours, peak memory, and periodic utilization/power
samples. Writes a JSON summary on exit — including on SIGTERM (SLURM timeout).

Usage::

    from preprocessing.gpu_monitor import GPUMonitor

    with GPUMonitor(output_dir="data/out", device=device, world_size=4):
        # ... GPU work ...
        pass
    # gpu_monitor.json written on exit
"""

import atexit
import json
import signal
import threading
import time
from pathlib import Path

import torch

try:
    import pynvml

    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


class GPUMonitor:
    """Context manager that tracks GPU hours and utilization metrics.

    Args:
        output_dir: Directory to write gpu_monitor.json.
        device: Torch device for this rank (e.g. ``torch.device("cuda:0")``).
        world_size: Total number of GPUs across all ranks.
        rank: This process's rank. Only rank 0 writes the summary file.
        sample_interval_s: Seconds between GPU metric samples (default 30).
    """

    def __init__(
        self,
        output_dir: str | Path,
        device: torch.device,
        world_size: int,
        rank: int = 0,
        sample_interval_s: float = 30.0,
    ):
        self._output_dir = Path(output_dir)
        self._device = device
        self._world_size = world_size
        self._rank = rank
        self._interval = sample_interval_s

        self._t_start: float = 0.0
        self._samples: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_handle = None
        self._written = False
        self._prev_sigterm = signal.SIG_DFL

    def __enter__(self):
        self._t_start = time.time()
        torch.cuda.reset_peak_memory_stats(self._device)
        self._init_pynvml()
        self._start_sampler()
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        atexit.register(self._write_summary)
        return self

    def __exit__(self, *exc):
        self._stop_sampler()
        self._write_summary()
        self._shutdown_pynvml()
        signal.signal(signal.SIGTERM, self._prev_sigterm)

    # ── pynvml lifecycle ─────────────────────────────────────────────

    def _init_pynvml(self) -> None:
        if not _HAS_PYNVML or not torch.cuda.is_available():
            return
        pynvml.nvmlInit()
        idx = self._device.index if self._device.index is not None else 0
        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

    def _shutdown_pynvml(self) -> None:
        if self._nvml_handle is not None:
            pynvml.nvmlShutdown()
            self._nvml_handle = None

    # ── sampling ─────────────────────────────────────────────────────

    def _sample_once(self) -> dict | None:
        if self._nvml_handle is None:
            return None
        util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
        temp = pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "t": round(time.time() - self._t_start, 1),
            "util_pct": util.gpu,
            "mem_pct": util.memory,
            "power_w": round(power_mw / 1000, 1),
            "temp_c": temp,
        }

    def _sampler_loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            sample = self._sample_once()
            if sample is not None:
                self._samples.append(sample)

    def _start_sampler(self) -> None:
        # take an initial sample immediately
        sample = self._sample_once()
        if sample is not None:
            self._samples.append(sample)
        self._thread = threading.Thread(target=self._sampler_loop, daemon=True)
        self._thread.start()

    def _stop_sampler(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # take a final sample
        sample = self._sample_once()
        if sample is not None:
            self._samples.append(sample)

    # ── output ───────────────────────────────────────────────────────

    def _build_summary(self) -> dict:
        elapsed_s = time.time() - self._t_start
        gpu_hours = elapsed_s * self._world_size / 3600

        summary: dict = {
            "gpu_hours": round(gpu_hours, 3),
            "wall_clock_s": round(elapsed_s, 1),
            "n_gpus": self._world_size,
        }

        if torch.cuda.is_available():
            summary["peak_memory_allocated_gb"] = round(
                torch.cuda.max_memory_allocated(self._device) / 1e9, 2
            )
            summary["peak_memory_reserved_gb"] = round(
                torch.cuda.max_memory_reserved(self._device) / 1e9, 2
            )

        if self._nvml_handle is not None:
            name = pynvml.nvmlDeviceGetName(self._nvml_handle)
            if isinstance(name, bytes):
                name = name.decode()
            summary["device"] = name

        if self._samples:
            summary["avg_utilization_pct"] = round(
                sum(s["util_pct"] for s in self._samples) / len(self._samples), 1
            )
            summary["avg_power_w"] = round(
                sum(s["power_w"] for s in self._samples) / len(self._samples), 1
            )
            summary["avg_temperature_c"] = round(
                sum(s["temp_c"] for s in self._samples) / len(self._samples), 1
            )
            summary["samples"] = self._samples

        return summary

    def _write_summary(self) -> None:
        if self._written or self._rank != 0:
            return
        self._written = True
        summary = self._build_summary()
        out_path = self._output_dir / "gpu_monitor.json"
        out_path.write_text(json.dumps(summary, indent=2) + "\n")

    # ── signal handling ──────────────────────────────────────────────

    def _sigterm_handler(self, signum, frame):
        self._stop_sampler()
        self._write_summary()
        self._shutdown_pynvml()
        # re-raise with previous handler
        signal.signal(signal.SIGTERM, self._prev_sigterm)
        signal.raise_signal(signal.SIGTERM)
