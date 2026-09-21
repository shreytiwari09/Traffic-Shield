"""
Hardware and latency profiling for the evaluation harness.

TWO PROBLEMS THIS FIXES

1. THE OLD GPU NUMBERS WERE NOT FROM THIS MACHINE.
   The previous harness shelled out to `nvidia-smi` and recorded whatever it
   printed. On the machine this project now runs on, `nvidia-smi` is not even
   on PATH — yet the committed metrics_report.json carries
   `gpu_mem_used_mb: 5350.7`, and reports 2063.6 MB of "GPU memory" for
   *Gemini*, which is a network call that touches no local GPU at all. Those
   figures were produced on different hardware and are meaningless here.

   Worse, the number it recorded was never per-model in the first place: it is
   whatever the whole machine's GPU happened to be holding at that instant,
   including other processes.

   So capability is now DETECTED, once, and reported explicitly. When there is
   no GPU the report says so — `{"available": false, "reason": "..."}` — rather
   than emitting a number that looks like a measurement. An absent metric that
   announces its own absence is honest; a fabricated one is not.

2. TTFT WAS UNMEASURABLE.
   The old harness called Ollama with `"stream": False` and Gemini
   non-streaming, then reported total latency. With no token-by-token events
   there is no first-token moment to observe, so Time To First Token simply
   did not exist. Generation is streamed here so TTFT and tokens/sec are real
   measurements rather than derived guesses.

WHAT "PEAK" MEANS HERE
Memory is sampled on a background thread for the duration of the call and the
maximum is kept, rather than taking one reading after the call returns (which
is what the old harness did, and which systematically misses the peak — by the
time a response arrives, the allocation that caused the peak has often already
been freed).
"""

import functools
import shutil
import subprocess
import threading
import time

import psutil


# ---------------------------------------------------------------------------
# Capability detection — run once, cached
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def gpu_capability() -> dict:
    """What GPU telemetry, if any, this machine can actually provide.

    Cached because it shells out and cannot change mid-run. Returns a dict that
    always states availability explicitly, so every consumer is forced to
    handle "no GPU" rather than defaulting a missing number to zero.
    """
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "vendor": None,
            "reason": "nvidia-smi not found on PATH — no NVIDIA GPU telemetry available on this host.",
        }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        line = out.decode().strip().split("\n")[0]
        name, total = (part.strip() for part in line.split(","))
        return {
            "available": True,
            "vendor": "nvidia",
            "device": name,
            "total_memory_mb": float(total),
            "reason": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "vendor": None,
            "reason": f"nvidia-smi present but failed ({type(exc).__name__}) — treating GPU telemetry as unavailable.",
        }


def gpu_memory_used_mb() -> float | None:
    if not gpu_capability()["available"]:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def host_profile() -> dict:
    """Static description of the machine the numbers were measured on.

    Embedded in the report so a set of latency figures can never be read
    without the hardware that produced them — the exact confusion the old
    committed report caused.
    """
    freq = None
    try:
        f = psutil.cpu_freq()
        freq = round(f.max or f.current, 0) if f else None
    except Exception:
        freq = None
    return {
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_max_mhz": freq,
        "total_ram_mb": round(psutil.virtual_memory().total / (1024 * 1024), 1),
        "gpu": gpu_capability(),
    }


# ---------------------------------------------------------------------------
# Peak sampling during a call
# ---------------------------------------------------------------------------
class ResourceSampler:
    """Samples memory (and GPU memory, where available) on a background thread
    while a generation call runs, keeping the peak.

    Used as a context manager around the call. The sampling interval is a
    compromise: fine enough to catch a model load, coarse enough that the
    sampler itself does not distort the CPU measurement it is taking.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_ram_mb: float | None = None
        self.peak_gpu_mb: float | None = None
        self.baseline_ram_mb: float | None = None
        self.cpu_percent: float | None = None
        self._gpu_ok = gpu_capability()["available"]
        self._proc = psutil.Process()

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)  # prime the counter
        while not self._stop.is_set():
            try:
                ram = psutil.virtual_memory().used / (1024 * 1024)
                self.peak_ram_mb = ram if self.peak_ram_mb is None else max(self.peak_ram_mb, ram)
                if self._gpu_ok:
                    g = gpu_memory_used_mb()
                    if g is not None:
                        self.peak_gpu_mb = g if self.peak_gpu_mb is None else max(self.peak_gpu_mb, g)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceSampler":
        self.baseline_ram_mb = round(psutil.virtual_memory().used / (1024 * 1024), 1)
        psutil.cpu_percent(interval=None)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.cpu_percent = round(psutil.cpu_percent(interval=None), 1)

    def result(self) -> dict:
        """Measured footprint.

        `peak_ram_delta_mb` is the headline: host-wide *used* memory is
        dominated by everything else running, so the delta over the baseline
        taken immediately before the call is far closer to what this
        generation actually cost. It is still host-wide, so a heavy background
        process inflates it — stated here rather than buried.
        """
        delta = None
        if self.peak_ram_mb is not None and self.baseline_ram_mb is not None:
            delta = round(self.peak_ram_mb - self.baseline_ram_mb, 1)
        gpu_cap = gpu_capability()
        return {
            "peak_ram_mb": round(self.peak_ram_mb, 1) if self.peak_ram_mb is not None else None,
            "baseline_ram_mb": self.baseline_ram_mb,
            "peak_ram_delta_mb": delta,
            "cpu_percent": self.cpu_percent,
            "gpu_available": gpu_cap["available"],
            "peak_gpu_mem_mb": self.peak_gpu_mb,
            "gpu_note": gpu_cap["reason"],
        }


# ---------------------------------------------------------------------------
# Latency math
# ---------------------------------------------------------------------------
# A provider that hands back the whole answer in one or two stream events is
# not streaming in any useful sense — the "decode window" after the first event
# is then a few milliseconds of transport, not generation.
_MIN_CHUNKS_FOR_DECODE_WINDOW = 3
# Even with enough chunks, a decode window this small a fraction of total time
# means the tokens did not actually arrive incrementally.
_MIN_WINDOW_FRACTION = 0.10


def throughput(
    completion_tokens: int | None,
    ttft_s: float | None,
    total_s: float | None,
    chunk_count: int = 0,
) -> dict | None:
    """Tokens per second, with the basis it was computed on.

    Ideally this measures the GENERATION phase only, excluding the time before
    the first token: on a CPU-only host, prompt ingestion and model load
    dominate wall-clock time (measured here: 71s TTFT against 119s total for
    llama3.1:8b), so folding them in yields a "tokens/sec" that mostly reflects
    prompt length rather than decode speed.

    But that only holds when tokens genuinely arrive incrementally. Gemini's
    stream returned a full answer in ~2 events, leaving a 21 ms window and an
    absurd 2244 tok/s. So the decode-window basis is used only when the stream
    really was incremental; otherwise this falls back to total elapsed time and
    SAYS SO in `basis`, because a throughput number whose denominator is
    unstated is not comparable across providers.
    """
    if not completion_tokens or not total_s or total_s <= 0:
        return None

    window = None
    if ttft_s is not None and total_s > ttft_s:
        candidate = total_s - ttft_s
        if chunk_count >= _MIN_CHUNKS_FOR_DECODE_WINDOW and candidate / total_s >= _MIN_WINDOW_FRACTION:
            window = candidate

    if window is not None:
        return {
            "tokens_per_second": round(completion_tokens / window, 2),
            "basis": "decode_window",
            "window_s": round(window, 3),
        }
    return {
        "tokens_per_second": round(completion_tokens / total_s, 2),
        "basis": "total_elapsed",
        "window_s": round(total_s, 3),
        "note": "Response did not arrive incrementally enough to isolate decode time; "
                "computed over total elapsed time, so it includes prompt ingestion.",
    }
