"""Shared logging, progress/ETA reporting and atomic checkpoint helpers for the pipeline scripts."""
import os
import time
import torch

_log_file = None


def set_log_file(path):
    """Mirror every log line into this file as well as the terminal."""
    global _log_file
    _log_file = path


def fmt_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


def atomic_save(obj, path):
    """torch.save that can never leave a half-written file behind if the process is killed mid-write."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def checkpoint_len(path):
    """How many items a progress checkpoint holds (None if it cannot be read)."""
    try:
        obj = torch.load(path)
    except Exception:
        return None
    if isinstance(obj, dict) and "meta" in obj:
        return len(obj["meta"])
    if isinstance(obj, (list, tuple)):
        return len(obj)
    return None


class Progress:
    """Counts finished items in a stage and logs 'done/total, left, elapsed, eta' lines.

    A line is logged whenever `every` items have finished since the last line, whenever
    `every_seconds` have passed, and when the stage completes.
    """

    def __init__(self, label, total, done=0, every=None, every_seconds=60):
        self.label, self.total, self.done = label, total, done
        self.start_done = done
        self.every, self.every_seconds = every, every_seconds
        self.t0 = self.last_print = time.time()
        if done:
            log(f"{label}: resuming with {done}/{total} already done, {total - done} left")
        else:
            log(f"{label}: 0/{total} done, {total} left")

    def step(self, n=1, extra=""):
        self.done += n
        now = time.time()
        due = (
            (self.every and (self.done - self.start_done) % self.every == 0)
            or now - self.last_print >= self.every_seconds
            or self.done >= self.total
        )
        if due:
            self.report(extra)

    def report(self, extra=""):
        now = time.time()
        self.last_print = now
        elapsed = now - self.t0
        made = self.done - self.start_done
        rate = made / elapsed if elapsed > 0 and made > 0 else None
        left = self.total - self.done
        pct = 100.0 * self.done / self.total if self.total else 100.0
        eta = fmt_duration(left / rate) if rate else "?"
        line = f"{self.label}: {self.done}/{self.total} ({pct:.1f}%) | {left} left | elapsed {fmt_duration(elapsed)} | eta {eta}"
        if rate:
            line += f" | {rate:.2f} items/s" if rate >= 1 else f" | {1 / rate:.1f}s per item"
        if extra:
            line += f" | {extra}"
        log(line)


class Stage:
    """Context manager that logs a numbered stage header, its duration, and remembers the current stage."""
    current = None

    def __init__(self, idx, total, name):
        self.idx, self.total, self.name = idx, total, name
        self.skipped = False

    def __enter__(self):
        Stage.current = self
        self.t0 = time.time()
        log(f"===== stage {self.idx}/{self.total}: {self.name} =====")
        return self

    def skip(self, why):
        self.skipped = True
        log(f"already done ({why}) - skipping")

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None and not self.skipped:
            log(f"stage {self.idx}/{self.total} finished in {fmt_duration(time.time() - self.t0)}")
        return False
