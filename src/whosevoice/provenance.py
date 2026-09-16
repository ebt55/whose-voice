"""Record how a result CSV was produced, next to the CSV.

The dilution bug (notes/17) hid for one reason: `embed_dilution.csv` recorded the
*nominal* density the script was asked for and nothing about how that density was
realised, so two byte-identical rows under different labels looked like a measurement.
No result CSV in this repo carried `n`, `seed`, `chunk`, `n_boot`, the encoder or the
commit it was run from, which is also why `metrics_undefended.csv` cannot be traced to
its inputs.

`write_results` writes the CSV and a `<name>.meta.json` sidecar carrying exactly that.
The sidecar is cheap, is diffable, and answers the first question a careful reader asks.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_sha(repo: Path | None = None) -> str:
    """Short SHA of HEAD, with a `+dirty` marker, or "unknown" outside a checkout."""
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{sha}+dirty" if dirty else sha
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return "unknown"


def run_meta(**fields: Any) -> dict[str, Any]:
    """Standard provenance block: caller's parameters plus commit, time and argv."""
    return {
        "git_sha": git_sha(),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": " ".join(sys.argv),
        **fields,
    }


def write_results(df, path: Path, meta: dict[str, Any] | None = None) -> Path:
    """Write `df` to `path` and its provenance to `path` + '.meta.json'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    block = dict(meta or {})
    block.setdefault("rows", int(len(df)))
    block.setdefault("columns", list(df.columns))
    sidecar.write_text(json.dumps(run_meta(**block), indent=2), encoding="utf-8")
    return sidecar


def _take_lock(path: Path, stale_after: float) -> Path:
    """Refuse to start if another live process is already writing `path`.

    Why this exists. On 2026-09-16 four `run_embed_dilution.py --targets-only` processes
    ended up running at once, because a flaky `tasklist | grep` check reported "no python
    running" when there was, and each apparent failure was answered with a relaunch. They
    all wrote results/embed_dilution_K5.csv. The checkpointing that makes a single run
    crash-safe makes concurrent runs *destructive*: a run that starts from scratch
    truncates the file to its own first density, so a 30-row partial became a 2-row one
    and ~17 minutes of completed work vanished with no error anywhere.

    Nothing in the pipeline noticed. The CSV stayed well-formed, the sidecar stayed
    internally consistent, and only a row count that had gone *down* gave it away. A
    results file with no writer identity cannot tell you this happened - the same blind
    spot as the missing `realised_density` column in notes/17, one level up.

    The lock is advisory and deliberately dumb: a sidecar file holding the owning PID and
    argv. A lock whose PID is dead, or older than `stale_after`, is reclaimed rather than
    left to block the next run forever.
    """
    lock = path.with_suffix(path.suffix + ".lock")
    if lock.exists():
        try:
            held = json.loads(lock.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            held = {}
        pid, started = held.get("pid"), held.get("started_epoch", 0)
        if pid and _pid_alive(pid) and (time.time() - started) < stale_after:
            raise SystemExit(
                f"{path.name} is already being written by PID {pid} "
                f"({held.get('argv', '?')}, started {held.get('started_utc', '?')}).\n"
                f"Two runs writing one results file silently destroy each other's work "
                f"(see whosevoice.provenance.exclusive_output). Wait for it, or kill it "
                f"and delete {lock.name}.")
        print(f"note: reclaiming stale lock {lock.name} (pid {pid} not running)")
    lock.write_text(json.dumps({
        "pid": os.getpid(),
        "argv": " ".join(sys.argv),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "started_epoch": time.time(),
    }, indent=2), encoding="utf-8")
    return lock


@contextmanager
def exclusive_output(path: Path, stale_after: float = 6 * 3600):
    """Context-manager form of the output lock. See `_take_lock`."""
    lock = _take_lock(path, stale_after)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def acquire_output_lock(path: Path, stale_after: float = 6 * 3600) -> Path:
    """Take the output lock for the life of the process, releasing it at exit.

    For scripts whose whole body writes one results file: one call near the top beats
    re-indenting main() into a `with` block. Release is registered with atexit, so it
    happens on normal return, on an unhandled exception and on SystemExit - but NOT if
    the process is SIGKILLed, which is why `_take_lock` reclaims locks whose PID is dead.
    """
    lock = _take_lock(path, stale_after)
    atexit.register(lambda: lock.unlink(missing_ok=True))
    return lock


def _pid_alive(pid: int) -> bool:
    """True if `pid` is a live process. Windows has no os.kill(pid, 0) equivalent."""
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True
