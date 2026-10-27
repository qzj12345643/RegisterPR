#!/usr/bin/env python3
"""Stop a persistent training process group at an explicit timezone-aware time."""

import argparse
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{datetime.now(timezone.utc).isoformat()}] {message}\n")
        stream.flush()


def process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, PermissionError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgid", type=int, required=True)
    parser.add_argument("--leader-pid", type=int, required=True)
    parser.add_argument("--stop-at", required=True, help="ISO-8601 time, including timezone")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    target = datetime.fromisoformat(args.stop_at)
    if target.tzinfo is None:
        raise SystemExit("--stop-at must include a timezone offset")
    target_utc = target.astimezone(timezone.utc)
    log(args.log, f"watchdog started pid={os.getpid()} target_pgid={args.pgid} target_leader={args.leader_pid}")
    log(args.log, f"scheduled_stop={target.isoformat()} ({target_utc.isoformat()})")

    delay = (target_utc - datetime.now(timezone.utc)).total_seconds()
    if delay > 0:
        log(args.log, f"sleeping_seconds={delay:.1f}")
        time.sleep(delay)

    cmdline = process_cmdline(args.leader_pid)
    if not cmdline:
        log(args.log, "target leader is already gone; no signal sent")
        return 0
    if "run_joint_future_registers_persistent" not in cmdline:
        log(args.log, f"refusing to signal unexpected leader cmdline={cmdline!r}")
        return 2

    log(args.log, "sending SIGTERM to the training process group")
    try:
        os.killpg(args.pgid, signal.SIGTERM)
    except ProcessLookupError:
        log(args.log, "training process group was already gone")
        return 0

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            os.killpg(args.pgid, 0)
        except ProcessLookupError:
            log(args.log, "training process group exited after SIGTERM")
            return 0
        time.sleep(5.0)

    log(args.log, "training process group did not exit within 60 seconds; sending SIGKILL")
    try:
        os.killpg(args.pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    log(args.log, "scheduled stop complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
