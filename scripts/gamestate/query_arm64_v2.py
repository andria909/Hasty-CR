"""Hardened CLI for the pinned, read-only ARM64 game-state reader.

This keeps the v1 memory/layout checks and output schema. It improves ADB
discovery and turns missing-device/process/timeout failures into concise errors.
It never starts a game or changes device settings.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.gamestate.arm64_entities import CaptureError, PACKAGE
from scripts.gamestate.query_arm64 import (
    BINARY, STATE_BINARY, CurrentStateReader, summarize,
)


def resolve_adb(explicit: str | None = None) -> str:
    """Find adb in explicit, SDK, or PATH locations without changing the host."""
    if explicit:
        return explicit
    candidates = [os.environ.get("ANDROID_ADB")]
    for key in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        root = os.environ.get(key)
        if root:
            candidates.append(str(Path(root) / "platform-tools" / "adb"))
    candidates.extend([
        str(Path.home() / "Library/Android/sdk/platform-tools/adb"),
        "/tmp/Hasty-android-tools/platform-tools/adb",
        shutil.which("adb"),
    ])
    for candidate in candidates:
        if candidate and (candidate == "adb" or Path(candidate).is_file()):
            return candidate
    raise CaptureError("adb not found; install Android SDK Platform-Tools or pass --adb")


def create_reader(adb: str, serial: str, binary: str, own_state: bool, *, su: str | None = None):
    try:
        kwargs = {"own_state": own_state}
        if su is not None:
            kwargs["su"] = su
        return CurrentStateReader(adb, serial, binary, **kwargs)
    except subprocess.CalledProcessError as exc:
        command = exc.cmd[-1] if isinstance(exc.cmd, list) and exc.cmd else ""
        if command == f"pidof {PACKAGE}":
            raise CaptureError(
                "Clash Royale is not running on this device; open the game and retry. "
                "This script does not launch it."
            ) from exc
        raise CaptureError(f"ADB command failed while attaching to {serial}: {command}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CaptureError(f"ADB timed out while attaching to {serial}") from exc
    except ValueError as exc:
        raise CaptureError(f"Could not read the Clash Royale process on {serial}") from exc


def query(reader: CurrentStateReader) -> dict:
    try:
        return reader.query()
    except subprocess.TimeoutExpired as exc:
        raise CaptureError("Native state query timed out; no cached state was returned") from exc
    except subprocess.CalledProcessError as exc:
        raise CaptureError("ADB failed during the native state query") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", help="adb executable (otherwise discovered from Android SDK/PATH)")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--su", help="Already installed Android su executable; does not root the device")
    parser.add_argument("--binary", default=BINARY)
    parser.add_argument("--own-state", action="store_true",
                        help="Use v8 native private hand, cycle, mana and battle tick")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--period-ms", type=float, default=100)
    parser.add_argument("--wait-for-battle", type=float, default=0,
                        help="Maximum seconds waiting for a readable battle; does not start one")
    parser.add_argument("--output", type=Path, help="New directory for raw requests and summary")
    args = parser.parse_args(argv)
    if (not 1 <= args.requests <= 10000 or not 0 <= args.period_ms <= 10000
            or not 0 <= args.wait_for_battle <= 600):
        parser.error("Invalid finite request count, period or wait")
    if args.requests > 1 and args.output is None:
        parser.error("--output is required for a benchmark")

    try:
        adb = resolve_adb(args.adb)
        reader = create_reader(adb, args.serial, args.binary, args.own_state, su=args.su)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "identity.json").write_text(json.dumps({
                "pid": reader.memory.pid,
                "libg_base": hex(reader.memory.libg_base),
                "serial": args.serial,
                "binary": args.binary,
            }, indent=2))

        deadline = time.monotonic() + args.wait_for_battle
        while True:
            first = query(reader)
            if first["status"] == "ok" or time.monotonic() >= deadline:
                break
            time.sleep(.25)

        rows = []
        stream = (args.output / "requests.jsonl").open("x") if args.output else None
        try:
            for index in range(args.requests):
                started = time.monotonic()
                row = first if index == 0 else query(reader)
                rows.append(row)
                if stream:
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                else:
                    print(json.dumps(row))
                if index + 1 < args.requests:
                    time.sleep(max(0, args.period_ms / 1000 - (time.monotonic() - started)))
        finally:
            if stream:
                stream.close()

        if args.output:
            (args.output / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
            print(json.dumps(summarize(rows), indent=2))
        return 0
    except CaptureError as exc:
        print(f"query_arm64_v2: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"query_arm64_v2: output/ADB error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
