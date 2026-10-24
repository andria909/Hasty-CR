"""Capture the guarded native ARM64 reader; never starts or controls a battle."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

from scripts.gamestate.arm64_entities import FirstObservations, Memory, u64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="/tmp/Hasty-android-tools/platform-tools/adb")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--collection", required=True, type=lambda s: int(s, 0))
    parser.add_argument("--binary", default="/data/local/tmp/hasty_arm64_entities_20261022")
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--interval-ms", type=int, default=20)
    parser.add_argument("--screenshots-every", type=float, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600 or not 1 <= args.interval_ms <= 1000:
        parser.error("Invalid duration or sampling interval")
    memory = Memory(args.adb, args.serial)
    head = memory.read(args.collection, 24)
    vtable = u64(head)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "identity.json").write_text(json.dumps(
        {"pid": memory.pid, "serial": args.serial, "collection": hex(args.collection),
         "vtable": hex(vtable), "layout": "official-arm64-160402020",
         "binary": args.binary, "interval_ms": args.interval_ms}, indent=2))
    tracker = FirstObservations()
    counts: Counter = Counter()
    timings, samples, rejects, windows = [], 0, Counter(), []
    previous_start = None
    last_sequence = None

    def screenshot(number: int) -> dict:
        start = time.time_ns()
        name = f"screen_{number:04d}.png"
        try:
            data = subprocess.check_output(memory.adb + ["exec-out", "screencap", "-p"],
                                           timeout=15)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("Invalid PNG")
            (args.output / name).write_bytes(data)
            return {"file": name, "host_start_ns": start, "host_end_ns": time.time_ns()}
        except (subprocess.SubprocessError, RuntimeError) as error:
            return {"error": str(error), "host_start_ns": start,
                    "host_end_ns": time.time_ns()}

    command = memory.adb + ["exec-out", args.binary, str(memory.pid),
                            hex(args.collection), hex(vtable), str(args.seconds),
                            str(args.interval_ms)]
    screen_jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool, \
            (args.output / "frames.jsonl").open("x") as frames_file, \
            (args.output / "events.jsonl").open("x") as events_file, \
            (args.output / "errors.jsonl").open("x") as errors_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, bufsize=1)
        next_screen = time.monotonic()
        try:
            for line in process.stdout:
                received = time.time_ns()
                row = json.loads(line)
                row["host_received_ns"] = received
                if last_sequence is not None and row["sequence"] != last_sequence + 1:
                    raise RuntimeError("Missing native stream sequence")
                last_sequence = row["sequence"]
                if "rejected" in row:
                    rejects[row["rejected"]] += 1
                    errors_file.write(json.dumps(row) + "\n")
                    errors_file.flush()
                    continue
                samples += 1
                timings.append(row["read_us"] / 1000)
                frames_file.write(json.dumps(row) + "\n")
                frames_file.flush()
                row["sample_start_ns"] = row["device_start_ns"]
                row["sample_end_ns"] = row["device_end_ns"]
                for event in tracker.update(row):
                    event["device_first_seen_window_ns"] = [previous_start,
                                                            row["device_end_ns"]]
                    event["host_received_ns"] = received
                    event["position_source"] = "first_observed_body_position"
                    events_file.write(json.dumps(event) + "\n")
                    events_file.flush()
                    counts[event["name"]] += 1
                    windows.append((row["device_end_ns"] - previous_start) / 1e6)
                    print(f"NEW {event['name']} side={event['side']} "
                          f"xy=({event['x']},{event['y']}) "
                          f"id={event['entity_id']}", flush=True)
                previous_start = row["device_start_ns"]
                if args.screenshots_every > 0 and time.monotonic() >= next_screen:
                    if not screen_jobs or screen_jobs[-1].done():
                        screen_jobs.append(pool.submit(screenshot, len(screen_jobs)))
                        next_screen = time.monotonic() + args.screenshots_every
            code = process.wait(timeout=5)
            if code:
                raise RuntimeError(f"Native capture exited {code}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        screenshots = [job.result() for job in screen_jobs]
    (args.output / "screenshots.json").write_text(json.dumps(screenshots, indent=2))
    summary = {"frames": samples, "rejected_by_reason": dict(rejects),
               "first_observed_by_name": dict(counts), "placements_confirmed": 0,
               "read_ms_median": statistics.median(timings) if timings else None,
               "read_ms_max": max(timings) if timings else None,
               "first_seen_window_ms_max": max(windows) if windows else None,
               "screenshots": len(screenshots),
               "limitations": ["Not an atomic engine tick snapshot.",
                               "Body appearances are not card placement commands.",
                               "No proof of complete coverage of spell/effect collections."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
