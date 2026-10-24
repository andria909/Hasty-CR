"""Read current ARM64 battle entities or benchmark complete Mac/ADB requests.

Run with ``python -m scripts.gamestate.query_arm64 --help`` from the repo root.
No game input or memory writes. See agent.md for the pinned build and setup.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from scripts.gamestate.arm64_entities import CaptureError, Memory
from scripts.gamestate.elixir import ElixirTracker

BINARY = '/data/local/tmp/hasty_arm64_entities_20261022_v6'
STATE_BINARY = '/data/local/tmp/hasty_arm64_entities_20261022_v8'


class CurrentStateReader:
    """Attach once (ABI/version/hash guards), resolve the battle on every query.

    Recreate this object if the game process restarts. ``side`` is the engine's
    player index; it must not be assumed to mean ally/enemy. Coordinates are
    current body positions, not card-placement commands.
    """

    def __init__(self, adb: str, serial: str, binary: str = BINARY, *, own_state: bool = False):
        self.memory = Memory(adb, serial)
        if own_state and binary == BINARY:
            binary = STATE_BINARY
        if not binary.startswith('/data/local/tmp/') or any(
                c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/-.'
                for c in binary):
            raise ValueError('Binary must be a simple /data/local/tmp/ path')
        m = self.memory
        self.command = m.adb + ['exec-out', binary, str(m.pid), hex(m.libg_base),
                               hex(m.libg_base + 0x1936450), '1', '20',
                               '--auto-state-once' if own_state else '--auto-once']

    def query(self) -> dict:
        """Fresh snapshot. Timing includes process launch, ADB and JSON parsing.

        Rejected reads remain rejected: no silent retries or cached coordinates.
        The returned ``units`` excludes projectiles, and includes towers.
        """
        start = time.perf_counter_ns()
        wall = time.time_ns()
        process = subprocess.run(self.command, capture_output=True, timeout=5)
        try:
            frame = json.loads(process.stdout)
        except (ValueError, UnicodeError) as exc:
            raise CaptureError(f'Native reader failed ({process.returncode}): '
                               f'{process.stderr.decode(errors="replace")[:200]}') from exc
        if process.returncode in (0, 4) and frame.get('status') == 'no_battle':
            status = 'no_battle'
        elif process.returncode in (0, 3) and 'rejected' in frame:
            status = 'rejected'
        elif process.returncode == 0 and isinstance(frame.get('entities'), list):
            if frame.get('entity_count') != len(frame['entities']):
                raise CaptureError('Entity count mismatch')
            status = 'ok'
        else:
            raise CaptureError(f'Unexpected native response: {frame}')
        units = [dict(e, x_tiles=e['x'] / 1000, y_tiles=e['y'] / 1000)
                 for e in frame.get('entities', []) if e['entity_id'] // 1_000_000 == 5]
        elapsed = (time.perf_counter_ns() - start) / 1e6
        result = {'status': status, 'request_host_ns': wall,
                'received_host_ns': time.time_ns(), 'round_trip_ms': elapsed,
                'units': units, 'frame': frame}
        if 'own_state' in frame:
            result['own_state'] = frame['own_state']
        return result

    def query_with_elixir(self, tracker: ElixirTracker, *,
                          game_seconds: float) -> dict:
        """Enrich a fresh memory snapshot with separately modelled own elixir.

        The caller supplies trusted elapsed GAME time and records confirmed
        plays on tracker. Native entities do not yet supply those events or
        the game clock. An invalid read never exposes a usable economy value.
        Use a new tracker for every match; invalidate on missed own events.
        """
        row = self.query()
        row['own_elixir'] = (tracker.advance(game_seconds).to_dict()
                             if row['status'] == 'ok' else None)
        return row


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if r['status'] == 'ok']
    values = sorted(r['round_trip_ms'] for r in valid)
    stats = None if not values else {
        'median': statistics.median(values),
        'p95': values[max(0, math.ceil(.95 * len(values)) - 1)],
        'max': max(values), 'under_100ms': sum(v <= 100 for v in values),
        'under_150ms': sum(v <= 150 for v in values)}
    return {'requests': len(rows), 'accepted': len(valid),
            'rejected': sum(r['status'] == 'rejected' for r in rows),
            'no_battle': sum(r['status'] == 'no_battle' for r in rows),
            'round_trip_ms': stats,
            'native_read_ms_median': statistics.median(
                r['frame']['read_us'] / 1000 for r in valid) if valid else None,
            'entity_count_range': [min(r['frame']['entity_count'] for r in valid),
                                   max(r['frame']['entity_count'] for r in valid)] if valid else None,
            'scope': 'Request on Mac -> native root resolution/read -> ADB response '
                     '-> JSON parsing and unit coordinate conversion. Initial ABI/version/hash '
                     'attachment excluded. Not placement-event latency or screenshot recognition.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb', default='/tmp/Hasty-android-tools/platform-tools/adb')
    parser.add_argument('--serial', required=True)
    parser.add_argument('--binary', default=BINARY)
    parser.add_argument('--own-state', action='store_true',
                        help='Use v8 native private hand, cycle, mana and battle tick')
    parser.add_argument('--requests', type=int, default=1)
    parser.add_argument('--period-ms', type=float, default=100)
    parser.add_argument('--wait-for-battle', type=float, default=0,
                        help='Maximum seconds waiting for a readable battle; does not start one')
    parser.add_argument('--output', type=Path, help='New directory for raw requests and summary')
    args = parser.parse_args()
    if not 1 <= args.requests <= 10000 or not 0 <= args.period_ms <= 10000 or not 0 <= args.wait_for_battle <= 600:
        parser.error('Invalid finite request count, period or wait')
    if args.requests > 1 and args.output is None:
        parser.error('--output is required for a benchmark')
    reader = CurrentStateReader(args.adb, args.serial, args.binary, own_state=args.own_state)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'identity.json').write_text(json.dumps({
            'pid': reader.memory.pid, 'libg_base': hex(reader.memory.libg_base),
            'serial': args.serial, 'binary': args.binary}, indent=2))
    deadline = time.monotonic() + args.wait_for_battle
    while True:
        first = reader.query()
        if first['status'] == 'ok' or time.monotonic() >= deadline:
            break
        time.sleep(.25)
    rows = []
    stream = (args.output / 'requests.jsonl').open('x') if args.output else None
    try:
        for index in range(args.requests):
            started = time.monotonic()
            row = first if index == 0 else reader.query()
            rows.append(row)
            if stream:
                stream.write(json.dumps(row) + '\n')
                stream.flush()
            else:
                print(json.dumps(row))
            if index + 1 < args.requests:
                time.sleep(max(0, args.period_ms / 1000 - (time.monotonic() - started)))
    finally:
        if stream:
            stream.close()
    if args.output:
        summary = summarize(rows)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
