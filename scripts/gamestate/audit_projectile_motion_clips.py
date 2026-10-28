"""Measure contiguous native projectile clips; never reads the running game.

Each clip has the same world, address and ID, matching frame/own clocks, no
missing observation, no contradictory duplicate tick and at most five ticks
between samples. This measures XY velocity, not impact time or attack phase.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from scripts.gamestate.audit_recorded_fireball import fit
from scripts.gamestate.audit_recorded_tick_speeds import quantiles
from scripts.gamestate.audit_targeting_history import clock_frame, identity
from sim.projectile_calibration import projectile_speed

RAW_SPEEDS = dict(TowerCannonball=1000, ArcherArrow=600,
                  MusketeerProjectile=1000, FireballSpell=600)
BEFORE_SPEEDS = dict(TowerCannonball=16666, ArcherArrow=10000,
                     MusketeerProjectile=20000, FireballSpell=12000)


def fitted_clip(samples):
    """Reject contradictory ticks, discontinuities and curved trajectories."""
    by_tick = {}
    for tick, x, y in samples:
        value = (tick, x, y)
        if tick in by_tick and by_tick[tick] != value:
            return None
        by_tick[tick] = value
    points = list(by_tick.values())
    if any(not 0 < b[0]-a[0] <= 5 for a, b in zip(points, points[1:])):
        return None
    result = fit(points)
    if result is None or result['speed_millitiles_per_tick'] < 1:
        return None
    return dict(result, points=points)


def audit(paths):
    clips, sources, clocks = [], [], []
    counts = Counter()
    body_counts = Counter()
    for path in map(Path, paths):
        digest = hashlib.sha256()
        active = {}
        previous = None
        clock_anchor = None
        segment = 0

        def finish(key):
            track = active.pop(key)
            result = fitted_clip(track.pop('samples'))
            counts['candidate_clips'] += 1
            if result is not None:
                clips.append(dict(track, **result))

        with path.open('rb') as stream:
            for line_no, raw in enumerate(stream):
                digest.update(raw)
                row = json.loads(raw)
                frame = clock_frame(row)
                counts['raw_frames'] += 1
                boundary = (frame is None or previous is None
                            or frame['world'] != previous['world']
                            or not 0 <= frame['tick']-previous['tick'] <= 5)
                if boundary:
                    for key in list(active):
                        finish(key)
                    segment += 1
                    clock_anchor = None
                previous = frame
                if frame is None:
                    continue
                counts['valid_clock_frames'] += 1
                tick = frame['tick']
                device_ns = row['frame'].get('device_end_ns')
                if type(device_ns) is int:
                    if clock_anchor is None:
                        clock_anchor = (tick, device_ns)
                    elif tick-clock_anchor[0] >= 20:
                        dt = (device_ns-clock_anchor[1])/1e9
                        if dt > 0:
                            clocks.append((tick-clock_anchor[0])/dt)
                        clock_anchor = (tick, device_ns)
                bodies = frame['bodies'].values()
                body_counts.update(b['name'] for b in bodies)
                present = {identity(b): b for b in bodies if b['name'] in RAW_SPEEDS}
                for key in list(active):
                    if key not in present:
                        finish(key)
                for key, body in present.items():
                    if key not in active:
                        active[key] = dict(source=str(path), segment=segment,
                            world=frame['world'], identity=key, first_frame=line_no,
                            samples=[])
                    track = active[key]
                    track['last_frame'] = line_no
                    track['samples'].append((tick, body['x'], body['y']))
        for key in list(active):
            finish(key)
        sources.append(dict(path=str(path), sha256=digest.hexdigest()))
    clock_summary = quantiles(clocks)
    if not clocks or not 19.5 < clock_summary['median'] < 20.5:
        raise ValueError('The captures do not establish a 20 Hz timebase')
    grouped = defaultdict(list)
    for clip in clips:
        name = clip['identity'][2]
        speed = clip['speed_millitiles_per_tick']*20
        span_ms = clip['span_ticks']*50
        distance = speed*span_ms/1000
        clip['speed_mt_s'] = speed
        clip['before_duration_error_ms'] = distance/BEFORE_SPEEDS[name]*1000-span_ms
        clip['current_duration_error_ms'] = distance/projectile_speed(name, RAW_SPEEDS[name])*1000-span_ms
        grouped[name].append(clip)
    summary = {}
    for name, rows in sorted(grouped.items()):
        summary[name] = dict(clips=len(rows), recordings=len({r['source'] for r in rows}),
            speed_mt_s=quantiles([r['speed_mt_s'] for r in rows]),
            before_mt_s=BEFORE_SPEEDS[name], current_mt_s=projectile_speed(name, RAW_SPEEDS[name]),
            before_duration_error_ms=quantiles([r['before_duration_error_ms'] for r in rows]),
            current_duration_error_ms=quantiles([r['current_duration_error_ms'] for r in rows]))
    return dict(schema='contiguous-native-projectile-clips-v1', sources=sources,
        counts=dict(counts), body_observations=dict(body_counts), clock_hz=clock_summary,
        projectiles=summary, clips=clips,
        limitations=['Coordinates precede the clock read; no atomic snapshot guarantee.',
            'Duration errors refer to the observed straight segment, not birth-to-impact.',
            'No inferred target, muzzle offset, damage, AOE or attack timer is calibrated here.',
            'Clips within a battle are correlated; counts are not independent matches.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.paths)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(clock_hz=result['clock_hz'], projectiles=result['projectiles'])))
