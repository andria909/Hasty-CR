"""Offline native-trace audit: observed speed, map anchors, targeting coverage.

No Android access or input. Device monotonic time is a real-time denominator,
not an independently validated game clock. Trajectory filters select candidates
for free movement; they cannot prove absence of buffs or network stutter.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from sim import arena
from sim.engine import Battle
from sim.entities import make_tower, make_unit, speed_to_mt_per_sec
from sim.gamedata import load_gamedata, to_snake_case
from sim.readiness import report as readiness_report

TOWERS = ('KingTower', 'PrincessTower')


def stats(values):
    if not values:
        return None
    values = sorted(values)
    return {'n': len(values), 'median': statistics.median(values),
            'p05': values[max(0, math.ceil(.05 * len(values)) - 1)],
            'p95': values[math.ceil(.95 * len(values)) - 1],
            'min': values[0], 'max': values[-1]}


def unit_profiles():
    result = {}
    for card in load_gamedata(level=11).values():
        if card.unit is not None:
            key = to_snake_case(card.unit.name)
            if key in result and result[key] != card.unit:
                raise ValueError(f'Ambiguous profile for {key}')
            result[key] = card.unit
    return result


def simulate_speed(spec):
    battle = Battle()
    source = battle.add(make_unit(0, spec, 1, arena.Point(3500, 12000)))
    source.deploy_remaining_ms = 0
    battle.add(make_tower(0, -1, arena.Point(3500, 1000), 1_000_000, 0, 1000, 0))
    start = source.pos
    for _ in range(40):
        battle.step(50)
    return math.hypot(source.pos.x - start.x, source.pos.y - start.y) / 1000 / 2


def targeting_cases(profiles):
    def run(source_name, targets):
        battle = Battle()
        source = battle.add(make_unit(0, profiles[source_name], 1, arena.Point(6000, 12000)))
        source.deploy_remaining_ms = 0
        options = {}
        for name, x, y in targets:
            target = battle.add(make_unit(0, profiles[name], -1, arena.Point(x, y)))
            target.deploy_remaining_ms = 0
            options[target.uid] = name
        battle._acquire_target(source)
        return {'source': source_name, 'candidates': [x[0] for x in targets],
                'sim_selected': options.get(source.target_uid),
                'real_selected': None, 'real_comparison': 'unverified_no_target_uid_in_trace'}
    cases = [
        run('knight', [('mega_minion', 6500, 12000), ('musketeer', 8000, 12000)]),
        run('musketeer', [('mega_minion', 6500, 12000), ('knight', 8500, 12000)]),
        run('hog_rider', [('knight', 6500, 12000), ('cannon', 8500, 12000)]),
    ]
    battle = Battle()
    source = battle.add(make_unit(0, profiles['knight'], 1, arena.Point(6000, 12000)))
    old = battle.add(make_unit(0, profiles['musketeer'], -1, arena.Point(8500, 12000)))
    battle._acquire_target(source)
    new = battle.add(make_unit(0, profiles['knight'], -1, arena.Point(7000, 12000)))
    battle.now_ms = 500 - source.uid * 50 % 500
    battle._acquire_target(source)
    cases.append({'source': 'knight', 'scenario': 'closer_enemy_while_old_target_still_valid',
                  'previous': old.name, 'new': new.name,
                  'sim_switched_on_rescan': source.target_uid == new.uid,
                  'real_comparison': 'unverified_requires_target_lock_probe'})
    return cases


def speed_candidates(rows, profiles):
    tracks = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get('status') != 'ok':
            continue
        frame = row['frame']
        timestamp = frame.get('device_start_ns')
        if not isinstance(timestamp, int):
            continue
        # Reader does not expose world ID: repeated crown-tower addresses scope
        # a continuous trace conservatively. Input itself must be one live match.
        identity = tuple(sorted((u['entity_id'], u.get('address'), u['side'])
                                for u in row['units'] if u['name'] in TOWERS))
        for entity in row['units']:
            if entity['name'] not in TOWERS:
                key = (identity, entity['entity_id'], entity.get('address'), entity['name'], entity['side'])
                tracks[key].append((timestamp / 1e9, entity, index))
    candidates = defaultdict(list)
    ids = defaultdict(set)
    for key, observations in tracks.items():
        name = to_snake_case(key[3])
        if name not in profiles:
            continue
        profile = profiles[name]
        first = observations[0][0]
        for start in range(len(observations)):
            t0, a, index0 = observations[start]
            if t0 - first < .5:
                continue
            end = start + 1
            while end < len(observations) and observations[end][0] - t0 < .55:
                end += 1
            if end >= len(observations):
                continue
            t1, b, index1 = observations[end]
            dt = t1 - t0
            if not .55 <= dt <= .85:
                continue
            points = observations[start:end + 1]
            if any(q['hp'] != a['hp'] or q['hp'] <= 0 for _, q, _ in points):
                continue
            if any(r['status'] != 'ok' for r in rows[index0:index1 + 1]):
                continue
            if any(v[0] - u[0] > .25 for u, v in zip(points, points[1:])):
                continue
            chord = math.hypot(b['x'] - a['x'], b['y'] - a['y'])
            path = sum(math.hypot(v[1]['x'] - u[1]['x'], v[1]['y'] - u[1]['y'])
                       for u, v in zip(points, points[1:]))
            if path <= 0 or chord / path < .98:
                continue
            clear = True
            for _, body, frame_index in points:
                for other in rows[frame_index]['units']:
                    if other['entity_id'] == body['entity_id'] or other['hp'] <= 0:
                        continue
                    gap = math.hypot(other['x'] - body['x'], other['y'] - body['y'])
                    # Conservative proximity filter, NOT proof of an idle-free,
                    # unbuffed, unblocked engine state.
                    radius = 2000 if other['side'] == body['side'] else max(2000, profile.range_mt + 1000)
                    if other['name'] in TOWERS and other['side'] == body['side']:
                        continue
                    if gap < radius:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                candidates[name].append(chord / dt / 1000)
                ids[name].add(key)
    result = {}
    for name in sorted({to_snake_case(k[3]) for k in tracks}):
        if name not in profiles:
            result[name] = {'status': 'unmapped_profile'}
            continue
        nominal = speed_to_mt_per_sec(profiles[name].speed_mt_per_sec) / 1000
        values = candidates[name]
        result[name] = {
            'nominal_sim_tiles_per_second': nominal,
            'sim_isolated_measured_tiles_per_second': simulate_speed(profiles[name]),
            'real_candidate_tiles_per_second': stats(values),
            'candidate_body_tracks': len(ids[name]),
            'candidate_median_vs_nominal_pct':
                (statistics.median(values) / nominal - 1) * 100 if values and nominal else None,
            'status': 'exploratory_real_time_estimate_not_validated_base_speed',
        }
    return result


def geometry(first):
    simulated = []
    for side, anchors, king in ((0, arena.ENEMY_PRINCESS, arena.ENEMY_KING),
                                 (1, arena.ALLY_PRINCESS, arena.ALLY_KING)):
        for lane, point in dict(anchors, king=king).items():
            simulated.append((side, lane, point))
    result = []
    for raw in first['units']:
        if raw['name'] not in TOWERS:
            continue
        lane = 'king' if raw['name'] == 'KingTower' else 'left' if raw['x'] < 9000 else 'right'
        sim = next(p for side, name, p in simulated if side == raw['side'] and name == lane)
        result.append({'engine_side': raw['side'], 'lane': lane,
                       'real': [raw['x'] / 1000, raw['y'] / 1000],
                       'sim': [sim.x / 1000, sim.y / 1000],
                       'real_minus_sim_tiles': [(raw['x'] - sim.x) / 1000, (raw['y'] - sim.y) / 1000]})
    return result


def audit(path: Path, start_row: int):
    all_rows = [json.loads(line) for line in path.open() if line.strip()]
    rows = all_rows[start_row:]
    accepted = [r for r in rows if r.get('status') == 'ok']
    if not accepted:
        raise ValueError('No accepted snapshots')
    profiles = unit_profiles()
    result = {
        'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'excluded_leading_rows': start_row, 'rows': len(rows),
        'speed': speed_candidates(rows, profiles),
        'geometry': geometry(accepted[0]),
        'targeting': targeting_cases(profiles),
        'sim_readiness': readiness_report(),
        'coordinate_contract': {
            'training_observation_y': 'clip(round(31 - world_y_tiles), 0, 31)',
            'training_action_y': 'world_y_millitiles = grid_y * 1000 + 500',
            'status': 'legacy_observation_action_y_mismatch_preserved_for_v25',
        },
        'limitations': [
            'One existing live-match trace; no new battles or CV.',
            'Native target IDs, movement buffs, attack state and trusted game clock are absent.',
            'Speed uses device real time; unknown stalls/speed changes preclude asserting game-time base speed.',
            'Overlapping windows are correlated, not independent trials.',
            'No targeting accuracy percentage is computable without verified target labels.',
            'Geometry comparison assumes the documented native side0-lowY / side1-highY world in this capture.',
            'Client data absent: current simulator loads public snapshot fallback; no unsupported source graphs in readiness does not prove full coverage.',
        ]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--start-row', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.start_row < 0:
        parser.error('--start-row must be nonnegative')
    result = audit(args.trace, args.start_row)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
