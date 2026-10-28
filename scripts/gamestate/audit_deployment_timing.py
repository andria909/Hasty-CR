"""Read-only deployment timing evidence from saved native frames and confirmed inputs.

Candidate memory fields and first movement are NOT an activation oracle.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

BODY_CARD = {'Knight': 'knight', 'HogRider': 'hog_rider', 'Musketeer': 'musketeer',
             'ElectroSpirit': 'electro_spirit', 'Skeleton': 'skeletons', 'Cannon': 'cannon'}


def distribution(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    def percentile(q):
        p = (len(values) - 1) * q
        lo = int(p)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (p - lo)
    return dict(n=len(values), min=min(values), median=statistics.median(values),
                mean=statistics.mean(values), p05=percentile(.05), p95=percentile(.95), max=max(values))


def collect(events, frames):
    starts = sorted((e for e in events if e['event'] == 'MATCH_START'), key=lambda e: e['host_ns'])
    times = [e['host_ns'] for e in starts]
    plays = defaultdict(list)
    for e in events:
        if e['event'] == 'PLAY':
            plays[e['match'], e['tick'], e['card']].append(e)
    accepted = []
    for e in events:
        if e['event'] != 'PLAY_ACK':
            continue
        source = plays[e['match'], e['source_tick'], e['card']]
        if len(source) != 1 or e['tick'] < source[0]['tick']:
            raise ValueError('Missing, ambiguous or backwards confirmed input')
        accepted.append(dict(source[0], ack_tick=e['tick']))
    bodies = {}
    for frame in frames:
        index = bisect_right(times, frame['received_host_ns']) - 1
        if index < 0:
            continue
        start = starts[index]
        state = frame.get('frame', {}).get('own_state') or {}
        tick = state.get('tick')
        if tick is None or tick < start['tick'] or state.get('own_side') != start['side']:
            continue
        for u in frame.get('units', []):
            if u['side'] != start['side'] or u['name'] not in BODY_CARD:
                continue
            deploy = u.get('deployment_tick_candidate', -1)
            seq = u.get('play_sequence_candidate', -1)
            if deploy < 0 or seq < 0 or tick < deploy:
                continue
            key = (start['match'], state.get('world'), u['entity_id'], deploy, seq)
            row = bodies.get(key)
            if row is None:
                row = bodies[key] = dict(match=start['match'], entity_id=u['entity_id'],
                    card=BODY_CARD[u['name']], deployment_tick_candidate=deploy,
                    sequence_candidate=seq, first_seen_tick=tick, first_position=[u['x'], u['y']],
                    last_tick=tick, first_movement_tick=None)
            elif tick > row['last_tick']:
                if (row['first_movement_tick'] is None and tick <= deploy + 80
                        and math.dist(row['first_position'], [u['x'], u['y']]) >= 100):
                    row['first_movement_tick'] = tick
                row['last_tick'] = tick
    groups = defaultdict(list)
    for row in bodies.values():
        groups[row['match'], row['card'], row['deployment_tick_candidate'], row['sequence_candidate']].append(row)
    rows = []
    for (match, card, deploy, seq), units in sorted(groups.items()):
        first_seen = min(u['first_seen_tick'] for u in units)
        candidates = [p for p in accepted if p['match'] == match and p['card'] == card
                      and 0 <= deploy - p['tick'] <= 60
                      and deploy <= p['ack_tick'] + 4 and abs(first_seen - p['ack_tick']) <= 40]
        if len(candidates) != 1:
            continue
        p = candidates[0]
        movements = [u['first_movement_tick'] for u in units if u['first_movement_tick'] is not None]
        moved = min(movements) if movements else None
        rows.append(dict(match=match, card=card, source_tick=p['tick'], ack_tick=p['ack_tick'],
            deployment_tick_candidate=deploy, sequence_candidate=seq, body_count=len(units),
            first_seen_tick=first_seen, first_movement_tick=moved,
            input_to_candidate_ms=(deploy-p['tick'])*50, input_to_ack_ms=(p['ack_tick']-p['tick'])*50,
            candidate_to_first_seen_ms=(first_seen-deploy)*50,
            candidate_to_first_movement_ms=None if moved is None else (moved-deploy)*50,
            units=units))
    return dict(confirmed_plays=len(accepted), candidate_body_groups=len(groups), matched_groups=len(rows), rows=rows)


def audit(paths):
    runs = []
    for root in map(Path, paths):
        events_raw = (root/'events.jsonl').read_bytes()
        digest = hashlib.sha256()
        def frames():
            with (root/'frames.jsonl').open('rb') as f:
                for line in f:
                    digest.update(line)
                    yield json.loads(line)
        result = collect([json.loads(line) for line in events_raw.splitlines()], frames())
        runs.append(dict(root=str(root), events_sha256=hashlib.sha256(events_raw).hexdigest(),
                         frames_sha256=digest.hexdigest(), **result))
    rows = [row for run in runs for row in run['rows']]
    fields = ['input_to_candidate_ms', 'input_to_ack_ms', 'candidate_to_first_seen_ms',
              'candidate_to_first_movement_ms']
    summary = {card: {field: distribution([r[field] for r in rows if r['card'] == card and r[field] is not None])
                     for field in fields} for card in sorted({r['card'] for r in rows})}
    return dict(schema='native-deployment-timing-evidence-v1', tick_ms=50, movement_threshold_mt=100,
        summary=summary, runs=runs, live_games_started=False,
        limitations=['Candidate deployment and sequence fields remain unvalidated memory interpretations.',
            'First movement >=100mt within4s is an observed displacement, not exact activation; attacks, collisions, knockback and sampling can alter it.',
            'Unique confirmed matches only; spells excluded; multi-body spawns grouped, not independent samples.',
            'Observed ACK is hand rotation, not exact deployment or readiness. These data alone cannot prove all timing correct.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = audit(a.runs)
    with Path(a.output).open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result['summary'], indent=2))
