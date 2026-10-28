"""Contact context for saved early displacement, not verified collision labels."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from scripts.gamestate.audit_sim_alignment import unit_profiles
from sim.gamedata import to_snake_case


def radius(raw, profiles):
    # These are SIMULATOR radii: differences are evidence to investigate,
    # not grounds to declare live bodies physically overlapping.
    if raw['name'] in ('PrincessTower', 'KingTower'):
        from sim.entities import make_tower
        from sim.arena import Point
        return make_tower(0, 1, Point(0, 0), 1, 0, 1000, 0,
                          king=raw['name'] == 'KingTower').collision_radius_mt
    profile = profiles.get(to_snake_case(raw['name']))
    if profile is None or profile.flying:
        return None
    return profile.collision_radius_mt


def contacts(source, bodies, profiles):
    sr = radius(source, profiles)
    if sr is None:
        return []
    rows = []
    for other in bodies:
        if other['entity_id'] == source['entity_id'] or other['hp'] <= 0:
            continue
        rr = radius(other, profiles)
        if rr is None:
            continue
        distance = math.hypot(source['x']-other['x'], source['y']-other['y'])
        clearance = distance - sr - rr
        if clearance <= 200:
            rows.append(dict(entity_id=other['entity_id'], name=other['name'], side=other['side'],
                distance_mt=distance, sim_clearance_mt=clearance,
                same_spawn=(other['side']==source['side'] and
                    other.get('play_sequence_candidate', -1)>=0 and
                    other.get('play_sequence_candidate')==source.get('play_sequence_candidate') and
                    other.get('deployment_tick_candidate')==source.get('deployment_tick_candidate'))))
    return sorted(rows, key=lambda r:r['sim_clearance_mt'])


def audit(path):
    source = Path(path).read_bytes()
    evidence = json.loads(source)
    profiles = unit_profiles()
    results = []
    for run in evidence['runs']:
        root = Path(run['root'])
        raw_events = (root/'events.jsonl').read_bytes()
        if hashlib.sha256(raw_events).hexdigest() != run['events_sha256']:
            raise ValueError('Events changed since timing audit')
        starts = sorted((json.loads(line) for line in raw_events.splitlines()
                         if json.loads(line)['event']=='MATCH_START'), key=lambda r:r['host_ns'])
        times = [r['host_ns'] for r in starts]
        windows = {}
        for row in run['rows']:
            delay = row['candidate_to_first_movement_ms']
            if delay is None or delay>=1000:
                continue
            for unit in row['units']:
                moved = unit['first_movement_tick']
                if moved is not None and (moved-unit['deployment_tick_candidate'])*50<1000:
                    key = (row['match'], unit['entity_id'], unit['deployment_tick_candidate'])
                    windows[key] = dict(root=str(root), match=row['match'], card=row['card'],
                        entity_id=unit['entity_id'], sequence=row['sequence_candidate'],
                        deploy_tick=unit['deployment_tick_candidate'], first_seen=unit['first_seen_tick'],
                        moved_tick=moved, snapshots=[])
        digest = hashlib.sha256()
        with (root/'frames.jsonl').open('rb') as f:
            for line in f:
                digest.update(line)
                frame = json.loads(line)
                i = bisect_right(times, frame['received_host_ns'])-1
                if i<0:
                    continue
                state = frame.get('frame', {}).get('own_state') or {}
                tick = state.get('tick')
                if tick is None or state.get('own_side')!=starts[i]['side']:
                    continue
                for body in frame.get('units', []):
                    key = (starts[i]['match'], body['entity_id'], body.get('deployment_tick_candidate'))
                    window = windows.get(key)
                    if window and window['first_seen']<=tick<=window['moved_tick']:
                        if window['snapshots'] and tick<=window['snapshots'][-1]['tick']:
                            continue
                        window['snapshots'].append(dict(tick=tick, position=[body['x'],body['y']],
                            contacts=contacts(body, frame['units'], profiles)))
        if digest.hexdigest() != run['frames_sha256']:
            raise ValueError('Frames changed since timing audit')
        for window in windows.values():
            all_contacts=[c for s in window['snapshots'] for c in s['contacts']]
            window['near_contact_observed']=bool(all_contacts)
            window['same_spawn_nearby']=any(c['same_spawn'] for c in all_contacts)
            window['negative_sim_clearance']=any(c['sim_clearance_mt']<0 for c in all_contacts)
            results.append(window)
    groups={}
    for row in results:
        key=(row['root'],row['match'],row['card'],row['sequence'])
        group=groups.setdefault(key,dict(near_contact_observed=False,same_spawn_nearby=False,negative_sim_clearance=False))
        for field in group:
            group[field] |= row[field]
    return dict(schema='early-displacement-contact-context-v1', input=str(path),
        input_sha256=hashlib.sha256(source).hexdigest(), groups=len(groups), bodies=len(results),
        group_counts=dict(Counter(k for row in groups.values() for k,v in row.items() if v)),
        rows=results, mechanics_changed=False, labels_verified=False, live_games_started=False,
        limitations=['Contacts use simulator radii plus200mt; proximity is not proof of a collision.',
                     'Recorded motion could also reflect knockback, spawn formation or memory-field semantics.',
                     'Native target and activation state are absent; this does not establish attack readiness.'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--timing',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    result=audit(args.timing)
    with Path(args.output).open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
