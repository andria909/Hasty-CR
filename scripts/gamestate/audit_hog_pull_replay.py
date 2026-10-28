"""Compare fixed Hog sight hypotheses after a recorded river crossing.

Diagnostic only: a smaller radius can hide an incorrect route/initial lock.
No simulator parameter is changed and no hypothesis is accepted automatically.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from scripts.gamestate.audit_targeting_history import clock_frame, identity
from scripts.gamestate.replay_recorded_episode import extract, replay


def audit(paths):
    cases, sources = [], []
    for path in map(Path, paths):
        raw = path.read_bytes()
        sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
        frames = [clock_frame(json.loads(s)) for s in raw.splitlines()]
        seen = set()
        for index, frame in enumerate(frames):
            if frame is None or index == 0:
                continue
            previous = frames[index-1]
            if (previous is None or frame['world'] != previous['world']
                    or not 0 < frame['tick']-previous['tick'] <= 5):
                continue
            for uid, body in frame['bodies'].items():
                key = (frame['world'], identity(body))
                if body['name'] != 'HogRider' or key in seen:
                    continue
                old = previous['bodies'].get(uid)
                if old is None or identity(old) != identity(body):
                    continue
                crossing = (old['y'] < 16650 <= body['y'] if body['side'] == 0
                            else old['y'] > 15350 >= body['y'])
                if not crossing:
                    continue
                cannons = [b for b in frame['bodies'].values()
                    if b['name'] == 'Cannon' and b['side'] != body['side']
                    and b['hp'] > 0 and math.hypot(b['x']-body['x'], b['y']-body['y']) < 10000]
                if not cannons:
                    continue
                seen.add(key)
                end = frame['tick']
                for following in frames[index+1:]:
                    if (following is None or following['world'] != frame['world']
                            or not 0 <= following['tick']-end <= 5):
                        break
                    other = following['bodies'].get(uid)
                    if other is None or identity(other) != identity(body):
                        break
                    end = following['tick']
                    if end-frame['tick'] >= 50:
                        break
                if end-frame['tick'] < 30:
                    continue
                try:
                    clip = extract(path, index, end, uid, f'postriver_{len(cases)}')
                except ValueError:
                    continue
                hypotheses = {}
                for sight in (5500, 6500, 7500, 9500):
                    result = replay(clip, sight_override=sight)
                    hypotheses[str(sight)] = {k: result[k] for k in
                        ('mean_position_error_tiles', 'max_position_error_tiles',
                         'simulated_target_changes')}
                cases.append(dict(source=str(path), frame=index, first_tick=frame['tick'],
                    last_tick=end, uid=uid, side=body['side'],
                    cannons=[[b['x'], b['y']] for b in cannons], hypotheses=hypotheses))
    differences = [c['hypotheses']['5500']['mean_position_error_tiles']
                   - c['hypotheses']['9500']['mean_position_error_tiles'] for c in cases]
    return dict(schema='recorded-hog-postriver-sight-diagnostic-v1', sources=sources,
        cases=cases, summary=dict(cases=len(cases),
            mean_error_by_sight={str(s): mean(c['hypotheses'][str(s)]['mean_position_error_tiles']
                for c in cases) if cases else None for s in (5500, 6500, 7500, 9500)},
            smaller_radius_better=sum(d < -.02 for d in differences),
            smaller_radius_worse=sum(d > .02 for d in differences),
            effectively_unchanged=sum(abs(d) <= .02 for d in differences)),
        accepted_physics_change=False,
        limitations=['Focal movement only with recorded context; no complete battle reconstruction.',
            'Initial target is unknown; the scan begins after crossing and does not validate river pathing.',
            'Same-match cases are correlated; this is not independent statistical validation.',
            'The 5.5 hypothesis also changes tower visibility/retention, not only new building acquisition.',
            'A smaller XY error alone cannot identify the real acquisition rule.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.paths)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result['summary']))
