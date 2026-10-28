"""Paired, conditioned native Hog replays for lateral building acquisition.

This isolates post-river trajectories. It cannot validate jump routing, native
target pointers, damage, or a complete independent battle. Fixture cohorts
predate this change; the additional cohort is a regression check, not a new
untouched holdout (it was previously used to reject a radial sight reduction).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from scripts.gamestate.replay_recorded_episode import extract, replay


def verify_sources(fixture):
    """Reconstruct every portable clip from its hash-pinned native source."""
    for item in fixture['episodes']+fixture.get('full_crossing_controls',[]):
        clip = item['clip']
        rebuilt = extract(clip['source'], clip['frames'][0]['frame_index'],
                          clip['frames'][-1]['tick'], clip['focal_identity'][0], clip['name'])
        # extract uses tuples for identities; JSON fixtures use lists.
        if json.loads(json.dumps(rebuilt)) != clip:
            raise ValueError('Raw episode differs from fixture: '+clip['name'])


def audit(fixture):
    rows=[]
    modes={'radial_9500':('legacy_radial_v1',None),
           'radial_5500':('legacy_radial_v1',5500),
           'side_clip':('side_clip_v1',None)}
    for item in fixture['episodes']:
        clip=item['clip'];results={}
        for label,(mode,sight) in modes.items():
            results[label]=replay(clip,sight_override=sight,
                                 battle_options={'hog_sight_mode':mode})
        rows.append(dict(cohort=item['cohort'],name=clip['name'],results=results))
    cohorts={}
    for cohort in sorted({r['cohort']for r in rows}):
        group=[r for r in rows if r['cohort']==cohort]
        differences=[r['results']['side_clip']['mean_position_error_tiles']
                     -r['results']['radial_9500']['mean_position_error_tiles'] for r in group]
        cohorts[cohort]=dict(episodes=len(group),
            mean_error_tiles={m:mean(r['results'][m]['mean_position_error_tiles']for r in group)for m in modes},
            better=sum(d<-.02 for d in differences),worse=sum(d>.02 for d in differences),
            unchanged_within_002=sum(abs(d)<=.02 for d in differences))
    controls=[]
    for item in fixture.get('full_crossing_controls',[]):
        clip=item['clip']
        controls.append(dict(name=clip['name'],results={label:replay(clip,
            battle_options={'hog_sight_mode':mode}) for label,mode in
            [('radial_9500','legacy_radial_v1'),('side_clip','side_clip_v1')]}))
    return dict(schema='native-hog-side-clip-audit-v1',cohorts=cohorts,episodes=rows,
        full_crossing_controls=controls,
        scope='Only new lateral Hog/non-crown-building acquisition is changed.',
        limitations=['Native target pointers and initial locks are unavailable.',
            'Conditioned replay starts after river crossing; jump remains incorrect.',
            'Other bodies follow recorded positions; no independent battle, damage or body separation.',
            'Same-match episodes are correlated; XY error is not a model win rate.',
            'Neither cohort is a new untouched test set.',
            'Back clipping and other unit types are not enabled by this patch.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-sources',action='store_true')
    args=p.parse_args();raw=args.fixtures.read_bytes();fixture=json.loads(raw)
    if args.verify_sources:verify_sources(fixture)
    result=audit(fixture)
    result['fixture']=dict(path=str(args.fixtures),sha256=hashlib.sha256(raw).hexdigest(),
                           raw_sources_reconstructed=args.verify_sources)
    result['implementation']=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in map(Path,(__file__,'scripts/gamestate/replay_recorded_episode.py',
                             'sim/engine.py','sim/entities.py','sim/gamedata.py','sim/public_gamedata.py'))]
    with args.output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['cohorts']))
