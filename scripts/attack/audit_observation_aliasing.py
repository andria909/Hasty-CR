"""Controlled simulator counterexample: same public frame, different intent.

The intervention is on simulator target retention, NOT a proposed hidden input.
It establishes partial observability, not the cause of every live error.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sim.continuous_push import ContinuousPushEnv
from sim.entities import make_unit,MOVING
from sim.adapter import grid_to_point


def audit():
    env=ContinuousPushEnv();env.reset(seed=101120);match=env.match
    threat=match.battle.add(make_unit(0,env.cards['knight'].unit,-1,grid_to_point(9,16,1),0))
    baits=[match.battle.add(make_unit(0,env.cards['knight'].unit,1,grid_to_point(x,18,1),0))for x in(6,12)]
    for unit in [threat]+baits:unit.deploy_remaining_ms=0
    snapshot=env.snapshot();observations=[];outcomes=[]
    for target in baits:
        env.restore(snapshot);unit=env.match.battle.entities[threat.uid]
        unit.target_uid=target.uid;unit.state=MOVING
        observations.append(env._observe())
        initial=dict(x=unit.pos.x,y=unit.pos.y,target_uid=unit.target_uid)
        for _ in range(20):env.match.step(50)
        unit=env.match.battle.entities[threat.uid]
        outcomes.append(dict(initial=initial,after_1000ms=dict(x=unit.pos.x,y=unit.pos.y,target_uid=unit.target_uid)))
    identical={k:bool(np.array_equal(observations[0][k],observations[1][k]))for k in observations[0]}
    a,b=[r['after_1000ms']for r in outcomes];separation=math.hypot(a['x']-b['x'],a['y']-b['y'])
    if not all(identical.values())or separation<1000:raise ValueError('Counterexample not reproduced')
    return dict(schema='controlled-public-observation-aliasing-v1',same_model_input=identical,
        next_position_separation_mt=separation,branches=outcomes,elapsed_ms=1000,
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()for n in
            ('sim/env.py','sim/eval_defense_duel.py','sim/continuous_push.py','sim/engine.py')},
        learning_changed=False,live_games_started=False,
        limits=['Synthetic hidden-target intervention, not a recorded live transition',
            'Does not establish optimal spell action or verify real targeting physics',
            'History may reduce ambiguity; no claim it makes the game fully observable'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=audit()
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
