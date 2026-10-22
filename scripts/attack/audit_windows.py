"""Bounded reference actions at minimum attack windows, train/validation only.

This checks simulator feasibility of the reference action, not learned quality,
real-game parity or optimal timing. It never trains or opens the emulator.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from sim.attack_env import AttackEnv
from sim.attack_scenarios import build_catalog


def reference_action(env):
    s=env.scenario
    goal_card='hog_rider' if s.goal_kind=='hog_tower_damage' else s.finisher_spell
    if any(a['accepted'] and a['card']==goal_card for a in env._actions):return 0
    card=goal_card
    cycling=card not in env.match.players[1].hand
    if cycling:
        # A deterministic reference, not an optimal cycle policy. Favor cheap
        # bodies, keeping spells available for the actual attack exercise.
        hand=env.match.players[1].hand
        bodies=[name for name in hand if name not in env._spells]
        card=min(bodies or hand,key=lambda name:(env._cards[name].cost,name))
    slot=env.match.players[1].hand.index(card)
    if cycling and card not in env._spells:
        x,y=9,29
    elif s.goal_kind=='spell_finish' or cycling:
        target=env.match.towers[-1][s.target_lane if s.target_lane!='auto' else s.lane]
        x,y=int(target.pos.x//1000),int(target.pos.y//1000)
    else:
        x=3 if s.lane=='left' else 14
        y=29 if s.setup_kind=='back' else 12 if s.setup_kind=='pocket' else 17
    candidate=1+slot*576+y*18+x
    return candidate if env._action_mask()[candidate] else 0


def audit(*,split,seed=42,initial_hand_mode='scenario'):
    if split not in ('train','validation'):
        raise ValueError('Window calibration must not inspect held-out test')
    root=Path(__file__).resolve().parents[2]
    source_names=('sim/attack_env.py','sim/attack_scenarios.py','scripts/attack/audit_windows.py')
    source_sha={name:hashlib.sha256((root/name).read_bytes()).hexdigest()for name in source_names}
    scenarios,catalog=build_catalog()
    selected=[s for s in scenarios if s.split==split and s.deadline_visibility=='hidden'
              and not s.enemy_plays and (s.goal_kind!='spell_finish' or s.single_cast_sufficient)]
    records=[]
    for original in selected:
        low=original.deadline_ms_range[0]
        s=replace(original,deadline_ms_range=(low,low+1))
        env=AttackEnv(scenario=s,split=split,initial_hand_mode=initial_hand_mode)
        env.reset(seed=seed)
        for _ in range(1201):
            _,_,terminal,truncated,info=env.step(reference_action(env))
            if terminal or truncated:break
        else:raise RuntimeError('Finite reference-action audit bound exceeded')
        m=info['metrics']
        records.append(dict(scenario_id=original.scenario_id,family_id=s.family_id,
            goal_kind=s.goal_kind,setup_kind=s.setup_kind,lane=s.lane,
            minimum_window_ms=low,actual_window_ms=env._deadline_ms,
            elapsed_ms=m['elapsed_ms'],terminal_reason=m['terminal_reason'],
            goal_reached=m['terminal_reason'] in ('goal_success','enemy_king_destroyed'),
            hog_target_tower_damage=m['hog_target_tower_damage'],
            initial_hand=m['initial_hand'],initial_queue=m['initial_queue'],
            illegal_actions=m['illegal_actions'],actions=env._actions))
    if any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest for name,digest in source_sha.items()):
        raise RuntimeError('Source changed during audit; report is not reproducible')
    return dict(schema='attack-reference-window-audit-v1',split=split,seed=seed,initial_hand_mode=initial_hand_mode,
        source_sha256=source_sha,catalog_count=catalog['count'],episodes=len(records),
        goal_reached=sum(r['goal_reached']for r in records),records=records,
        quality_accepted=False,all_windows_calibrated=False,trained=False,
        limitations=['Deterministic release plus cheap-body cycle reference; not an optimal or learned policy',
            'No enemy defenses or multi-cast finishers in this initial timing check',
            'One seed per scenario; no conclusion for other hands or timers',
            'Simulator timing only; real-game mechanics remain unverified'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('train','validation'),required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--initial-hand-mode',choices=('scenario','shuffle'),default='scenario')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    report=audit(split=args.split,seed=args.seed,initial_hand_mode=args.initial_hand_mode)
    report['wall_seconds']=time.monotonic()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ('records','source_sha256')}))


if __name__=='__main__':main()
