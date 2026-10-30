"""Paid central-Cannon combinations on the same bank-teacher TRAIN fixtures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import product
import json
from pathlib import Path
from scripts.attack.build_bank_counterfactuals import opening
from scripts.attack.search_late_air_defense import rollout,rank
from sim.cannon_placement import CENTRAL_CELLS


def context(spec):
    seed,lane,bank,hand_kind,*extra=spec
    support=bool(extra and extra[0])
    env=opening(seed,lane,bank,15)
    if support:
        from sim.adapter import grid_to_point
        enemy=env.match.players[-1]
        if 'baby_dragon' not in enemy.hand:raise ValueError('Paid support must be in the real current hand')
        before=enemy.elixir
        if not env.match.play_card(-1,'baby_dragon',grid_to_point(3 if lane=='left'else 14,12,1)):
            raise ValueError('Paid support rejected')
        if before-enemy.elixir!=4000:raise ValueError('Wrong support elixir cost')
    player=env.match.players[1]
    player.hand=['cannon','electro_spirit','zap','musketeer'if hand_kind=='musketeer'else'fireball']
    player.queue=[c for c in player.deck if c not in player.hand]
    hp={k:t.hitpoints for k,t in env.match.towers[1].items()}
    records=[]
    def attempt(kind,first=None,second=None,third=None):
        row=rollout(env.match,hp,env.cards,env.spells,lane,first,second,third=third)
        row.update(kind=kind,first=first,second=second,third=third)
        records.append(row);return row
    attempt('wait')
    singles=[]
    for x,y in CENTRAL_CELLS:
        row=attempt('cannon',dict(card='cannon',cell=[x,y]))
        if row['valid']:singles.append(row)
    if hand_kind=='musketeer':
        for x,y in [(3 if lane=='left'else 14,20),(6 if lane=='left'else 11,22),(3 if lane=='left'else 14,24)]:
            attempt('musketeer',dict(card='musketeer',cell=[x,y]))
    for single in sorted(singles,key=rank)[:2]:
        first=single['first']
        for helper in ('electro_spirit','zap','fireball'if hand_kind=='fireball'else'musketeer'):
            for offset,variant in product((1500,3000,5000),range(3)):
                attempt('cannon_'+helper,first,dict(card=helper,offset_ms=offset,variant=variant))
        for offset,variant in product((1500,3000,5000),range(3)):
            attempt('cannon_spirit_zap',first,dict(card='electro_spirit',offset_ms=offset,variant=1),
                dict(card='zap',offset_ms=offset+250,variant=variant))
    best={kind:min((r for r in records if r['kind']==kind and r['valid']),key=rank,default=None)
          for kind in sorted({r['kind']for r in records})}
    return dict(seed=seed,lane=lane,bank=bank,hand=hand_kind,support=support,best=best,trials=records,
        normal_range_mode=env.match.battle.normal_range_mode,hog_sight_mode=env.match.battle.hog_sight_mode)


def search(output,smoke=False,workers=2,support=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cases=list(product((352,353,354),('left','right'),(3000,6000,10000),('musketeer','fireball')))
    cases=[(*case,support)for case in cases]
    if smoke:cases=cases[:1]
    rows=[]
    with(output/'contexts.jsonl').open('x')as stream:
        if workers==1:
            iterator=map(context,cases);pool=None
        else:
            pool=ProcessPoolExecutor(max_workers=workers);iterator=pool.map(context,cases)
        try:
            for row in iterator:
                if row['normal_range_mode']!='edge_to_edge_v1' or row['hog_sight_mode']!='side_clip_v1':raise ValueError('Wrong combat modes')
                stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
                print(json.dumps(dict(completed=len(rows),total=len(cases),trials=sum(len(r['trials'])for r in rows))),flush=True)
        finally:
            if pool:pool.shutdown()
    summary={}
    for kind in sorted({k for row in rows for k in row['best']}):
        found=[row['best'][kind]for row in rows if row['best'].get(kind)]
        summary[kind]=dict(valid_contexts=len(found),no_damage=sum(r['clear_without_damage']for r in found),
            mean_damage=sum(r['total_damage']for r in found)/len(found)if found else None)
    result=dict(schema='bank-central-cannon-search-v1',contexts=len(rows),split='train',summary=summary,support=support,
        raw_sha256=hashlib.sha256((output/'contexts.jsonl').read_bytes()).hexdigest(),
        limitations=['Synthetic fixed hand/bank, paid Balloon and optional paid Baby Dragon, no reinforcements',
            'All36 central Cannon cells; two best singles retained for helper search',
            'Bounded observed-position helper targeting, not global optimum or live winrate'],
        live_games=False,quality_accepted=False)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--support',action='store_true')
    a=p.parse_args();search(a.output,a.smoke,a.workers,a.support)
