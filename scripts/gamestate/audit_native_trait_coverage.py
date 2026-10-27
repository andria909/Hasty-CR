"""Compare old card-only traits with exact character lookup on recorded matches."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from scripts.gamestate.model_bridge import NativeModelAdapter
from sim.gamedata import to_snake_case


def audit(root):
    root=Path(root);adapter=NativeModelAdapter(own_side=1,allow_unknown_profiles=True,allow_incomplete_history=True,
        coordinate_mode='physical_live_cells_v2',geometry='native_anchors_candidate_v1')
    old={to_snake_case(c.unit.name)for c in adapter.cards.values()if c.unit is not None}
    count=Counter();before=Counter();after=Counter();resolved=Counter();ids=defaultdict(set);bodies=defaultdict(set);sources={}
    for folder,allowed,offset in [('run',{1,2,3},0),('recovery_run',set(range(1,7)),4)]:
        events=root/folder/'events.jsonl';starts={};ends={}
        for line in events.read_text().splitlines():
            e=json.loads(line);number=e.get('match')
            if number not in allowed:continue
            if e['event']=='MATCH_START':starts[number]=e['host_ns']
            if e['event']=='MATCH_END':ends[number]=e['host_ns']
        if set(starts)!=allowed or set(ends)!=allowed:raise ValueError('Complete autonomous match windows required')
        windows=[(n+offset,starts[n],ends[n])for n in allowed]
        path=root/folder/'frames.jsonl';digest=hashlib.sha256()
        for raw in path.open('rb'):
            digest.update(raw);r=json.loads(raw);own=r.get('own_state',{});frame=r.get('frame',{})
            if r.get('status')!='ok'or own.get('status')!='ok'or frame.get('battle_tick')!=own.get('tick'):continue
            number=next((n for n,a,b in windows if a<=r.get('received_host_ns',0)<=b),None)
            if number is None:continue
            count['active_clock_valid_snapshots']+=1
            for u in r.get('units',[]):
                if u.get('name')in ('KingTower','PrincessTower')or u.get('hp',0)<=0:continue
                name=u['name'];count['unit_snapshot_instances']+=1
                if to_snake_case(name)in old:continue
                before[name]+=1;ids[name].add(u['card_id']);bodies[name].add((number,u['entity_id'],u.get('address')))
                profile,alias=adapter.unit_profile(u)
                if profile is None:after[name]+=1
                else:resolved[name]+=1
        sources[str(path)]=digest.hexdigest();sources[str(events)]=hashlib.sha256(events.read_bytes()).hexdigest()
    return dict(schema='recorded-native-trait-coverage-v1',autonomous_matches=9,manual_match4_excluded=True,
        counters=dict(count),unknown_before=dict(before),unknown_after=dict(after),resolved_instances=dict(resolved),
        distinct_previously_unknown_bodies={name:len(v)for name,v in bodies.items()},
        native_card_ids={name:sorted(v)for name,v in ids.items()},sources=sources,
        changed_sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()for n in('scripts/gamestate/model_bridge.py','sim/gamedata.py','data/royaleapi/cards.json')},
        known_resolution=['GoblinBrawler exact complete character-table lookup','Goblin_Stab guarded Goblins catalogue card26000002 alias'],
        limitations=['Snapshot instances are not independent decisions or matches',
            'Tactical profiles remain nominal explicit level11; live HP is measured directly',
            'Unverified hero forms retain warnings and are not assigned invented base statistics',
            'This measures input coverage, not correct targeting, spell accuracy or win rate'],
        quality_accepted=False,live_games_started=False,training_sources_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    target=Path(a.output)
    if target.exists():raise ValueError('Preserve existing coverage audit')
    target.write_text(json.dumps(audit(a.root),indent=2)+'\n')
