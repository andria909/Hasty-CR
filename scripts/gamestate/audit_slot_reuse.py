"""Measure confirmed same-slot reuse from existing logs; no Android input."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def audit_events(events, cooldown_ms=2100):
    plays={}
    confirmed=[]
    for event in events:
        if event['event']=='PLAY':
            key=(event['match'],event['tick'],event['card'])
            if key in plays:raise ValueError('Ambiguous source action')
            plays[key]=event
        elif event['event']=='PLAY_ACK':
            key=(event['match'],event['source_tick'],event['card'])
            if key not in plays:raise ValueError('Confirmation has no source action')
            play=plays.pop(key)
            if event['tick']<play['tick']:raise ValueError('Confirmation before input')
            confirmed.append(dict(match=event['match'],slot=play['slot'],card=play['card'],
                source_tick=play['tick'],ack_tick=event['tick']))
    previous={};transitions=[]
    for action in sorted(confirmed,key=lambda x:(x['match'],x['source_tick'],x['slot'])):
        key=(action['match'],action['slot']);last=previous.get(key)
        if last is not None:
            delay=(action['source_tick']-last['ack_tick'])*50
            if delay<0:raise ValueError('Same slot reused before previous confirmation')
            transitions.append(dict(**action,previous_card=last['card'],previous_ack_tick=last['ack_tick'],
                ack_to_next_input_game_ms=delay,blocked_by_sim_cooldown=delay<cooldown_ms))
        previous[key]=action
    values=[r['ack_to_next_input_game_ms']for r in transitions]
    return dict(confirmed_plays=len(confirmed),transitions=transitions,
        transition_count=len(values),blocked_by_sim_cooldown=sum(r['blocked_by_sim_cooldown']for r in transitions),
        mean_ms=float(np.mean(values))if values else None,median_ms=float(np.median(values))if values else None,
        p05_ms=float(np.percentile(values,5))if values else None,
        immediate_reuse_cards=dict(Counter(r['card']for r in transitions if r['blocked_by_sim_cooldown'])))


def audit(paths):
    rows=[]
    for path in map(Path,paths):
        raw=path.read_bytes();events=[json.loads(line)for line in raw.splitlines()]
        rows.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),**audit_events(events)))
    return dict(schema='confirmed-slot-reuse-audit-v1',cooldown_ms=2100,runs=rows,
        transitions=sum(r['transition_count']for r in rows),
        blocked_by_sim_cooldown=sum(r['blocked_by_sim_cooldown']for r in rows),
        limitation='Native hand rotation confirms acceptance; timestamp is observed ACK, not exact spawn time. Offline timing comparison, not counterfactual win rate.',live_games_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--events',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.events)
    with Path(a.output).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='runs'}))
