"""Offline paired left/right probe on recorded native states; never sends input.

This diagnoses policy sensitivity, not alternative-model wins on the same game.
Pending taps are unavailable: use the fresh mechanical mask, not execution mask.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from scripts.attack.audit_native_series import sha
from scripts.gamestate.memory_play import own_context
from scripts.gamestate.model_bridge import MemoryPolicy, NativeModelAdapter, MissingModelInput
from sim.lane_reflection import reflect_actions, reflect_observation


def sample_frames(series_file, per_match=12):
    series=json.loads(Path(series_file).read_text());selected=[];sources={str(series_file):sha(series_file)}
    for run in dict.fromkeys(m['run'] for m in series['matches'] if not m.get('interrupted')):
        matches=[m for m in series['matches'] if m['run']==run and not m.get('interrupted')]
        events_path=Path(run)/'events.jsonl';frames_path=Path(run)/'frames.jsonl'
        sources[str(events_path)]=sha(events_path);sources[str(frames_path)]=sha(frames_path)
        events=[json.loads(x) for x in events_path.read_text().splitlines()]
        windows=[]
        for m in matches:
            start=next(e['host_ns'] for e in events if e['event']=='MATCH_START' and e['match']==m['local_match'])
            end=next(e['host_ns'] for e in events if e['event']=='MATCH_END' and e['match']==m['local_match'])
            windows.append((start,end,m['number']))
        counts=Counter();next_tick=Counter()
        with frames_path.open() as stream:
            for line in stream:
                row=json.loads(line);own=row.get('own_state',{})
                if row.get('status')!='ok' or own.get('status')!='ok' or own['tick']<20 or own['elixir_raw']<30000:
                    continue
                # Avoid imputing destroyed towers for an isolated sampled state.
                if sum(u['name'] in ('KingTower','PrincessTower') and u['hp']>0 for u in row['units'])!=6:
                    continue
                number=next((n for a,b,n in windows if a<=row['received_host_ns']<=b),None)
                if number is None or counts[number]>=per_match or own['tick']<next_tick[number]:continue
                selected.append((number,row));counts[number]+=1;next_tick[number]=own['tick']+100
    return selected,sources


def probe(series_file,models):
    torch.set_num_threads(1);frames,sources=sample_frames(series_file);results=[]
    for entry in models:
        path=Path(entry['checkpoint'])
        if sha(path)!=entry['sha256']:raise ValueError('Pinned model changed')
        policy=MemoryPolicy(path)
        if not policy.joint:raise ValueError('This paired probe requires the joint309 contract')
        rows=[];errors=Counter()
        for number,snapshot in frames:
            private=snapshot['own_state']
            adapter=NativeModelAdapter(own_side=private['own_side'],allow_incomplete_history=True,
                allow_unknown_profiles=True,coordinate_mode=policy.coordinate_mode,geometry=policy.geometry)
            try:
                view=adapter.encode(snapshot,own_context(private));policy.validate_view_contract(view)
                if not view.metadata['ready_for_inference']:raise MissingModelInput(str(view.metadata['blockers']))
            except MissingModelInput as e:
                errors[str(e)]+=1;continue
            obs=dict(planes=view.observation['planes'],scalars=np.concatenate((view.observation['scalars'],
                np.array([0.,0.,float(not view.metadata['enemy_bodies_present'])],dtype=np.float32))))
            mirrored=reflect_observation(obs);mask=view.action_mask
            with torch.inference_mode():
                logits,_=policy.network(torch.from_numpy(np.stack([obs['planes'],mirrored['planes']])),
                    torch.from_numpy(np.stack([obs['scalars'],mirrored['scalars']])))
                original=logits[0].numpy();back=reflect_actions(logits[1].numpy())
                distributions=[]
                for value in (original,back):
                    distributions.append(torch.softmax(torch.from_numpy(value).masked_fill(~torch.from_numpy(mask),-torch.inf),0).numpy())
            a,b=distributions;middle=(a+b)/2;eps=1e-30
            js=float(.5*(np.sum(a*np.log((a+eps)/(middle+eps)))+np.sum(b*np.log((b+eps)/(middle+eps))))/np.log(2))
            actions=[int(x.argmax()) for x in distributions]
            def choice(action):
                if action==0:return dict(card='wait',lane='none')
                if action>=2305:return dict(card='ability',lane='none')
                slot,cell=divmod(action-1,576);y,x=divmod(cell,18)
                return dict(card=own_context(private).hand[slot],lane='left' if x<9 else 'right',grid=[x,y])
            rows.append(dict(match=number,tick=private['tick'],js=js,equivariant_greedy=actions[0]==actions[1],
                             original=choice(actions[0]),mirrored_then_mapped_back=choice(actions[1])))
        results.append(dict(model=entry,samples=len(rows),rejected=dict(errors),
            mean_reflection_js=float(np.mean([r['js'] for r in rows])) if rows else None,
            greedy_reflection_agreement=float(np.mean([r['equivariant_greedy'] for r in rows])) if rows else None,
            original_lanes=dict(Counter(r['original']['lane'] for r in rows)),
            original_cards=dict(Counter(r['original']['card'] for r in rows)),rows=rows))
    return dict(schema='native-lane-reflection-probe-v1',sources=sources,results=results,
        limitations=['Off-policy recorded BC800 states: not a win-rate comparison.',
                     'Only states with six alive towers and at least three elixir; samples share matches.',
                     'Pending card commands and recent slot cooldowns are not replayed.',
                     'Reflection agreement alone does not establish tactical quality.'],live_games_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--series',required=True);p.add_argument('--models',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    result=probe(a.series,json.loads(Path(a.models).read_text()))
    with Path(a.output).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
