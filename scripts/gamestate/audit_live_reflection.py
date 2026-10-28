"""Read-only mirror probe on completed live frames, away from pending inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.gamestate.memory_play import own_context
from scripts.gamestate.model_bridge import MemoryPolicy,NativeModelAdapter,MissingModelInput
from sim.lane_reflection import reflect_observation,reflect_actions
from sim.env import ACTIONS,ClashEnv


def audit(root):
    root=Path(root);config=json.loads((root/'config.json').read_text())
    series=json.loads((root/'full_bc800/series/series.json').read_text());samples=[]
    for run in series['runs']:
        events=[json.loads(x) for x in (Path(run)/'events.jsonl').read_text().splitlines()]
        complete={m['local_match'] for m in series['matches'] if m['run']==run}
        starts={e['match']:e for e in events if e['event']=='MATCH_START' and e['match'] in complete}
        ends={e['match']:e for e in events if e['event']=='MATCH_END' and e['match'] in complete}
        last={};counts={}
        for line in (Path(run)/'frames.jsonl').open():
            frame=json.loads(line);stamp=frame['received_host_ns']
            if frame['status']!='ok' or frame.get('own_state',{}).get('status')!='ok':continue
            match=next((n for n,s in starts.items() if s['host_ns']<=stamp<=ends[n]['host_ns']),None)
            if match is None or counts.get(match,0)>=20 or stamp-last.get(match,0)<5e9:continue
            tick=frame['own_state']['tick']
            previous=[e for e in events if e.get('match')==match and e.get('tick',tick)>=0 and e.get('tick',tick)<tick]
            plays=[e for e in previous if e['event']=='PLAY']
            acknowledgements={(e['card'],e['source_tick']) for e in events
                if e.get('match')==match and e['event']=='PLAY_ACK' and e['tick']<=tick}
            # Omit ambiguous timeout/rejection histories instead of reconstructing reserved elixir.
            if any((e['card'],e['tick']) not in acknowledgements for e in plays):continue
            towers=[u for u in frame['units'] if u['name'] in ('PrincessTower','KingTower') and u['hp']>0]
            if len(towers)!=6:continue
            samples.append((run,match,frame));counts[match]=counts.get(match,0)+1;last[match]=stamp
    if not samples:raise ValueError('No qualifying recorded states')
    torch.set_num_threads(1);results=[]
    for model in config['models']:
        if hashlib.sha256(Path(model['checkpoint']).read_bytes()).hexdigest()!=model['sha256']:raise ValueError('Model changed')
        policy=MemoryPolicy(Path(model['checkpoint']));rows=[]
        for run,match,frame in samples:
            adapter=NativeModelAdapter(own_side=frame['own_state']['own_side'],allow_incomplete_history=True,
                allow_unknown_profiles=True,coordinate_mode=policy.coordinate_mode,geometry=policy.geometry)
            try:view=adapter.encode(frame,own_context(frame['own_state']))
            except MissingModelInput:continue
            if not view.metadata['ready_for_inference'] or not view.action_mask[1:].any():continue
            policy.validate_view_contract(view)
            observation=dict(view.observation)
            observation['scalars']=np.concatenate((observation['scalars'],np.array([0.,0.,float(not view.metadata['enemy_bodies_present'])],dtype=np.float32)))
            mirrored=reflect_observation(observation);mask=view.action_mask
            masks=np.stack((mask,reflect_actions(mask)))
            with torch.inference_mode():
                logits,_=policy.network(torch.from_numpy(np.stack((observation['planes'],mirrored['planes']))),
                    torch.from_numpy(np.stack((observation['scalars'],mirrored['scalars']))))
                logits=logits.masked_fill(~torch.from_numpy(masks),-torch.inf)
                actions=logits.argmax(-1).tolist()
            placements=[ClashEnv.decode(a) if a else None for a in actions]
            remap=reflect_actions(np.arange(ACTIONS))
            rows.append(dict(run=run,match=match,tick=frame['own_state']['tick'],enemy_bodies_present=bool(view.metadata['enemy_bodies_present']),actions=actions,placements=placements,
                mirror_equivariant=actions[0]==int(remap[actions[1]])))
        active=[r for r in rows if all(p is not None for p in r['placements'])]
        results.append(dict(model=model['label'],frames=len(rows),both_place=len(active),
            original_left=sum(r['placements'][0][1]<9 for r in active),
            mirrored_left=sum(r['placements'][1][1]<9 for r in active),
            same_screen_half=sum((r['placements'][0][1]<9)==(r['placements'][1][1]<9) for r in active),
            exact_mirror_actions=sum(r['mirror_equivariant'] for r in rows),rows=rows))
    return dict(schema='live-reflection-readonly-v1',sampled_frames=len(samples),results=results,
        limitations=['Completed BC800 games only; not representative independent test',
            'All prior-tick inputs acknowledged; full tower set; native history remains incomplete',
            'Reflection probes sensitivity, not optimal action or tactical correctness',
            'No online ensemble or policy modifications; second model tested on first-model states'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=audit(a.root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in x.items() if k!='rows'} for x in r['results']]))
