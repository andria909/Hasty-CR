"""Read-only exact-input decision audit for already completed live matches."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.gamestate.replay_decisions import load_capture
from scripts.gamestate.model_bridge import MemoryPolicy


def action_card(action, hand):
    if action == 0:return 'wait'
    if not 1 <= action < 2305:raise ValueError('Unsupported ability in captured decision')
    return hand[(action-1)//576]


def audit(run, checkpoint, matches):
    run, checkpoint = Path(run), Path(checkpoint)
    raw_events=(run/'events.jsonl').read_bytes()
    events=[json.loads(line)for line in raw_events.splitlines()]
    ended={e['match']for e in events if e['event']=='MATCH_END'}
    if not matches or not set(matches)<=ended:raise ValueError('Only completed matches may be audited')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    policy=MemoryPolicy(checkpoint);torch=policy.torch;torch.set_num_threads(1)
    rows=[];files={}
    for path in sorted((run/'decisions').glob('*.npz')):
        # A live writer may still be writing the newest *other* match. Only
        # inspect files older than the requested matches' terminal timestamp.
        end_ns=max(e['host_ns']for e in events if e['event']=='MATCH_END'and e['match']in matches)
        if path.stat().st_mtime_ns>end_ns:continue
        planes,scalars,mask,meta=load_capture(path,joint=policy.joint)
        if meta['match']not in matches:continue
        if meta['checkpoint_sha256']!=digest:raise ValueError('Live model SHA differs')
        if any(meta[k]!=getattr(policy,k)for k in ('geometry','coordinate_mode')):raise ValueError('Coordinate contract differs')
        hand=meta['hand']
        if len(hand)!=4 or len(set(hand))!=4:raise ValueError('Invalid captured hand')
        with torch.inference_mode():
            logits,value=policy.network(torch.from_numpy(planes)[None],torch.from_numpy(scalars)[None])
            if not torch.isfinite(logits).all()or not torch.isfinite(value).all():raise ValueError('Nonfinite inference')
            logits[0,~torch.from_numpy(mask)]=-torch.inf
            choice=int(logits.argmax(-1)[0]);probs=logits[0].softmax(-1).numpy()
        legal={card:int(mask[1+slot*576:1+(slot+1)*576].sum())for slot,card in enumerate(hand)}
        selected=action_card(choice,hand)
        rows.append(dict(match=meta['match'],tick=meta['tick'],hand=hand,legal_cells=legal,
            available_elixir=float(scalars[0])*10,reserved_elixir=meta['reserved_elixir'],
            pending_slots=meta['pending_slots'],selected=selected,
            recorded_action=meta['decision']['action'],replayed_action=choice,
            same_action=choice==meta['decision']['action'],
            value_error=abs(float(value[0])-meta['decision']['value']),
            wait_probability=float(probs[0]),
            can_play=bool(mask[1:2305].any()),
            slot_probability={card:float(probs[1+slot*576:1+(slot+1)*576].sum())for slot,card in enumerate(hand)}))
        files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not rows:raise ValueError('No captures for completed matches')
    summaries=[]
    for match in matches:
        samples=[r for r in rows if r['match']==match]
        ack=[e for e in events if e.get('match')==match and e['event']=='PLAY_ACK']
        plays=[e for e in events if e.get('match')==match and e['event']=='PLAY']
        timings={}
        for field in ('parser_ms','inference_ms','parser_request_to_input_return_ms'):
            values=[e['timing'][field]for e in plays]
            timings[field]=dict(n=len(values),mean=float(np.mean(values)),p95=float(np.percentile(values,95)))if values else None
        summaries.append(dict(match=match,samples=len(samples),same_actions=sum(r['same_action']for r in samples),
            max_value_error=max(r['value_error']for r in samples),
            confirmed_card_counts=dict(Counter(e['card']for e in ack)),
            attempted_card_counts=dict(Counter(e['card']for e in plays)),
            captured_policy_choices=dict(Counter(r['selected']for r in samples)),
            decisions_with_play_available=sum(r['can_play']for r in samples),
            voluntary_waits=sum(r['can_play']and r['selected']=='wait'for r in samples),
            available_elixir_below4=sum(r['available_elixir']<4 for r in samples),
            expensive_cards={card:dict(in_hand=sum(card in r['hand']for r in samples),
                legal=sum(r['legal_cells'].get(card,0)>0 for r in samples),
                chosen=sum(r['selected']==card for r in samples))for card in ('hog_rider','musketeer','fireball')},
            timings_ms=timings))
    return dict(schema='completed-live-decision-audit-v1',run=str(run),checkpoint_sha256=digest,
        events_prefix_bytes=len(raw_events),events_prefix_sha256=hashlib.sha256(raw_events).hexdigest(),capture_sha256=files,
        summaries=summaries,rows=rows,training_performed=False,
        limitation='Captured every-N policy calls, not independent states or all placements. Exact replay validates choice reproduction, not parser/physics correctness. No counterfactual gameplay or tactical success labels.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--matches',type=int,nargs='+',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.run,args.checkpoint,args.matches)
    with Path(args.output).open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(summaries=result['summaries'])))
