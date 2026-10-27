"""TRAIN-only diagnostic of mirrored policy averaging; no live/model changes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from scripts.attack.evaluate_continuous_pool import load,sha
from scripts.attack.evaluate_full_candidates import summarize_test
from sim.continuous_push import ContinuousPushEnv,PushCase,DECKS
from sim.env import ACTIONS
from sim.lane_reflection import reflect_observation,reflect_actions


def probabilities(network,observation,mask,ensemble):
    observations=[observation,reflect_observation(observation)] if ensemble else [observation]
    with torch.inference_mode():
        logits,_=network(torch.from_numpy(np.stack([o['planes'] for o in observations])),
                         torch.from_numpy(np.stack([o['scalars'] for o in observations])))
        if ensemble:
            indices=torch.from_numpy(reflect_actions(np.arange(ACTIONS)))
            logits=torch.stack((logits[0],logits[1][indices]))
        support=torch.from_numpy(mask)
        if not torch.isfinite(logits[:,support]).all():raise ValueError('Invalid legal policy logits')
        return torch.softmax(logits.masked_fill(~support[None],-torch.inf),dim=1).mean(dim=0).numpy()


def run(checkpoint,digest,output):
    checkpoint,output=Path(checkpoint),Path(output)
    if sha(checkpoint)!=digest:raise ValueError('Model differs')
    protocol=dict(schema='reflection-ensemble-train-diagnostic-v1',checkpoint=str(checkpoint),sha256=digest,
        split='train',seeds=[3810101,3810102],tasks=['attack','defense'],families=list(DECKS),style='push',
        action_delay_ms=1350,modes=['plain','mirror_average'],reflected_views=[False,True],games=96,
        source_sha256=sha(__file__),independent_test_run=False,live_games_started=False,quality_accepted=False)
    output.mkdir(parents=True,exist_ok=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.set_num_threads(1)
    entry=dict(file=str(checkpoint.absolute()),sha256=digest,scalars=309,mode='joint_enemy_presence_v1')
    network=load(entry,Path('.'),'cpu');rows=[];timings={m:[] for m in protocol['modes']}
    with(output/'games.jsonl').open('x')as stream:
        for mode in protocol['modes']:
            for reflected in protocol['reflected_views']:
                for task in protocol['tasks']:
                    for family in protocol['families']:
                        for seed in protocol['seeds']:
                            env=ContinuousPushEnv(split='train',action_delay_ms=1350,max_ms=310000)
                            case=PushCase(task,family,'push','train');env.scenarios=[case]
                            env.reset(seed=seed,scenario_id=case.scenario_id);env.reflected=reflected
                            ties=decisions=0
                            for _ in range(1242):
                                mask=env._mask()
                                if mask[1:].any():
                                    start=time.perf_counter_ns();p=probabilities(network,env._observe(),mask,mode=='mirror_average')
                                    timings[mode].append((time.perf_counter_ns()-start)/1e6)
                                    ties+=int(np.count_nonzero(p==p.max())>1);decisions+=1;action=int(p.argmax())
                                else:action=0
                                _,_,done,truncated,info=env.step(action)
                                if done or truncated:break
                            else:raise ValueError('Finite game did not finish')
                            row=dict(info['metrics'],mode=mode,reflected_view=reflected,seed=seed,
                                     greedy_ties=ties,inference_decisions=decisions)
                            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                            print(json.dumps(dict(completed=len(rows),total=protocol['games'],mode=mode)),flush=True)
    summaries={}
    for mode in protocol['modes']:
        selected=[r for r in rows if r['mode']==mode]
        s=summarize_test(selected);s['lanes']={lane:sum(sum(r['accepted_card_lane_counts'][lane].values()) for r in selected) for lane in ('left','right')}
        s['by_reflected_view']={str(v):summarize_test([r for r in selected if r['reflected_view']==v]) for v in (False,True)}
        s['inference_ms']=dict(mean=float(np.mean(timings[mode])),p95=float(np.quantile(timings[mode],.95)))
        s['greedy_ties']=sum(r['greedy_ties'] for r in selected);summaries[mode]=s
    result=dict(protocol=protocol,summaries=summaries,rows=rows,
        limitations=['TRAIN development only, not model acceptance or live win-rate',
                    'Reflected views share identical initial games; outcomes are correlated',
                    'Symmetric probabilities do not prove tactical quality; argmax ties can break symmetry',
                    'Fixed1350ms delay is a measured approximation, not full live execution'])
    with(output/'result.json').open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('checkpoint','sha256','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();run(args.checkpoint,args.sha256,args.output)
