"""TRAIN-only counterfactual: vary ONLY the legacy attack/defense scalar.

No optimization, action override, live game or claim of optimal Hog use.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sim.continuous_push import ContinuousPushEnv,CONTRACT
from sim.env import DECK_26
from sim.train_attack_ppo import initialize_attack_policy


def states():
    env=ContinuousPushEnv(split='train');planes=[];scalars=[];masks=[];records=[]
    for index in range(300):
        seed=5900000+index;case=env.scenarios[index%len(env.scenarios)]
        obs,info=env.reset(seed=seed,scenario_id=case.scenario_id);env.match.players[1].elixir=10000
        for _ in range(16):obs,_,_,_,info=env.step(0)
        hand=obs['scalars'][7:39].reshape(4,8);slots=np.flatnonzero(hand[:,DECK_26.index('hog_rider')])
        if not len(slots):continue
        slot=int(slots[0]);mask=info['action_mask']
        if not mask[1+slot*576:1+(slot+1)*576].any()or obs['scalars'][-1]!=0:continue
        planes.append(obs['planes']);scalars.append(obs['scalars']);masks.append(mask)
        records.append(dict(seed=seed,scenario_id=case.scenario_id,hog_slot=slot,elapsed_ms=env.match.elapsed_ms))
        if len(records)==64:break
    if len(records)!=64:raise ValueError('64 TRAIN frames with Hog legal and enemies present required')
    return np.stack(planes),np.stack(scalars),np.stack(masks),records


def probe(weights,output):
    output=Path(output)
    if output.exists():raise ValueError('Preserve earlier probe')
    torch.set_num_threads(1);planes,scalars,masks,records=states();models=[]
    mask=torch.as_tensor(masks);hog=torch.zeros_like(mask)
    for index,row in enumerate(records):hog[index,1+row['hog_slot']*576:1+(row['hog_slot']+1)*576]=mask[index,1+row['hog_slot']*576:1+(row['hog_slot']+1)*576]
    hands=scalars[:,7:39].reshape(-1,4,8)
    for path in map(Path,weights):
        blob=torch.load(path,map_location='cpu',weights_only=False)
        net,_,_=initialize_attack_policy(blob,device='cpu',num_scalars=309,observation_contract=CONTRACT);net.eval();rows=[]
        for feature in(0.,.25,.5,.75,1.):
            alternative=scalars.copy();alternative[:,-1]=feature
            with torch.inference_mode():logits,_=net(torch.as_tensor(planes),torch.as_tensor(alternative))
            logits=logits.double();choice=logits.masked_fill(~mask,-torch.inf).argmax(-1)
            labels=['hold'if int(a)==0 else DECK_26[int(hands[i,(int(a)-1)//576].argmax())]for i,a in enumerate(choice)]
            logmass=torch.logsumexp(logits.masked_fill(~hog,-torch.inf),-1)-torch.logsumexp(logits.masked_fill(~mask,-torch.inf),-1)
            rows.append(dict(legacy_task_scalar=feature,greedy_cards=dict(Counter(labels)),mean_log_hog_probability=float(logmass.mean()),
                mean_hog_probability=float(logmass.exp().mean()),choices=choice.tolist()))
        models.append(dict(checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows))
    result=dict(schema='train-legacy-task-scalar-probe-v1',split='train_counterfactual',frames=records,models=models,
        fixed_public_inputs_sha256=hashlib.sha256(planes.tobytes()+scalars[:,:-1].tobytes()+masks.tobytes()).hexdigest(),
        invariant='Same planes, first308scalars and legal masks; only last legacy task scalar changed',
        training_performed=False,live_games_started=False,quality_accepted=False,
        limitations=['Artificial10-elixir TRAIN fixtures after4seconds of paid enemy play',
            'Sensitivity to a feature is not proof that the alternative action is correct',
            'Counterfactual task values are not proposed hidden information or automatic deployment changes'])
    output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();probe(a.weights,a.output)
