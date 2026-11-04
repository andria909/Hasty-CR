"""Compare frozen policies' joint MAP, card-marginal MAP and sampling on TRAIN games."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.continuous_push import CONTRACT
from sim.train_attack_ppo import initialize_attack_policy,make_training_env

MODES=('joint_greedy','card_greedy','sample')


def decode(logits,mask,mode,generator=None):
    if mode not in MODES or logits.ndim!=1 or len(logits)<2305 or mask.shape!=logits.shape or not bool(mask.any()):
        raise ValueError('Expected WAIT, four576-cell cards, optional abilities and known decoder')
    if not torch.isfinite(logits[mask]).all():raise ValueError('Nonfinite legal logits')
    masked=logits.masked_fill(~mask,-torch.inf)
    groups=torch.cat((masked[:1],torch.logsumexp(masked[1:2305].reshape(4,576),dim=1),masked[2305:]))
    group=int(groups.argmax());joint=int(masked.argmax())
    grouped=(0 if group==0 else 2305+group-5 if group>=5 else
             1+(group-1)*576+int(masked[1+(group-1)*576:1+group*576].argmax()))
    if mode=='joint_greedy':action=joint
    elif mode=='card_greedy':action=grouped
    else:
        if generator is None:raise ValueError('Dedicated sampling generator required')
        action=int(torch.multinomial(torch.softmax(masked,dim=0),1,generator=generator))
    if not mask[action]:raise ValueError('Decoder chose masked action')
    return action,dict(joint_action=joint,card_action=grouped,group_probabilities=torch.softmax(groups,dim=0).tolist())


def run(source,output):
    source,output=Path(source),Path(output);original=json.loads(source.read_text())
    if not original.get('complete')or original['protocol']['physics_pins']!=physics_pins():
        raise ValueError('Completed TRAIN baseline with current physics required')
    if any(r['split']!='train'for r in original['records']):raise ValueError('TRAIN contexts only')
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    protocol=dict(schema='paired-action-decoding-train-v1',source=str(source),source_sha256=file_sha(source),
        physics_pins=physics_pins(),driver_sha256=file_sha(__file__),modes=MODES,
        stochastic_seed_formula='2026110400 + replica*100 + context_index',
        live_games=False,training_modified=False,checkpoint_promotion=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'driver.py').write_bytes(Path(__file__).read_bytes())
    cache={};games=[]
    for index,case in enumerate(original['records']):
        weights=Path(case['weights']);config_path=weights.parent/'config.json'
        if file_sha(weights)!=case['checkpoint_sha256']or file_sha(config_path)!=case['config_sha256']:
            raise ValueError('Baseline policy/config changed')
        cfg=json.loads(config_path.read_text())
        if cfg.get('policy_temperature',1.)!=1. or cfg.get('card_exploration_epsilon',0.)!=0.:
            raise ValueError('This probe requires original unit-temperature/no-epsilon distribution')
        if str(weights)not in cache:
            net,_,_=initialize_attack_policy(torch.load(weights,map_location='cpu',weights_only=False),device='cpu',
                num_planes=20,num_scalars=309,observation_contract=CONTRACT)
            cache[str(weights)]=net.eval()
        net=cache[str(weights)]
        options={k:cfg[k]for k in ('curriculum','continuous_profile','validation_selection','continuous_focus',
            'geometry','coordinate_mode','max_ms','tower_support_weight','spell_waste_weight','crown_only_weight','dormant_king_weight')}
        options['action_delay_ms']=1350
        for mode in MODES:
            env=make_training_env(task_mode='joint',split='train',seed=case['episode_seed'],**options)
            obs,_=env.reset(seed=case['episode_seed'],scenario_id=case['scenario'])
            seed=2026110400+case['replica']*100+index;generator=torch.Generator().manual_seed(seed)
            counts=Counter();examples=[];total=0.
            while not env.done:
                mask=torch.as_tensor(env._mask())
                with torch.inference_mode():
                    logits,_=net(torch.as_tensor(obs['planes'])[None],torch.as_tensor(obs['scalars'])[None])
                    action,detail=decode(logits[0],mask,mode,generator)
                group_of=lambda a:0 if a==0 else 5+a-2305 if a>=2305 else 1+(a-1)//576
                counts['decisions']+=1;counts['choices']+=int(mask.sum()>1)
                different=group_of(detail['joint_action'])!=group_of(detail['card_action'])
                counts['joint_vs_card_different_group']+=different
                if different and len(examples)<5:
                    examples.append(dict(at_ms=env.match.elapsed_ms,hand=list(env.match.players[1].hand),
                        available_elixir=float(obs['scalars'][0])*10,native_elixir=env.match.players[1].elixir/1000,
                        legal_cells_by_slot=mask[1:2305].reshape(4,576).sum(1).tolist(),**detail))
                obs,reward,*_=env.step(action);total+=reward
            metrics=env._metrics();baseline_exact=None
            if mode=='joint_greedy':
                baseline_exact=metrics==case['original_metrics']
                if not baseline_exact:raise ValueError('Legacy joint-greedy baseline failed exact replay')
            if metrics['illegal_actions']:raise ValueError('Decoder issued illegal command')
            row=dict(replica=case['replica'],scenario=case['scenario'],episode_seed=case['episode_seed'],
                mode=mode,sampling_seed=seed if mode=='sample'else None,checkpoint_sha256=case['checkpoint_sha256'],
                baseline_exact=baseline_exact,metrics=metrics,episode_return=total,counts=dict(counts),examples=examples)
            games.append(row);(output/f'game_{len(games):03d}.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps(dict(game=len(games),total=3*len(original['records']),mode=mode,replica=case['replica'],
                scenario=case['scenario'],outcome=metrics['outcome'])),flush=True)
    if protocol['physics_pins']!=physics_pins()or protocol['driver_sha256']!=file_sha(__file__):
        raise ValueError('Diagnostic changed during execution')
    result=dict(protocol=protocol,complete=True,games=games)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
