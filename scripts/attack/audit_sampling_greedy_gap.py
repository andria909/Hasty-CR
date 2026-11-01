"""Measure sampling versus greedy action groups on pinned diagnostic states.

States were selected when the old policy waited with defense available. They
are intentionally biased diagnostics, not estimates of whole-match frequency.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle

import torch

from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.continuous_push import CONTRACT
from sim.train_attack_ppo import initialize_attack_policy, training_distribution, tempered_logits
from sim.train_defense_ppo import tensor_observations, action_masks


def group_metrics(probabilities, masks):
    if probabilities.ndim != 2 or probabilities.shape != masks.shape or probabilities.shape[1] < 2305:
        raise ValueError('Complete four-slot action space required')
    if masks[:, 2305:].any():
        raise ValueError('This diagnostic requires the fixed deck without abilities')
    if not torch.isfinite(probabilities).all() or not torch.allclose(probabilities.sum(-1),torch.ones_like(probabilities[:,0])):
        raise ValueError('Normalized finite probabilities required')
    if (probabilities[~masks] != 0).any() or (probabilities < 0).any():
        raise ValueError('Illegal action mass')
    slots=probabilities[:,1:2305].reshape(-1,4,576)
    mass=torch.cat((probabilities[:,:1],slots.sum(-1)),dim=-1)
    choice=probabilities.argmax(-1)
    greedy_group=torch.where(choice==0,0,1+(choice-1)//576)
    largest_group=mass.argmax(-1)
    def entropy(p):
        return -(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)
    full=entropy(probabilities);group=entropy(mass)
    return dict(group_mass=mass,greedy_action=choice,greedy_group=greedy_group,
                largest_group=largest_group,full_entropy=full,group_entropy=group,
                conditional_cell_entropy=full-group,
                legal_cells=masks[:,1:2305].reshape(-1,4,576).sum(-1))


def audit(states,prepared,output):
    states,prepared=Path(states),Path(prepared)
    manifest=json.loads((states/'manifest.json').read_text())
    prep=json.loads((prepared/'preparation.json').read_text())
    protocol=json.loads((prepared/'dataset/protocol.json').read_text())
    from scripts.kaggle.prepare_policy_adaptation_pilot import validate_protocol
    validate_protocol(protocol)
    if (manifest['physics_pins']!=physics_pins() or manifest['split']!='train'
            or len(manifest['records'])!=106 or len(prep['config']['models'])!=4):
        raise ValueError('Pinned 106-state diagnostic and four initializers required')
    trial=protocol['cases'][0]
    temperature=trial.get('policy_temperature',1.)
    epsilon=trial.get('card_exploration_epsilon',0.)
    observations=[];infos=[];hands=[]
    for row in manifest['records']:
        path=states/row['snapshot']
        if file_sha(path)!=row['sha256'] or row['action']!=0 or row['split']!='train':
            raise ValueError('Diagnostic snapshot changed')
        env=pickle.loads(path.read_bytes())
        if env.pending or env.match.players[1].hand!=row['hand']:
            raise ValueError('Snapshot execution/hand differs')
        observations.append(env._observe());infos.append(env._info());hands.append(row['hand'])
    torch.set_num_threads(1)
    planes,scalars=tensor_observations(observations,'cpu');masks=action_masks(infos,'cpu')
    reports=[]
    for model in prep['config']['models']:
        path=prepared/'dataset'/model['file']
        if file_sha(path)!=model['sha256']:raise ValueError('Initializer changed')
        net,_,_=initialize_attack_policy(torch.load(path,map_location='cpu',weights_only=False),
                    device='cpu',num_planes=20,num_scalars=309,observation_contract=CONTRACT)
        net.eval();rows=[]
        with torch.inference_mode():
            for start in range(0,len(observations),16):
                end=min(start+16,len(observations))
                logits,_=net(planes[start:end],scalars[start:end])
                dist=training_distribution(tempered_logits(logits,temperature),masks[start:end],epsilon)
                metrics=group_metrics(dist.probs,masks[start:end])
                actual_greedy=logits.masked_fill(~masks[start:end],-torch.inf).argmax(-1)
                for j in range(end-start):
                    i=start+j;groups=['wait',*hands[i]]
                    greedy=int(actual_greedy[j]);g=0 if greedy==0 else 1+(greedy-1)//576
                    mass=metrics['group_mass'][j].tolist()
                    rows.append(dict(snapshot=manifest['records'][i]['snapshot'],fold=manifest['records'][i]['fold'],
                        greedy_card=groups[g],greedy_action=greedy,
                        largest_mass_card=groups[int(metrics['largest_group'][j])],
                        probability_by_card=dict(zip(groups,mass)),
                        legal_cells=dict(zip(hands[i],metrics['legal_cells'][j].tolist())),
                        full_entropy=float(metrics['full_entropy'][j]),group_entropy=float(metrics['group_entropy'][j]),
                        conditional_cell_entropy=float(metrics['conditional_cell_entropy'][j])))
        counts=Counter(r['greedy_card'] for r in rows);mass=Counter()
        for r in rows:mass.update(r['probability_by_card'])
        reports.append(dict(id=model['id'],checkpoint_sha256=model['sha256'],states=len(rows),
            greedy_counts=dict(counts),expected_sampling_counts=dict(mass),
            greedy_wait_with_play_mass_above_90=sum(r['greedy_card']=='wait'and r['probability_by_card']['wait']<.1 for r in rows),
            greedy_card_differs_from_largest_group=sum(r['greedy_card']!=r['largest_mass_card']for r in rows),
            mean_entropy={k:sum(r[k]for r in rows)/len(rows)for k in ('full_entropy','group_entropy','conditional_cell_entropy')},rows=rows))
    result=dict(schema='sampling-greedy-gap-diagnostic-v1',manifest_sha256=file_sha(states/'manifest.json'),
        protocol_sha256=file_sha(prepared/'dataset/protocol.json'),source_sha256=file_sha(__file__),models=reports,
        temperature=temperature,card_exploration_epsilon=epsilon,physics_pins=physics_pins(),
        scope='106 reused TRAIN snapshots deliberately selected for old-policy WAIT with legal defense; not whole-game frequencies',
        live_games=False,policy_changed=False,training_changed=False)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('states','prepared','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();r=audit(a.states,a.prepared,a.output)
    print(json.dumps([{k:v for k,v in row.items()if k!='rows'}for row in r['models']],indent=2))
