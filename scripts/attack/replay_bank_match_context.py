"""Independently execute selected counterfactual commands through the real env."""
import argparse
import json
from pathlib import Path
import pickle

import torch

from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.train_attack_ppo import initialize_attack_policy,tensor_observations,action_masks


def replay(context,output):
    torch.set_num_threads(1);context=Path(context);saved=json.loads(context.read_text())
    protocol=json.loads((context.parent/'protocol.json').read_text());root=Path(protocol['root'])
    if (protocol['physics_pins']!=physics_pins()or
            file_sha(root/'manifest.json')!=protocol['manifest_sha256']or
            file_sha(context.parent/'driver.py')!=protocol['source_sha256']):
        raise ValueError('Pinned current search required')
    m=json.loads((root/'manifest.json').read_text());state=m['records'][saved['index']]
    if state!=saved['state']or file_sha(root/state['snapshot'])!=state['sha256']or state['split']!='train':
        raise ValueError('Snapshot differs')
    if file_sha(m['weights'])!=protocol['weights_sha256']:raise ValueError('Continuation checkpoint changed')
    net,_,_=initialize_attack_policy(torch.load(m['weights'],map_location='cpu',weights_only=False),device='cpu',
        num_planes=20,num_scalars=309,observation_contract='joint_public_presence_continuous_slots_v2')
    net.eval();rows=[]
    wanted={0,*saved['successful_actions']}
    trials=[r for r in saved['finals']if r['action']in wanted]
    if len(trials)!=len(wanted)*2:raise ValueError('Both delays and baseline required')
    for trial in trials:
        env=pickle.loads((root/state['snapshot']).read_bytes())
        env.action_delay_ms=trial['delay_ms'];initial_hp=sum(t.hitpoints for t in env.match.towers[1].values())
        if env.pending or not env._mask()[trial['action']]:raise ValueError('Initial command unavailable')
        obs,_,terminal,truncated,info=env.step(trial['action'])
        with torch.inference_mode():
            while not (terminal or truncated):
                logits,_=net(*tensor_observations([obs],'cpu'))
                action=int(logits.masked_fill(~action_masks([info],'cpu'),-torch.inf).argmax(-1).item())
                obs,_,terminal,truncated,info=env.step(action)
        damage=initial_hp-sum(t.hitpoints for t in env.match.towers[1].values())
        if info['metrics']!=trial['metrics']or damage!=trial['own_damage']:
            raise ValueError('Independent full-match replay differs')
        rows.append(dict(action=trial['action'],delay_ms=trial['delay_ms'],outcome=info['metrics']['outcome'],
            own_damage=damage,all_terminal_metrics_exact=True))
    if protocol['physics_pins']!=physics_pins():raise ValueError('Physics changed during replay')
    result=dict(schema='bank-match-context-independent-replay-v1',context=str(context),
        context_sha256=file_sha(context),source_sha256=file_sha(__file__),index=saved['index'],
        fold=state['fold'],replayed=rows,labels_ready=False,full_training_accepted=False)
    Path(output).write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=replay(a.context,a.output);print(json.dumps(dict(replayed=len(r['replayed']),exact=True)))
