"""Paired TRAIN continuations: cheap play versus saving for a Musketeer.

One intervention per complete game. Opponent, queued commands, mana and RNG
are copied with the simulator. This is a diagnosis, not a deployable policy.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle

import torch

from scripts.attack.collect_bank_match_states import checked_snapshot
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.continuous_push import CONTRACT, DECKS
from sim.train_attack_ppo import initialize_attack_policy, make_training_env


def decision(net, env, card=None):
    obs=env._observe();mask=torch.as_tensor(env._mask()).clone()
    if card is not None:
        allowed=torch.zeros_like(mask)
        for slot,name in enumerate(env.match.players[1].hand):
            if name==card:allowed[1+slot*576:1+(slot+1)*576]=True
        mask &= allowed
        if not mask.any():return None
    with torch.inference_mode():
        logits,_=net(torch.as_tensor(obs['planes'])[None],torch.as_tensor(obs['scalars'])[None])
        if not torch.isfinite(logits[0,mask]).all():raise ValueError('Nonfinite actor')
        return int(logits[0].masked_fill(~mask,-torch.inf).argmax())


def candidate(env, action):
    if action==0 or env.match.elapsed_ms<5000:return False
    hand=env.match.players[1].hand
    if 'musketeer' not in hand:return False
    card=hand[(action-1)//576]
    available=env.match.players[1].elixir/1000-sum(env.cards[e[1]].cost for e in env.pending)
    return (card in ('skeletons','electro_spirit','zap') and available<4
            and any(e.alive and e.side==-1 and not e.is_tower for e in env.match.battle.entities.values()))


def continuation(payload, net, mode):
    env=pickle.loads(payload)
    start=env.match.elapsed_ms;before=env._metrics();commands=len(env.actions)
    forced=None;wait_steps=0;released_at=None
    while not env.done:
        action=decision(net,env)
        if mode!='baseline' and released_at is None:
            musk=decision(net,env,'musketeer')
            if musk is not None or env.match.elapsed_ms-start>=10000:
                released_at=env.match.elapsed_ms
                if mode=='reserve_then_musketeer' and musk is not None:
                    action=musk;forced=musk
            else:action=0;wait_steps+=1
        env.step(action)
    after=env._metrics()
    return dict(mode=mode,metrics=after,wait_steps=wait_steps,released_at_ms=released_at,
                forced_musketeer_action=forced,terminal=True,
                own_loss_delta=after['own_fraction_loss']-before['own_fraction_loss'],
                enemy_damage_delta=after['enemy_fraction_damage']-before['enemy_fraction_damage'],
                accepted_cards=dict(Counter(c['card'] for c in env.actions[commands:])),
                spent_delta=after['elixir_spent']-before['elixir_spent'],
                illegal_delta=after['illegal_actions']-before['illegal_actions'])


def run(collected, output, families=None):
    torch.set_num_threads(1);collected=Path(collected);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    pins=physics_pins();source=file_sha(__file__)
    protocol=dict(schema='cheap-cycle-paired-train-v1',physics_pins=pins,source_sha256=source,
        families=families or sorted(DECKS),seeds=[462,463],train_seed_base=202611400,
        modes=['baseline','reserve_only','reserve_then_musketeer'],wait_limit_ms=10000,
        scope='First cheap action with Musketeer in hand, available mana<4, visible enemy. '
              'Full TRAIN continuations with original policy thereafter; not an optimal defense.',
        live_games=False,training_changed=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'driver.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    for replica in protocol['seeds']:
        selected=collected/f'pilot_control03_s{replica}_summary.json'
        selection=json.loads(selected.read_text());weights=collected/selection['checkpoint']
        if file_sha(weights)!=selection['checkpoint_sha256']:raise ValueError('Selected weights changed')
        cfgpath=weights.parent/'config.json';cfg=json.loads(cfgpath.read_text())
        keys=('curriculum','continuous_profile','validation_selection','continuous_focus','geometry',
              'coordinate_mode','max_ms','tower_support_weight','spell_waste_weight',
              'crown_only_weight','dormant_king_weight')
        options={k:cfg[k] for k in keys};options['action_delay_ms']=1350
        if options['continuous_profile']['hog_sight_clipping']!=1:raise ValueError('Current physics required')
        net,_,_=initialize_attack_policy(torch.load(weights,map_location='cpu',weights_only=False),
            device='cpu',num_planes=20,num_scalars=309,observation_contract=CONTRACT)
        net.eval()
        for index,family in enumerate(protocol['families']):
            seed=protocol['train_seed_base']+sorted(DECKS).index(family)
            scenario=f'defense::{family}/push'
            env=make_training_env(task_mode='joint',split='train',seed=seed,**options)
            env.reset(seed=seed,scenario_id=scenario);saved=None;state=None
            while not env.done:
                action=decision(net,env)
                if saved is None and candidate(env,action):
                    saved=checked_snapshot(env)
                    state=dict(at_ms=env.match.elapsed_ms,action=action,
                        selected_card=env.match.players[1].hand[(action-1)//576],
                        hand=list(env.match.players[1].hand),pending=len(env.pending),
                        native_elixir=env.match.players[1].elixir/1000,
                        available_elixir=env.match.players[1].elixir/1000-sum(env.cards[e[1]].cost for e in env.pending))
                env.step(action)
            row=dict(replica=replica,scenario=scenario,split='train',episode_seed=seed,state=state,
                weights=str(weights),checkpoint_sha256=file_sha(weights),config_sha256=file_sha(cfgpath),
                original_metrics=env._metrics(),branches=[])
            if saved is not None:
                snapshot=output/f's{replica}_{family}.pickle';snapshot.write_bytes(saved)
                row.update(snapshot=snapshot.name,snapshot_sha256=file_sha(snapshot))
                for mode in protocol['modes']:
                    branch=continuation(saved,net,mode)
                    if mode=='baseline' and branch['metrics']!=row['original_metrics']:
                        raise ValueError('Copied baseline failed exact terminal replay')
                    if branch['illegal_delta']:raise ValueError('Intervention spent unavailable mana')
                    row['branches'].append(branch)
            results.append(row)
            (output/f's{replica}_{family}.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps(dict(done=len(results),total=len(protocol['families'])*2,
                family=family,replica=replica,captured=state is not None,
                outcomes=[b['metrics']['outcome'] for b in row['branches']])),flush=True)
    if pins!=physics_pins() or source!=file_sha(__file__):raise ValueError('Diagnostic source changed')
    result=dict(protocol=protocol,complete=True,records=results)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collected',required=True);p.add_argument('--output',required=True)
    p.add_argument('--families',nargs='+',choices=sorted(DECKS))
    a=p.parse_args();run(a.collected,a.output,a.families)
