"""Complete TRAIN games to audit tactical-start accounting, not candidate quality."""
import argparse
import json
import math
from pathlib import Path

import torch

from scripts.attack.evaluate_continuous_pool import load
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.continuous_tactical_starts import TacticalStartContinuousPushEnv, PRIMARY
from sim.train_attack_ppo import code_fingerprint

OPTIONS = dict(action_delay_ms=1350, max_ms=310000,
    continuous_profile=dict(cannon_center_only=1,action_cooldown_ms=0,tower_damage_all=1),
    tower_support_weight=.15,spell_waste_weight=.1,tactical_start_fraction=1.)


def audit(root, initial, profile=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    net=load(dict(file=str(Path(initial).absolute()),sha256=file_sha(initial),scalars=309),Path('/'),'cpu')
    fingerprint=code_fingerprint('joint','continuous_v1',tower_support_weight=.15,
                                 spell_waste_weight=.1,tactical_start_fraction=1.)
    options=dict(OPTIONS)
    if profile:options['continuous_profile']=json.loads(Path(profile).read_text())
    env=TacticalStartContinuousPushEnv(**options)
    seeds={}
    for seed in range(844000,844100):
        env.reset(seed=seed,scenario_id='defense::air/push')
        seeds.setdefault(env.tactical_start['own_princess_mask'],seed)
        if len(seeds)==4:break
    if len(seeds)!=4:raise ValueError('Tower-state coverage missing')
    rows=[]
    with (root/'episodes.jsonl').open('x') as f:
        for family in PRIMARY:
            for expected_mask,seed in sorted(seeds.items()):
                obs,info=env.reset(seed=seed,scenario_id=f'defense::{family}/push')
                start=env.tactical_start;before=info['metrics']
                if start['own_princess_mask']!=expected_mask:raise ValueError('Setup mask changed across families')
                initial_bodies={e.uid for e in env.match.battle.entities.values() if e.alive and e.side==-1 and not e.is_tower}
                cleared_at=None;total=0.;steps=0
                while not env.done:
                    with torch.inference_mode():
                        logits,_=net(torch.from_numpy(obs['planes'])[None],torch.from_numpy(obs['scalars'])[None])
                        mask=torch.from_numpy(info['action_mask'])[None]
                        if not torch.isfinite(logits[mask]).all():raise ValueError('Nonfinite policy')
                        action=int(logits.masked_fill(~mask,-torch.inf).argmax(-1).item())
                    obs,reward,terminated,truncated,info=env.step(action)
                    total+=reward;steps+=1
                    alive={e.uid for e in env.match.battle.entities.values() if e.alive}
                    if cleared_at is None and not (initial_bodies & alive):cleared_at=env.match.elapsed_ms
                    if steps>1242:raise ValueError('Unbounded tactical episode')
                after=info['metrics'];p=env.profile
                expected=(p['enemy_hp_reward']*(after['enemy_fraction_damage']-before['enemy_fraction_damage'])
                    -p['own_hp_penalty']*(after['own_fraction_loss']-before['own_fraction_loss'])
                    -p['overflow_penalty']*(after['elixir_overflow']-before['elixir_overflow'])
                    -p['missed_fireball_penalty']*(after['missed_fireball_elixir']-before['missed_fireball_elixir'])
                    +p['fireball_value_reward']*(after['fireball_exchange_value']-before['fireball_exchange_value'])
                    +after['board_shaping_return']+after['reserve_shaping_return']
                    -.1*after['illegal_actions']+after['tower_support_reward']-after['empty_spell_penalty'])
                expected+=p['win_reward'] if after['outcome']=='win' else -p['loss_penalty'] if after['outcome']=='loss' else 0
                if not math.isclose(total,expected,rel_tol=1e-9,abs_tol=1e-7):
                    raise ValueError('Reward includes setup history or accounting mismatch')
                if after['illegal_actions'] or after['cannon_outside_center']:
                    raise ValueError('Rejected command or noncentral Cannon')
                if not terminated or truncated:raise ValueError('Game did not reach a game terminal')
                row=dict(family=family,seed=seed,start=start,steps=steps,episode_return=total,
                    reconstructed_return=expected,metrics=after,initial_body_ids_cleared_at_ms=cleared_at,
                    continued_after_initial_bodies_died=cleared_at is not None and env.match.elapsed_ms>cleared_at,
                    enemy_plays_after_setup=len(env.enemy_actions)-len(start['enemy_setup']))
                rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                print(json.dumps(dict(completed=len(rows),total=24,family=family,outcome=after['outcome'])),flush=True)
    if fingerprint!=code_fingerprint('joint','continuous_v1',tower_support_weight=.15,
                                     spell_waste_weight=.1,tactical_start_fraction=1.):
        raise ValueError('Source changed during audit')
    result=dict(schema='tactical-full-game-accounting-v1',source_code_sha256=fingerprint,
        initial_sha256=file_sha(initial),source_sha256=file_sha(__file__),
        episodes_sha256=file_sha(root/'episodes.jsonl'),games=len(rows),seed_by_mask=seeds,
        continued_after_initial_bodies_died=sum(r['continued_after_initial_bodies_died'] for r in rows),
        games_with_further_enemy_plays=sum(r['enemy_plays_after_setup']>0 for r in rows),
        reward_accounting_verified=True,quality_accepted=False,options=options,
        limitations=['TRAIN technical games with only four correlated initial seeds.',
            'Cleared initial body IDs excludes later children spawned by those bodies.',
            'No training or checkpoint promotion; synthetic states are not real-history reconstructions.'])
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--initial',required=True)
    p.add_argument('--profile')
    a=p.parse_args();audit(a.root,a.initial,a.profile)
