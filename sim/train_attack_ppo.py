"""Finite attack PPO, explicit timer inputs and exact same-source resume."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from sim.reward_profiles import load_reward_profile,reward_objects

from sim.attack_env import AttackEnv, OBSERVATION_CONTRACT
from sim.attack_sampler import AttackScenarioCycle
from sim.env import NUM_PLANES, NUM_SCALARS, ACTIONS
from sim.train_ppo import (build_network, checkpoint_architecture, load_policy_state,
                           masked_distribution, COMPACT_ARCHITECTURE)
from sim.train_defense_ppo import (generalized_advantage,normalize_advantages,require_finite,
    tensor_observations,action_masks,rng_state,restore_rng,validate_transition)


def tempered_logits(logits,temperature=1.):
    if not np.isfinite(temperature)or not 0<temperature<=100:raise ValueError('Invalid policy temperature')
    return logits if temperature==1 else logits/temperature


def training_distribution(logits,masks,epsilon=0.):
    if epsilon:
        from sim.card_exploration import card_mixture_distribution
        return card_mixture_distribution(logits,masks,epsilon)
    return masked_distribution(logits,masks)


def build_attack_network(architecture=COMPACT_ARCHITECTURE,device='cpu',*,num_scalars=None,
                         num_planes=NUM_PLANES,num_actions=ACTIONS):
    return build_network(num_planes,NUM_SCALARS+2 if num_scalars is None else num_scalars,
                         num_actions,architecture).to(device)


def load_attack_state(network,blob,*,exact=False,observation_contract=OBSERVATION_CONTRACT):
    incoming=blob['state_dict']['scalar.0.weight'].shape[1]
    target=network.state_dict()['scalar.0.weight'].shape[1]
    if incoming not in (NUM_SCALARS,NUM_SCALARS+2,target) or incoming>target or (exact and incoming!=target):
        raise ValueError('Unsupported attack scalar input shape')
    if exact and blob.get('config',{}).get('observation_contract')!=observation_contract:
        raise ValueError('Attack observation contract missing/mismatched')
    source_planes=blob['state_dict']['conv.0.weight'].shape[1]
    target_planes=network.state_dict()['conv.0.weight'].shape[1]
    if source_planes>target_planes or (exact and source_planes!=target_planes):
        raise ValueError('Attack plane input shape differs')
    load_policy_state(network,blob['state_dict'])
    return dict(source_scalar_count=incoming,target_scalar_count=target,
        appended_scalar_columns=target-incoming,
        appended_timer_columns_zero_initialized=incoming==NUM_SCALARS,
        observation_contract=observation_contract,source_plane_count=source_planes,target_plane_count=target_planes,
        exact_weight_shape=incoming==target and source_planes==target_planes)


def initialize_attack_policy(blob=None,device='cpu',architecture=COMPACT_ARCHITECTURE,*,
                             num_scalars=None,num_planes=NUM_PLANES,num_actions=ACTIONS,
                             observation_contract=OBSERVATION_CONTRACT):
    architecture=checkpoint_architecture(blob)if blob is not None else architecture
    network=build_attack_network(architecture,device,num_scalars=num_scalars,
                                 num_planes=num_planes,num_actions=num_actions)
    metadata=load_attack_state(network,blob,observation_contract=observation_contract)if blob is not None else dict(
        fresh_initialization=True,source_scalar_count=None,
        target_scalar_count=NUM_SCALARS+2 if num_scalars is None else num_scalars,
        observation_contract=observation_contract)
    return network,architecture,metadata


def make_attack_env(*,split='train',seed=0,deadline_scale=1.,**options):
    """Shared PPO/GRPO preparation: scale all families, never select on test."""
    if split not in ('train','validation'):
        raise ValueError('Attack training/evaluation forbids test split')
    if not np.isfinite(deadline_scale) or deadline_scale<=0:
        raise ValueError('deadline-scale must be finite and positive')
    env=AttackEnv(seed=seed,split=split,**options)
    scaled=[]
    for scenario in env.scenarios:
        low,high=[round(value*deadline_scale)for value in scenario.deadline_ms_range]
        if not 50<=low<high or high+scenario.opportunity_start_ms>600000:
            raise ValueError('Scaled deadline outside finite engine bounds')
        scaled.append(replace(scenario,deadline_ms_range=(low,high)))
    env.scenarios=scaled
    return env


def make_training_env(*,task_mode='joint',split='train',seed=0,curriculum='classic',validation_selection='overall',continuous_focus='uniform',history_frames=1,**options):
    if history_frames not in (1,2)or(history_frames!=1 and curriculum!='continuous_v1'):
        raise ValueError('History2 requires continuous curriculum')
    if validation_selection not in ('overall','worst_family'):raise ValueError('Unknown validation selection')
    if validation_selection!='overall'and curriculum!='continuous_v1':raise ValueError('Family selection requires continuous games')
    if curriculum=='continuous_v1':
        if task_mode!='joint':raise ValueError('Continuous curriculum requires joint policy')
        from sim.continuous_push import ContinuousPushEnv
        if history_frames==2:
            from sim.continuous_history import HistoryContinuousPushEnv
            env=HistoryContinuousPushEnv(split=split,seed=seed,**options)
        else:env=ContinuousPushEnv(split=split,seed=seed,**options)
        env.continuous_focus=continuous_focus
        return env
    if curriculum=='tactical_v1':
        if task_mode!='joint':raise ValueError('Tactical curriculum requires joint policy')
        from sim.tactical_strategy import TacticalStrategyEnv
        return TacticalStrategyEnv(split=split,seed=seed,**options)
    if curriculum!='classic':raise ValueError('Unknown curriculum')
    if task_mode=='attack':return make_attack_env(split=split,seed=seed,**options)
    if task_mode!='joint':raise ValueError('Unknown strategy task mode')
    if split not in ('train','validation'):raise ValueError('Training forbids test split')
    from sim.strategy_env import StrategyEnv
    return StrategyEnv(split=split,seed=seed,**options)


def training_sampler(env,seed=42,initial_hand_mode='shuffle',task_mode='joint'):
    if hasattr(env,'continuous_focus'):
        from sim.continuous_sampling import ContinuousScenarioCycle
        return ContinuousScenarioCycle(env.scenarios,seed,initial_hand_mode,env.continuous_focus)
    if task_mode=='attack':return AttackScenarioCycle(env.scenarios,seed=seed,initial_hand_mode=initial_hand_mode)
    from sim.strategy_sampler import StrategyScenarioCycle
    return StrategyScenarioCycle(env.scenarios,seed=seed,initial_hand_mode=initial_hand_mode)


def observation_contract(task_mode,curriculum='classic',history_frames=1):
    if history_frames==2:
        if curriculum!='continuous_v1':raise ValueError('History2 requires continuous curriculum')
        from sim.public_frame_history import CONTRACT
        return CONTRACT
    if curriculum=='continuous_v1':
        from sim.continuous_push import CONTRACT
        return CONTRACT
    if curriculum=='tactical_v1':
        from sim.tactical_strategy import CONTRACT
        return CONTRACT
    if task_mode=='attack':return OBSERVATION_CONTRACT
    from sim.strategy_env import OBSERVATION_CONTRACT as contract
    return contract


def summarize_tasks(records):
    from sim.defense_log_summary import summarize_episodes
    attacks=[r for r in records if r.get('task')=='attack' or
             (not r.get('task') and 'goal_kind'in r)]
    defenses=[r for r in records if r.get('task')=='defense' or
              (not r.get('task') and 'own_tower_hp_loss'in r and 'goal_kind'not in r)]
    return dict(attack=summarize_attack(attacks),defense=summarize_episodes(
        defenses,returns=[r.get('episode_return')for r in defenses]))


def summarize_attack(records):
    """Completed-episode outcomes, including known counts instead of fake zeros."""
    def summary(rows):
        result={'episodes':len(rows)}
        for key in ('attack_success','goal_success','goal_window_missed','successful_spell_finish'):
            known=[r.get(key)for r in rows if isinstance(r.get(key),bool)]
            result[key]=dict(known=len(known),count=sum(known),rate=sum(known)/len(known)if known else None)
        for key in ('episode_return','time_to_first_goal_card_ms','anticipatory_goal_card_ms',
                    'hog_target_tower_damage','elixir_spent','redundant_spells','missed_resolved_spells'):
            values=[r[key]for r in rows if isinstance(r.get(key),(int,float))]
            result[key]=dict(known=len(values),mean=float(np.mean(values))if values else None)
        return result
    result=summary(records)
    result['by']={}
    for field in ('timer_visible','goal_kind','setup_kind'):
        result['by'][field]={str(value):summary([r for r in records if r.get(field)==value])
                             for value in sorted({r.get(field)for r in records},key=str)}
    return result


def evaluate(network,device,*,episodes=32,seed=900000,env_options=None,task_mode='attack'):
    if episodes<1:raise ValueError('Positive validation episode count required')
    worst_family=(env_options or {}).get('validation_selection')=='worst_family'
    if worst_family and (episodes<6 or episodes%6):raise ValueError('Family validation requires a multiple of six games per task')
    records=[]
    was_training=network.training
    network.eval()
    try:
        with torch.no_grad():
            tasks=('attack','defense')if task_mode=='joint'else ('attack',)
            for task_index in range(episodes*len(tasks)):
                task=tasks[task_index//episodes]
                index=task_index%episodes
                episode_seed=seed+(task_index if worst_family else index)
                env=make_training_env(task_mode=task_mode,split='validation',seed=episode_seed,**(env_options or {}))
                if task_mode=='joint':
                    candidates=[getattr(s,'scenario_id',s)for s in env.scenarios
                                if getattr(s,'scenario_id',s).startswith(task+'::')]
                    if not candidates:raise ValueError('Missing validation task coverage')
                    if (env_options or {}).get('curriculum')=='tactical_v1'and task=='attack':
                        goals=('supported_hog','air_defense_counterpush','hog_defense_counterpush')
                        # Reserve validation cases for every new skill; the
                        # remainder still samples the original exercises.
                        if index%4<3:
                            goal=goals[index%4]
                            candidates=[s.scenario_id for s in env.scenarios if
                                s.task=='attack'and getattr(s.scenario,'tactical_goal',None)==goal]
                    chosen=(sorted(candidates)[index%len(candidates)]if worst_family
                            else random.Random(seed+index).choice(sorted(candidates)))
                    obs,info=env.reset(seed=episode_seed,scenario_id=chosen)
                else:obs,info=env.reset(seed=episode_seed)
                total=0.
                bound=max(1202,int(getattr(env,'max_ms',240000)/getattr(env,'profile',{}).get('decision_interval_ms',500))+2)
                for _ in range(bound):
                    logits,_=network(*tensor_observations([obs],device))
                    mask=action_masks([info],device)
                    require_finite(logits=logits)
                    action=int(logits.masked_fill(~mask,-torch.inf).argmax(-1).item())
                    obs,reward,terminal,truncated,info=env.step(action)
                    total+=reward
                    if terminal or truncated:
                        records.append(dict(info['metrics'],episode_return=total))
                        break
                else:raise RuntimeError('Attack validation exceeded finite step bound')
    finally:network.train(was_training)
    result=dict(mean_reward=float(np.mean([r['episode_return']for r in records])),
                records=records,summary=summarize_attack(records),by_task=summarize_tasks(records),
                split='validation',seed=seed,episodes_per_task=episodes,
                selection_metric='equal_task_mean_episode_return_provisional')
    if (env_options or {}).get('curriculum')=='continuous_v1':
        wins=sum(r['outcome']=='win'for r in records)
        draws=sum(r['outcome']=='draw'for r in records)
        unknown=sum(r['outcome']=='unknown'for r in records)
        result.update(wins=wins,draws=draws,unknown=unknown,games=len(records),
            mean_elixir_overflow=float(np.mean([r['elixir_overflow']for r in records])),
            selection_metric='full_game_win_draw_score_v1',
            selection_score=(wins+.5*draws)/len(records))
        if worst_family:
            from sim.continuous_validation import score_families
            result.update(score_families(records))
    if (env_options or {}).get('curriculum')=='tactical_v1':
        rates={}
        for goal in ('supported_hog','air_defense_counterpush','hog_defense_counterpush'):
            rows=[r for r in records if r.get('tactical_goal')==goal]
            rates[goal]=dict(episodes=len(rows),successes=sum(r.get('goal_success',False)for r in rows))
        if any(not r['episodes']for r in rates.values()):raise ValueError('Missing tactical validation skill')
        values=[r['successes']/r['episodes']for r in rates.values()]
        defense=[r for r in records if r.get('task')=='defense']
        no_damage=sum(r['no_tower_damage']for r in defense)/len(defense)
        result.update(tactical_skills=rates,selection_metric='tactical_min_macro_and_defense_no_damage_v1',
            selection_score=.5*min(values)+.25*sum(values)/len(values)+.25*no_damage)
    return result


def code_fingerprint(task_mode='attack',curriculum='classic',card_balance_weight=0.,history_frames=1):
    root=Path(__file__).resolve().parents[1]
    names=('sim/attack_env.py','sim/attack_scenarios.py','sim/attack_sampler.py',
           'sim/train_attack_ppo.py','sim/engine.py','sim/pathfind.py','sim/match.py','sim/env.py',
           'sim/defense_env.py','sim/defense_assets.py','sim/defense_scenarios.py',
           'sim/public_gamedata.py','sim/public_scenario_spells.py','sim/spells.py','sim/arena.py','sim/reward_profiles.py')
    if task_mode=='joint':names+=('sim/strategy_env.py','sim/strategy_sampler.py')
    if curriculum=='tactical_v1':names+=('sim/tactical_strategy.py',)
    if curriculum=='continuous_v1':names+=('sim/continuous_push.py','sim/gamedata.py','sim/entities.py',
        'sim/projectile_calibration.py','sim/kite_geometry.py','sim/opponents.py','sim/eval_defense_duel.py','sim/continuous_validation.py','sim/continuous_sampling.py')
    if curriculum=='continuous_v1':names+=('sim/board_potential.py','sim/tactical_rewards.py','sim/lane_reflection.py','sim/card_exploration.py')
    if card_balance_weight:names+=('sim/card_balance.py',)
    if history_frames==2:names+=('sim/public_frame_history.py','sim/continuous_history.py')
    return hashlib.sha256(json.dumps({name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                     for name in names},sort_keys=True).encode()).hexdigest()


def validate_resume(blob,config):
    if blob.get('schema')!='attack-ppo-boundary-v1' or not blob.get('exact_resume_available'):
        raise ValueError('Attack exact-resume boundary state required')
    keys=('envs','rollout','epochs','minibatch_size','seed','lr','gamma','gae_lambda','clip',
          'value_coef','entropy_coef','max_grad_norm','action_delay_ms','max_ms','deadline_scale',
          'visible_fraction','initial_hand_mode','geometry','geometry_sha256','coordinate_mode',
          'task_mode','observation_contract','num_scalars','source_code_sha256','catalog_sha256','device',
          'reward_profile','reward_profile_sha256','curriculum','card_balance_weight','card_balance_target','continuous_profile',
          'value_warmup_steps','target_kl','validation_selection','continuous_focus','validation_seed','history_frames','num_planes','policy_temperature',
          'supervised_replay_weight','supervised_replay_metadata','card_exploration_epsilon')
    for key in keys:
        default=0. if key in('supervised_replay_weight','card_exploration_epsilon')else None
        if blob['config'].get(key,default)!=config.get(key,default):
            raise ValueError(f'Attack resume configuration mismatch: {key}')
    required=('optimiser','rng','sampler','environment_states','observations','infos',
              'episode_returns','episode_counters','step','update')
    if any(key not in blob for key in required):raise ValueError('Incomplete attack resume state')


def train(args):
    exploration=getattr(args,'card_exploration_epsilon',0.)
    if not np.isfinite(exploration)or not 0<=exploration<=.5:raise ValueError('Invalid card exploration share')
    replay_weight=getattr(args,'supervised_replay_weight',0.)
    replay_data=getattr(args,'supervised_replay_data',None)
    if not np.isfinite(replay_weight)or not 0<=replay_weight<=10 or (replay_weight and not replay_data):
        raise ValueError('Invalid supervised replay controls')
    if replay_data and (getattr(args,'curriculum','classic')!='continuous_v1' or getattr(args,'history_frames',1)!=1):
        raise ValueError('Supervised replay requires the continuous20-plane public contract')
    temperature=getattr(args,'policy_temperature',1.)
    tempered_logits(torch.empty(0),temperature)
    balance_weight=getattr(args,'card_balance_weight',0.)
    balance_target=getattr(args,'card_balance_target','all_cards')
    if (getattr(args,'value_warmup_steps',0)<0 or not np.isfinite(getattr(args,'target_kl',0))
            or getattr(args,'target_kl',0)<0):raise ValueError('Invalid optimizer controls')
    if not np.isfinite(balance_weight)or not 0<=balance_weight<=1:raise ValueError('Card balance weight must be finite in [0,1]')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; CPU fallback forbidden')
    if min(args.updates,args.envs,args.rollout,args.epochs,args.minibatch_size,args.checkpoint_every,
           args.target_environment_steps,args.eval_episodes)<1:
        raise ValueError('Positive training counts required')
    for name in ('lr','gamma','gae_lambda','clip','value_coef','entropy_coef','max_grad_norm'):
        if not np.isfinite(getattr(args,name)):
            raise ValueError(f'Nonfinite attack hyperparameter: {name}')
    if args.lr<=0 or not 0<args.clip<1 or not 0<=args.gamma<=1 or not 0<=args.gae_lambda<=1 or args.max_grad_norm<=0:
        raise ValueError('Invalid attack optimizer/GAE bounds')
    if not np.isfinite(args.wallclock_seconds) or args.wallclock_seconds<=0:
        raise ValueError('Finite positive wallclock required')
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    options={key:getattr(args,key)for key in ('action_delay_ms','max_ms','deadline_scale',
        'visible_fraction','initial_hand_mode','geometry','coordinate_mode')}
    curriculum=getattr(args,'curriculum','classic')
    history_frames=getattr(args,'history_frames',1)
    if history_frames!=1 and curriculum!='continuous_v1':raise ValueError('History2 requires continuous curriculum')
    if curriculum!='classic':options['curriculum']=curriculum
    if curriculum=='continuous_v1':
        profile=getattr(args,'continuous_profile',None)
        if profile:profile=json.loads(Path(profile).read_text())
        if profile and (profile.get('board_potential_weight',0)or profile.get('reserve_potential_weight',0)):
            if 'shaping_gamma'in profile and profile['shaping_gamma']!=args.gamma:
                raise ValueError('Potential shaping gamma must equal PPO gamma')
            profile=dict(profile,shaping_gamma=args.gamma)
        options['continuous_profile']=profile
        options['validation_selection']=getattr(args,'validation_selection','overall')
        options['continuous_focus']=getattr(args,'continuous_focus','uniform')
        options['history_frames']=history_frames
    reward_profile,reward_sha=load_reward_profile(getattr(args,'reward_profile',None))
    defense_rewards,attack_rewards=reward_objects(reward_profile)
    if args.task_mode=='joint':options.update(defense_rewards=defense_rewards,attack_rewards=attack_rewards)
    else:options['rewards']=attack_rewards
    envs=[make_training_env(task_mode=args.task_mode,split='train',seed=args.seed+i,**options)for i in range(args.envs)]
    sampler=training_sampler(envs[0],seed=args.seed,initial_hand_mode=args.initial_hand_mode,task_mode=args.task_mode)
    observations,infos=map(list,zip(*[env.reset(**sampler.next())for env in envs]))
    num_scalars=len(observations[0]['scalars'])
    expected_scalars=NUM_SCALARS+(3 if args.task_mode=='joint'else 2)
    if num_scalars!=expected_scalars or any(len(obs['scalars'])!=num_scalars for obs in observations):
        raise ValueError('Attack timer feature dimension mismatch')
    blob=torch.load(args.resume or args.init,map_location='cpu',weights_only=False)if args.resume or args.init else None
    architecture=checkpoint_architecture(blob)if blob else args.architecture
    contract=observation_contract(args.task_mode,curriculum,history_frames)
    num_planes=observations[0]['planes'].shape[0]
    network=build_attack_network(architecture,device,num_scalars=num_scalars,num_planes=num_planes)
    initialization=load_attack_state(network,blob,exact=bool(args.resume),observation_contract=contract)if blob else {'fresh_initialization':True}
    optimizer=torch.optim.Adam(network.parameters(),lr=args.lr)
    replay=None
    if replay_data:
        from sim.supervised_replay import SuccessfulActionReplay
        replay=SuccessfulActionReplay(replay_data,device)
    config=dict(vars(args),architecture=architecture,num_scalars=num_scalars,num_planes=num_planes,history_frames=history_frames,policy_temperature=temperature,
        reward_profile=reward_profile,reward_profile_sha256=reward_sha,
        observation_contract=contract,source_code_sha256=code_fingerprint(args.task_mode,curriculum,balance_weight,history_frames),
        catalog_sha256=sampler.catalog_sha256,geometry_sha256=envs[0].geometry.fingerprint,
        initialization=initialization,init_sha256=(blob['config'].get('init_sha256')if args.resume else
            hashlib.sha256(Path(args.init).read_bytes()).hexdigest()if args.init else None))
    config.update(supervised_replay_weight=replay_weight,supervised_replay_metadata=replay.metadata if replay else None,
        card_exploration_epsilon=exploration)
    if curriculum=='continuous_v1':config['continuous_profile']=envs[0].profile
    step,start_update,best=0,0,-float('inf')
    counters=[0]*args.envs
    episode_returns=[0.]*args.envs
    if args.resume:
        validate_resume(blob,config)
        optimizer.load_state_dict(blob['optimiser'])
        sampler.load_state_dict(blob['sampler'])
        if len(blob['environment_states'])!=args.envs:raise ValueError('Attack environment count mismatch')
        for env,state in zip(envs,blob['environment_states']):env.restore(state)
        observations,infos=blob['observations'],blob['infos']
        episode_returns,counters=blob['episode_returns'],blob['episode_counters']
        step,start_update,best=blob['step'],blob['update'],blob['best_validation']
        restore_rng(blob['rng'])
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    started=time.monotonic()
    actual_cuda=torch.cuda.get_device_name(device)if device.type=='cuda'else None
    def checkpoint(update,best_save=False):
        state=dict(schema='attack-ppo-boundary-v1',state_dict=network.state_dict(),
            optimiser=optimizer.state_dict(),architecture=architecture,config=config,
            step=step,update=update,best_validation=best,rng=rng_state(),sampler=sampler.state_dict(),
            environment_states=[env.snapshot()for env in envs],observations=observations,infos=infos,
            episode_returns=episode_returns,episode_counters=counters,exact_resume_available=True)
        torch.save(state,output/(f'best_validation_{update:06d}.pt'if best_save else f'update_{update:06d}.pt'))
    last_update=start_update
    play_cursors=[len(env.match.play_log)for env in envs]
    for update in range(start_update+1,args.updates+1):
        if step>=args.target_environment_steps:break
        last_update=update
        rows,episodes=[],[]
        accepted_cards=Counter()
        for _ in range(args.rollout):
            planes,scalars=tensor_observations(observations,device)
            masks=action_masks(infos,device)
            with torch.no_grad():
                logits,values=network(planes,scalars)
                logits=tempered_logits(logits,temperature)
                require_finite(logits=logits,values=values)
                distribution=training_distribution(logits,masks,exploration)
                actions=distribution.sample();log_prob=distribution.log_prob(actions)
            final_obs,rewards,terminals,truncations,final_infos=map(list,zip(*[
                env.step(int(action))for env,action in zip(envs,actions.cpu().tolist())]))
            for index,env in enumerate(envs):
                log=env.match.play_log
                accepted_cards.update(e['card']for e in log[play_cursors[index]:]if e['side']==1)
                play_cursors[index]=len(log)
            for info in final_infos:validate_transition(info)
            with torch.no_grad():_,next_values=network(*tensor_observations(final_obs,device))
            rows.append((planes,scalars,masks,actions,log_prob,values.reshape(-1),
                torch.tensor(rewards,device=device),next_values.reshape(-1),
                torch.tensor(terminals,device=device),torch.tensor(truncations,device=device)))
            observations,infos=final_obs,final_infos
            for index,(terminal,truncated)in enumerate(zip(terminals,truncations)):
                episode_returns[index]+=float(rewards[index])
                if terminal or truncated:
                    episodes.append(dict(final_infos[index]['metrics'],episode_return=episode_returns[index]))
                    episode_returns[index]=0.;counters[index]+=1
                    observations[index],infos[index]=envs[index].reset(**sampler.next())
                    play_cursors[index]=0
            step+=args.envs
        planes,scalars,masks,actions,old_log,values,rewards,next_values,terminals,truncations=[torch.stack(x)for x in zip(*rows)]
        advantages,returns=generalized_advantage(rewards,values,next_values,terminals,truncations,args.gamma,args.gae_lambda)
        advantages=normalize_advantages(advantages).flatten()
        require_finite(advantages=advantages,returns=returns)
        planes,scalars=planes.flatten(0,1),scalars.flatten(0,1)
        masks,actions,old_log,returns=masks.flatten(0,1),actions.flatten(),old_log.flatten(),returns.flatten()
        losses=[];balance_losses=[];kl_values=[];kl_stopped=False;replay_losses=[]
        warmup=step<=getattr(args,'value_warmup_steps',0)
        for name,parameter in network.named_parameters():
            parameter.requires_grad_(not warmup or name.startswith('value'))
        for _ in range(args.epochs):
            for indices in torch.randperm(len(actions),device=device).split(args.minibatch_size):
                logits,predicted=network(planes[indices],scalars[indices])
                logits=tempered_logits(logits,temperature)
                dist=training_distribution(logits,masks[indices],exploration)
                ratio=(dist.log_prob(actions[indices])-old_log[indices]).exp()
                log_ratio=dist.log_prob(actions[indices])-old_log[indices]
                kl=float(((ratio-1)-log_ratio).mean().detach())
                kl_values.append(kl)
                if not warmup and getattr(args,'target_kl',0)>0 and kl>args.target_kl:
                    kl_stopped=True
                    break
                policy=-torch.minimum(ratio*advantages[indices],ratio.clamp(1-args.clip,1+args.clip)*advantages[indices]).mean()
                loss=policy+args.value_coef*(predicted.reshape(-1)-returns[indices]).square().mean()-args.entropy_coef*dist.entropy().mean()
                if balance_weight and not warmup:
                    from sim.card_balance import card_balance_loss
                    balance=card_balance_loss(logits,scalars[indices],masks[indices],balance_target)
                    loss=loss+balance_weight*balance
                    balance_losses.append(float(balance.detach()))
                if replay_weight and not warmup:
                    auxiliary=replay.loss(network)
                    loss=loss+replay_weight*auxiliary
                    replay_losses.append(float(auxiliary.detach()))
                require_finite(loss=loss,ratio=ratio)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(network.parameters(),args.max_grad_norm,error_if_nonfinite=True)
                optimizer.step();losses.append(float(loss.detach()))
            if kl_stopped:break
        record=dict(update=update,step=step,target_environment_steps=args.target_environment_steps,policy_temperature=temperature,
            card_exploration_epsilon=exploration,
            percent=min(100.,100*step/args.target_environment_steps),elapsed_seconds=time.monotonic()-started,
            actual_cuda=actual_cuda,loss=float(np.mean(losses)),gradient_norm=float(norm),episodes=episodes,
            train_summary=summarize_attack(episodes),by_task=summarize_tasks(episodes))
        record['card_usage']=dict(accepted_counts=dict(accepted_cards),balance_weight=balance_weight,
            target=balance_target,mean_balance_js=float(np.mean(balance_losses))if balance_losses else 0.)
        record['optimizer_controls']=dict(value_only_warmup=warmup,kl_stopped=kl_stopped,
            mean_approximate_kl=float(np.mean(kl_values)))
        record['supervised_replay']=dict(weight=replay_weight,optimizer_batches=len(replay_losses),
            mean_loss=float(np.mean(replay_losses))if replay_losses else None,
            train_examples=replay.metadata['train_examples']if replay else 0)
        is_best=False
        if args.eval_every>0 and update%args.eval_every==0:
            record['validation']=evaluate(network,device,episodes=args.eval_episodes,
                seed=getattr(args,'validation_seed',900000),env_options=options,task_mode=args.task_mode)
            score=record['validation'].get('selection_score',record['validation']['mean_reward']);is_best=score>best;best=max(best,score)
        budget_stop=time.monotonic()-started>=args.wallclock_seconds
        stop=budget_stop or step>=args.target_environment_steps or update==args.updates
        if update%args.checkpoint_every==0 or stop:checkpoint(update)
        if is_best:checkpoint(update,True)
        with (output/'metrics.jsonl').open('a')as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps({key:record[key]for key in ('update','step','percent','elapsed_seconds','actual_cuda','loss')},
              default=str),flush=True)
        concise_keys=('episodes','attack_success','goal_window_missed','episode_return',
                      'time_to_first_goal_card_ms','elixir_spent','redundant_spells')
        print(json.dumps(dict(train={k:record['train_summary'][k]for k in concise_keys})),flush=True)
        if 'validation'in record:
            print(json.dumps(dict(validation={k:record['validation']['summary'][k]for k in concise_keys})),flush=True)
        if args.task_mode=='joint':
            defense=record['by_task']['defense']
            print(json.dumps(dict(defense={key:defense[key]for key in
                ('n','no_damage','no_damage_pct','mean_hp_loss','princess_deaths','king_deaths','timeouts','invalid','mean_return')})),flush=True)
        if stop:break
    if last_update==start_update:
        checkpoint(start_update)
    result=dict(status='target_completed'if step>=args.target_environment_steps else 'incomplete_budget',
        step=step,update=last_update,target_environment_steps=args.target_environment_steps,
        target_met=step>=args.target_environment_steps,quality_accepted=False,
        best_validation=None if not np.isfinite(best)else best,observation_contract=contract,task_mode=args.task_mode)
    (output/'training_result.json').write_text(json.dumps(result,indent=2)+'\n')
    return output


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group()
    source.add_argument('--init');source.add_argument('--resume')
    p.add_argument('--reward-profile',help='Explicit coefficient JSON; stored and checked on resume')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--task-mode',choices=('joint','attack'),default='joint')
    p.add_argument('--curriculum',choices=('classic','tactical_v1','continuous_v1'),default='classic')
    p.add_argument('--continuous-profile',help='Explicit full-game auxiliary reward and timing JSON')
    p.add_argument('--history-frames',type=int,choices=(1,2),default=1)
    p.add_argument('--value-warmup-steps',type=int,default=0)
    p.add_argument('--target-kl',type=float,default=0.)
    p.add_argument('--validation-selection',choices=('overall','worst_family'),default='overall')
    p.add_argument('--continuous-focus',choices=('uniform','air_golem'),default='uniform')
    p.add_argument('--validation-seed',type=int,default=900000)
    p.add_argument('--card-balance-weight',type=float,default=0.)
    p.add_argument('--card-balance-target',choices=('all_cards','units'),default='all_cards')
    p.add_argument('--policy-temperature',type=float,default=1.)
    p.add_argument('--card-exploration-epsilon',type=float,default=0.)
    p.add_argument('--supervised-replay-data',help='Pinned TRAIN successful-action labels; grouped holdout excluded')
    p.add_argument('--supervised-replay-weight',type=float,default=0.)
    p.add_argument('--architecture',default=COMPACT_ARCHITECTURE)
    p.add_argument('--initial-hand-mode',choices=('scenario','shuffle'),default='shuffle')
    p.add_argument('--geometry',choices=('legacy_v1','native_anchors_candidate_v1'),default='legacy_v1')
    p.add_argument('--coordinate-mode',choices=('physical_live','legacy_training','physical_live_cells_v2'),default='physical_live_cells_v2')
    for name,value in (('updates',100000),('target-environment-steps',1000000),('envs',8),('rollout',128),
        ('epochs',3),('minibatch-size',256),('seed',42),('checkpoint-every',50),('eval-every',100),('eval-episodes',32),('max-ms',240000)):
        p.add_argument('--'+name,type=int,default=value)
    for name,value in (('deadline-scale',1.),('visible-fraction',.5),('wallclock-seconds',21600.),('action-delay-ms',150.),
        ('lr',5e-5),('gamma',.997),('gae-lambda',.95),('clip',.2),('value-coef',.5),('entropy-coef',.01),('max-grad-norm',.5)):
        p.add_argument('--'+name,type=float,default=value)
    return p


if __name__=='__main__':train(parser().parse_args())
