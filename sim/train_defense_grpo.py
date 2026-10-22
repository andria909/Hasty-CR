"""Grouped, on-policy defense RL; train only in the authorized Kaggle job.

Each group resets the deterministic public-observation environment to one seed
and scenario. Returns are standardized within that group. Each trajectory's
masked clipped objective is averaged over its own steps before averaging the
trajectories: long defenses do not receive more optimization weight. This is
an episodic GRPO adaptation, not token-level LLM GRPO or a credit-assignment
claim. There is no value loss; the unused value head is frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from sim.train_ppo import (build_network, checkpoint_architecture,
                           load_policy_state, masked_distribution)


def group_advantages(returns, epsilon=1e-8):
    values = torch.as_tensor(returns, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError('non-finite group returns')
    std = values.std(unbiased=False)
    degenerate = bool(std <= epsilon)
    return torch.zeros_like(values) if degenerate else (values-values.mean())/std, degenerate


def checked_distribution(logits, mask):
    if mask.dtype != torch.bool or mask.shape != logits.shape or not mask.any(-1).all():
        raise ValueError('invalid or empty legal-action support')
    if not torch.isfinite(logits[mask]).all():
        raise ValueError('non-finite legal logits')
    return masked_distribution(logits, mask)


def trajectory_objective(logits, reference_logits, masks, actions, old_logp,
                         advantage, clip=0.2, beta=0.02, entropy_coef=0.0):
    """Exact categorical KL on identical legal supports; no sampled-KL bias."""
    policy = checked_distribution(logits, masks)
    reference = checked_distribution(reference_logits.detach(), masks)
    if not masks.gather(1, actions[:, None]).all():
        raise ValueError('recorded action is outside stored support')
    new_logp = policy.log_prob(actions)
    ratio = (new_logp-old_logp.detach()).exp()
    advantage = torch.as_tensor(advantage, device=logits.device)
    surrogate = torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)
    # Illegal entries have logp=-inf. Replace before multiplication to avoid 0*nan.
    delta = torch.where(masks, policy.logits-reference.logits, 0.0)
    kl = (policy.probs*delta).sum(-1).mean()
    entropy = policy.entropy().mean()
    return -surrogate.mean()+beta*kl-entropy_coef*entropy, kl, entropy


def tensors(observations, device):
    return (torch.as_tensor(np.stack([o['planes'] for o in observations]), device=device),
            torch.as_tensor(np.stack([o['scalars'] for o in observations]), device=device))


def validate_episode_info(info):
    """Technical script failures are not game outcomes or training signals."""
    metrics = info.get('metrics', {})
    reasons = (info.get('terminal_reason'), metrics.get('terminal_reason'))
    if any(str(reason).startswith('invalid_script') for reason in reasons):
        scenario = metrics.get('scenario_id', info.get('scenario_id', 'unknown'))
        raise ValueError(f'invalid scripted defense scenario {scenario}: {reasons}')


def rollout(env, network, device, generator, seed, scenario_id=None, greedy=False, deadline=None, on_step=None):
    obs, info = env.reset(seed=seed, scenario_id=scenario_id)
    validate_episode_info(info)
    result = dict(observations=[], masks=[], actions=[], old_logp=[], rewards=[])
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("rollout wallclock limit")
        planes, scalars = tensors([obs], device)
        mask = torch.as_tensor(np.asarray(info['action_mask']), device=device, dtype=torch.bool)[None]
        with torch.no_grad():
            logits, _ = network(planes, scalars)
            dist = checked_distribution(logits, mask)
            action = (dist.probs.argmax(-1) if greedy else
                      torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1))
        result['observations'].append({k: np.array(v, copy=True) for k,v in obs.items()})
        result['masks'].append(mask[0].cpu().numpy().copy())
        result['actions'].append(int(action.item()))
        result['old_logp'].append(float(dist.log_prob(action).item()))
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        validate_episode_info(info)
        if on_step is not None:
            on_step()
        result['rewards'].append(float(reward))
        if terminated or truncated:
            result['metrics'] = info.get('metrics', {})
            result['return'] = sum(result['rewards'])
            result['terminated'], result['truncated'] = bool(terminated), bool(truncated)
            return result


RESUME_KEYS = ('group_size','epochs','batch_steps','seed','lr','clip','beta','entropy',
               'action_delay_ms','max_ms','initial_elixir','difficulty',
               'validation_every','validation_episodes','architecture','device_name',
               'init_sha256','trajectory_weighting','torch_version','environment_code_sha256',
               'scenario_ids','scenario_ids_sha256')


def restore_training_state(blob,network,reference,optimizer,sampler,generator,config):
    if not blob.get('resume_supported',False) or blob.get('checkpoint_phase')!='boundary':
        raise ValueError('Checkpoint is inside an unfinished group/validation; exact resume unsupported. Use a boundary checkpoint.')
    if blob.get('resume_schema')!='defense-grpo-boundary-v1':
        raise ValueError('Checkpoint lacks supported exact-resume metadata')
    differences=[key for key in RESUME_KEYS if blob['config'].get(key)!=config.get(key)]
    if differences:
        raise ValueError(f'Resume configuration mismatch: {differences}')
    load_policy_state(network,blob['state_dict'])
    load_policy_state(reference,blob['reference_state_dict'])
    optimizer.load_state_dict(blob['optimizer'])
    sampler.load_state_dict(blob['sampler'])
    torch.set_rng_state(blob['torch_rng'].cpu())
    if blob['cuda_rng']:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG cannot be restored without CUDA')
        torch.cuda.set_rng_state_all([state.cpu() for state in blob['cuda_rng']])
    generator.set_state(blob['action_rng'].cpu())
    np.random.set_state(blob['numpy_rng'])
    random.setstate(blob['python_rng'])
    return dict(completed=blob['updates'],steps=blob['environment_steps'],
                degenerate_count=blob['degenerate_groups'],
                actual_steps=dict(blob['actual_environment_steps']),
                best_validation=blob['best_validation'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--init', type=Path)
    source.add_argument('--resume', type=Path, help='Exact continuation from a completed-group checkpoint; updates is total target')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--group-size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-steps', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--beta', type=float, default=0.02)
    p.add_argument('--entropy', type=float, default=0.001)
    p.add_argument('--action-delay-ms', type=int, default=150)
    p.add_argument('--max-ms', type=int, default=240000)
    p.add_argument('--initial-elixir', type=float, default=10)
    p.add_argument('--scenario-ids',type=Path,help='Explicit audited train curriculum JSON; validation unaffected')
    p.add_argument('--difficulty', choices=('simple','medium','complex','impossible'))
    p.add_argument('--checkpoint-every',type=int,default=1)
    p.add_argument('--save-seconds', type=float, default=300)
    p.add_argument('--wallclock-seconds', type=float, default=3600)
    p.add_argument('--validation-every', type=int, default=10)
    p.add_argument('--validation-episodes', type=int, default=8)
    args = p.parse_args(argv)
    if args.group_size < 2 or min(args.updates,args.epochs,args.validation_every,args.validation_episodes,args.batch_steps) < 1:
        p.error('positive updates/epochs/validation settings and group-size >=2 required')
    if args.save_seconds <= 0 or args.wallclock_seconds <= 0 or not 0 < args.clip < 1 or args.beta < 0:
        p.error('invalid time/objective settings')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; refusing CPU fallback')
    from sim.defense_env import DefenseEnv
    from sim.defense_sampler import ScenarioCycle
    from sim.defense_curriculum import curriculum_selection
    from sim.defense_checkpointing import checkpoint_due
    checkpoint_due(0,args.checkpoint_every)
    from sim.env import NUM_PLANES, NUM_SCALARS, ACTIONS
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed+1)
    blob = torch.load(args.resume or args.init, map_location=device, weights_only=False)
    architecture = checkpoint_architecture(blob)
    network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS, architecture).to(device)
    load_policy_state(network, blob['state_dict'])
    for name, parameter in network.named_parameters():
        if name.startswith('value.'):
            parameter.requires_grad_(False)
    reference = copy.deepcopy(network).eval().requires_grad_(False)
    network.eval()  # no stochastic hidden layers; action sampling has its own RNG
    optimizer = torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=args.lr)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['init_sha256'] = (blob['config']['init_sha256'] if args.resume else
                             hashlib.sha256(args.init.read_bytes()).hexdigest())
    config['architecture'] = architecture
    config['device_name'] = torch.cuda.get_device_name(device) if device.type=='cuda' else str(device)
    config['trajectory_weighting'] = 'mean steps per trajectory, then mean group'
    config['torch_version'] = str(torch.__version__)
    contract_files=('defense_env.py','defense_scenarios.py','defense_assets.py',
                    'public_scenario_spells.py','engine.py','entities.py','match.py','env.py',
                    'gamedata.py','spells.py','defense_sampler.py','defense_curriculum.py')
    contract=hashlib.sha256()
    for filename in contract_files:
        contract.update(filename.encode()); contract.update(Path(__file__).with_name(filename).read_bytes())
    config['environment_code_sha256'] = contract.hexdigest()
    env_kwargs = dict(action_delay_ms=args.action_delay_ms,max_ms=args.max_ms,initial_elixir=args.initial_elixir)
    env = DefenseEnv(seed=args.seed, split='train', difficulty=args.difficulty, **env_kwargs)
    env.scenarios,curriculum=curriculum_selection(env.scenarios,args.scenario_ids)
    config.update(curriculum)
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    sampler = ScenarioCycle([s.scenario_id for s in env.scenarios],seed=args.seed)
    validation = DefenseEnv(seed=900000, split='validation', difficulty=args.difficulty, **env_kwargs)
    start = last_save = time.monotonic()
    steps = degenerate_count = completed = 0
    best_validation = -float('inf')
    actual_steps = dict(train=0,validation=0)
    save_sequence = 0
    checkpoint_phase = 'boundary'
    if args.resume:
        restored=restore_training_state(blob,network,reference,optimizer,sampler,generator,config)
        completed=restored['completed']; steps=restored['steps']
        degenerate_count=restored['degenerate_count']; actual_steps=restored['actual_steps']
        best_validation=restored['best_validation']
    def maybe_save():
        nonlocal last_save
        if time.monotonic()-last_save >= args.save_seconds:
            save('periodic'); last_save=time.monotonic()
    def count_train():
        actual_steps['train'] += 1
        maybe_save()
    def count_validation():
        actual_steps['validation'] += 1
        maybe_save()

    def save(label):
        nonlocal save_sequence
        save_sequence += 1
        # Unique names avoid overwriting any user's existing artifact.
        torch.save(dict(state_dict=network.state_dict(), reference_state_dict=reference.state_dict(),
            optimizer=optimizer.state_dict(),architecture=architecture,config=config,
            sampler=sampler.state_dict(),resume_schema='defense-grpo-boundary-v1',
            resume_supported=checkpoint_phase=='boundary',checkpoint_phase=checkpoint_phase,
            best_validation=best_validation,
            updates=completed,groups=completed,environment_steps=steps,
            degenerate_groups=degenerate_count, actual_environment_steps=dict(actual_steps), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            action_rng=generator.get_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate()),args.output/f'{label}_u{completed:06d}_s{save_sequence:06d}.pt')

    save('boundary')
    for update in range(completed,args.updates):
        if time.monotonic()-start >= args.wallclock_seconds:
            break
        # Preserve an exact recovery point before entering a potentially long group.
        if checkpoint_phase!='boundary':
            raise RuntimeError('Unexpected unfinished group')
        checkpoint_phase='collection'
        group_context = sampler.next()
        seed = group_context['seed']
        try:
            trajectories = [rollout(env,network,device,generator,seed,
                scenario_id=group_context['scenario_id'],
                deadline=start+args.wallclock_seconds,on_step=count_train) for _ in range(args.group_size)]
        except TimeoutError:
            # Incomplete groups are not optimized or mixed with complete returns.
            break
        returns = [t['return'] for t in trajectories]
        advantages, degenerate = group_advantages(returns)
        steps += sum(len(t['actions']) for t in trajectories)
        degenerate_count += int(degenerate)
        stats = []
        checkpoint_phase='update'
        if not degenerate:
            for _ in range(args.epochs):
                optimizer.zero_grad()
                # Backprop per trajectory, accumulating equal group weights.
                for trajectory, advantage in zip(trajectories,advantages):
                    length = len(trajectory['actions'])
                    for begin in range(0,length,args.batch_steps):
                        end = min(begin+args.batch_steps,length)
                        planes, scalars = tensors(trajectory['observations'][begin:end],device)
                        masks = torch.as_tensor(np.stack(trajectory['masks'][begin:end]),device=device,dtype=torch.bool)
                        actions = torch.tensor(trajectory['actions'][begin:end],device=device)
                        old_logp = torch.tensor(trajectory['old_logp'][begin:end],device=device)
                        logits,_ = network(planes,scalars)
                        with torch.no_grad():
                            ref_logits,_ = reference(planes,scalars)
                        loss,kl,entropy = trajectory_objective(logits,ref_logits,masks,actions,old_logp,
                            advantage.to(device),args.clip,args.beta,args.entropy)
                        if not torch.isfinite(loss):
                            raise RuntimeError('non-finite GRPO objective')
                        (loss*((end-begin)/length)/args.group_size).backward()
                        stats.append([loss.item(),kl.item(),entropy.item()])
                torch.nn.utils.clip_grad_norm_(network.parameters(),0.5,error_if_nonfinite=True)
                optimizer.step()
        completed = update+1
        checkpoint_phase='boundary'
        record = dict(update=completed,groups=completed,environment_steps=steps,
            group_seed=seed,scenario_id=group_context['scenario_id'],
            sampler=sampler.state_dict(),returns=returns,degenerate=degenerate,
            degenerate_groups=degenerate_count,lengths=[len(t['actions']) for t in trajectories],
            metrics=[t['metrics'] for t in trajectories],elapsed_seconds=time.monotonic()-start,
            actual_environment_steps=dict(actual_steps),
            loss=float(np.mean([s[0] for s in stats])) if stats else None,
            kl=float(np.mean([s[1] for s in stats])) if stats else None,
            entropy=float(np.mean([s[2] for s in stats])) if stats else None)
        if completed%args.validation_every==0 and time.monotonic()-start < args.wallclock_seconds:
            checkpoint_phase='validation'
            evaluations = []
            for i in range(args.validation_episodes):
                try:
                    evaluations.append(rollout(validation,network,device,generator,900000+i,greedy=True,
                        deadline=start+args.wallclock_seconds,on_step=count_validation))
                except TimeoutError:
                    break
            record['validation'] = dict(mean_return=float(np.mean([t['return'] for t in evaluations])) if evaluations else None,
                completed_episodes=len(evaluations),
                metrics=[t['metrics'] for t in evaluations],split='validation',
                environment_steps=sum(len(t['actions']) for t in evaluations))
            checkpoint_phase='boundary'
            if len(evaluations)==args.validation_episodes and record['validation']['mean_return'] > best_validation:
                best_validation = record['validation']['mean_return']; save('best_validation')
        with (args.output/'updates.jsonl').open('a') as stream:
            stream.write(json.dumps(record,default=str)+'\n')
        print(json.dumps(record,default=str),flush=True)
        if checkpoint_due(completed,args.checkpoint_every):
            save('boundary')
        maybe_save()
    save('final')


if __name__ == '__main__':
    main()
