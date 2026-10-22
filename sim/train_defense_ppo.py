"""Finite, checkpointed PPO for the shared defense curriculum (Kaggle CUDA)."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from sim.defense_sampler import ScenarioCycle
from sim.train_ppo import (build_network, checkpoint_architecture,
                           load_policy_state, masked_distribution)


def generalized_advantage(rewards, values, next_values, terminated, truncated,
                          gamma=0.997, gae_lambda=0.95):
    """Bootstrap time limits from final observations, never across resets."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]).to(values.dtype) * next_values[t] - values[t]
        continuation = ~(terminated[t] | truncated[t])
        carry = delta + gamma * gae_lambda * continuation.to(values.dtype) * carry
        advantages[t] = carry
    return advantages, advantages + values


def normalize_advantages(advantages):
    return (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)


def require_finite(**values):
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"nonfinite {name}")


def tensor_observations(observations, device):
    return (torch.as_tensor(np.stack([o['planes'] for o in observations]), device=device),
            torch.as_tensor(np.stack([o['scalars'] for o in observations]), device=device))


def action_masks(infos, device):
    mask = torch.as_tensor(np.stack([i['action_mask'] for i in infos]), device=device, dtype=torch.bool)
    if not mask.any(dim=1).all():
        raise ValueError('empty legal action mask')
    return mask



def validate_transition(info):
    """Broken scripted attacks are data errors, never timeout training data."""
    metrics = info.get('metrics') or {}
    reasons = (info.get('terminal_reason'), metrics.get('terminal_reason'))
    for reason in reasons:
        if isinstance(reason, str) and reason.startswith('invalid_script'):
            scenario = (info.get('scenario_id') or metrics.get('scenario_id')
                        or info.get('scenario') or metrics.get('scenario') or 'unknown')
            raise ValueError(f'Invalid scripted attack: scenario={scenario}, reason={reason}')


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def evaluate(network, device, *, episodes, seed, env_options):
    """Validation only; no test-set-driven checkpoint selection."""
    from sim.defense_env import DefenseEnv
    records = []
    was_training = network.training
    network.eval()
    with torch.no_grad():
        for episode in range(episodes):
            env = DefenseEnv(seed=seed + episode, split='validation', **env_options)
            obs, info = env.reset(seed=seed + episode)
            total = 0.0
            while True:
                logits, _ = network(*tensor_observations([obs], device))
                mask = action_masks([info], device)
                action = int(logits.masked_fill(~mask, -torch.inf).argmax(dim=1).item())
                obs, reward, terminal, truncated, info = env.step(action)
                validate_transition(info)
                total += float(reward)
                if terminal or truncated:
                    records.append({'reward': total, 'metrics': info.get('metrics', {}),
                                    'terminated': terminal, 'truncated': truncated})
                    break
    network.train(was_training)
    return {'mean_reward': float(np.mean([r['reward'] for r in records])), 'episodes': records}


def train(args):
    from sim.defense_env import DefenseEnv
    from sim.defense_curriculum import curriculum_selection
    from sim.defense_checkpointing import checkpoint_due
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; CPU training fallback forbidden')
    if not np.isfinite(args.wallclock_seconds) or args.wallclock_seconds <= 0:
        raise ValueError('wallclock-seconds must be finite and positive')
    if min(args.updates, args.envs, args.rollout, args.epochs, args.minibatch_size) < 1:
        raise ValueError('training sizes must be positive')
    checkpoint_due(0,args.checkpoint_every)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    options = {'action_delay_ms': args.action_delay_ms, 'max_ms': args.max_ms,
               'initial_elixir': args.initial_elixir, 'difficulty': args.difficulty}
    counters = [0] * args.envs
    envs = [DefenseEnv(seed=args.seed + i, split='train', **options) for i in range(args.envs)]
    curriculum=None
    for env in envs:
        env.scenarios,curriculum=curriculum_selection(env.scenarios,args.scenario_ids)
    sampler = ScenarioCycle([s.scenario_id for s in envs[0].scenarios], seed=args.seed)
    reset = [e.reset(**sampler.next()) for e in envs]
    observations, infos = map(list, zip(*reset))
    blob = torch.load(args.resume or args.init, map_location='cpu', weights_only=False)
    architecture = checkpoint_architecture(blob)
    network = build_network(observations[0]['planes'].shape[0],
                            observations[0]['scalars'].shape[0],
                            len(infos[0]['action_mask']), architecture).to(device)
    load_policy_state(network, blob['state_dict'])
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    step, start_update, best = 0, 0, -float('inf')
    exact_resume = False
    if args.resume:
        old_config = blob['config']
        for key in ('envs', 'rollout', 'seed', 'action_delay_ms', 'max_ms', 'initial_elixir', 'difficulty'):
            if old_config[key] != vars(args)[key]:
                raise ValueError(f'resume configuration mismatch: {key}')
        for key in ('scenario_ids','scenario_ids_sha256'):
            if old_config.get(key)!=curriculum[key]:
                raise ValueError(f'resume curriculum configuration mismatch: {key}')
        optimizer.load_state_dict(blob['optimiser'])
        step, start_update, best = blob['step'], blob['update'], blob['best_validation']
        counters = blob['episode_counters']
        if 'sampler' not in blob:
            raise ValueError('Resume checkpoint lacks scenario-cycle state')
        sampler.load_state_dict(blob['sampler'])
        if blob.get('environment_states') is not None:
            for env, state in zip(envs, blob['environment_states']):
                env.restore(state)
            observations, infos = blob['observations'], blob['infos']
            exact_resume = True
        else:
            # Explicit fresh-episode fallback: optimizer/steps restored, trajectory is not.
            counters = [n + 1 for n in counters]
            reset = [e.reset(**sampler.next()) for e in envs]
            observations, infos = map(list, zip(*reset))
        restore_rng(blob['rng'])
    config = dict(vars(args), architecture=architecture, exact_resume=exact_resume)
    config.update(curriculum)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    network.train()
    started = time.monotonic()
    actual_cuda = torch.cuda.get_device_name(device) if device.type == 'cuda' else None
    print(json.dumps(dict(event='START', device=str(device), cuda_available=torch.cuda.is_available(), actual_cuda=actual_cuda, step=step)), flush=True)
    for update in range(start_update + 1, start_update + args.updates + 1):
        rows, episodes = [], []
        for _ in range(args.rollout):
            planes, scalars = tensor_observations(observations, device)
            mask = action_masks(infos, device)
            with torch.no_grad():
                logits, values = network(planes, scalars)
                require_finite(logits=logits, values=values)
                dist = masked_distribution(logits, mask)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            transitions = [e.step(int(a)) for e, a in zip(envs, actions.cpu().tolist())]
            for transition in transitions:
                validate_transition(transition[4])
            final_obs, rewards, terminals, truncations, final_infos = map(list, zip(*transitions))
            with torch.no_grad():
                _, next_values = network(*tensor_observations(final_obs, device))
            rows.append((planes, scalars, mask, actions, log_probs, values.reshape(-1),
                         torch.tensor(rewards, device=device), next_values.reshape(-1),
                         torch.tensor(terminals, device=device), torch.tensor(truncations, device=device)))
            observations, infos = final_obs, final_infos
            for i, (terminal, truncated) in enumerate(zip(terminals, truncations)):
                if terminal or truncated:
                    episodes.append(final_infos[i].get('metrics', {}))
                    counters[i] += 1
                    observations[i], infos[i] = envs[i].reset(**sampler.next())
            step += args.envs
        planes, scalars, masks, actions, old_log, values, rewards, next_values, terminals, truncations = [torch.stack(x) for x in zip(*rows)]
        advantages, returns = generalized_advantage(rewards, values, next_values, terminals, truncations, args.gamma, args.gae_lambda)
        advantages = normalize_advantages(advantages).flatten()
        require_finite(advantages=advantages, returns=returns)
        planes, scalars = planes.flatten(0, 1), scalars.flatten(0, 1)
        masks, actions, old_log, returns = masks.flatten(0, 1), actions.flatten(), old_log.flatten(), returns.flatten()
        losses = []
        for _ in range(args.epochs):
            order = torch.randperm(len(actions), device=device)
            for indices in order.split(args.minibatch_size):
                logits, predicted = network(planes[indices], scalars[indices])
                dist = masked_distribution(logits, masks[indices])
                ratio = (dist.log_prob(actions[indices]) - old_log[indices]).exp()
                policy = -torch.minimum(ratio * advantages[indices], ratio.clamp(1-args.clip, 1+args.clip) * advantages[indices]).mean()
                value_loss = (predicted.reshape(-1) - returns[indices]).square().mean()
                loss = policy + args.value_coef * value_loss - args.entropy_coef * dist.entropy().mean()
                require_finite(loss=loss, ratio=ratio)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
        record = {'update': update, 'step': step, 'elapsed_seconds': time.monotonic()-started, 'actual_cuda': actual_cuda, 'loss': float(np.mean(losses)), 'gradient_norm': float(norm), 'episodes': episodes}
        is_best = False
        if args.eval_every > 0 and update % args.eval_every == 0:
            record['validation'] = evaluate(network, device, episodes=args.eval_episodes, seed=900000, env_options=options)
            score = record['validation']['mean_reward']
            is_best = score > best
            best = max(best, score)
        budget_stop=time.monotonic()-started>=args.wallclock_seconds
        save_update=checkpoint_due(update,args.checkpoint_every,
            final=update==start_update+args.updates,budget_stop=budget_stop)
        if save_update or is_best:
            snapshot = [e.snapshot() for e in envs] if all(hasattr(e, 'snapshot') and hasattr(e, 'restore') for e in envs) else None
            checkpoint = {'state_dict': network.state_dict(), 'optimiser': optimizer.state_dict(),
                          'architecture': architecture, 'step': step, 'update': update,
                          'best_validation': best, 'config': config, 'rng': rng_state(),
                          'episode_counters': counters, 'sampler': sampler.state_dict(), 'environment_states': snapshot,
                          'observations': observations, 'infos': infos,
                          'exact_resume_available': snapshot is not None}
            if save_update:
                torch.save(checkpoint, output / f'update_{update:06d}.pt')
            if is_best:
                torch.save(checkpoint, output / f'best_validation_{update:06d}.pt')
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')
        print(json.dumps({k: v for k, v in record.items() if k != 'episodes'}), flush=True)
        if budget_stop:
            print(json.dumps(dict(event='STOP', reason='wallclock_update_boundary', update=update, step=step, elapsed_seconds=time.monotonic()-started)), flush=True)
            break
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init', required=True)
    p.add_argument('--resume')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--checkpoint-every',type=int,default=1)
    p.add_argument('--wallclock-seconds', type=float, default=3600.)
    p.add_argument('--scenario-ids',help='Explicit audited train curriculum JSON; validation unaffected')
    p.add_argument('--difficulty', choices=('simple','medium','complex','impossible'))
    for name, default in [('updates', 5), ('envs', 2), ('rollout', 8), ('epochs', 4), ('minibatch-size', 64), ('seed', 42), ('eval-every', 5), ('eval-episodes', 8), ('max-ms', 240000)]:
        p.add_argument('--' + name, type=int, default=default)
    for name, default in [('lr', 3e-4), ('gamma', .997), ('gae-lambda', .95), ('clip', .2), ('value-coef', .5), ('entropy-coef', .01), ('max-grad-norm', .5), ('action-delay-ms', 150), ('initial-elixir', 10)]:
        p.add_argument('--' + name, type=float, default=default)
    return p


if __name__ == '__main__':
    train(parser().parse_args())
