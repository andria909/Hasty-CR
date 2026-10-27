"""Trace greedy decisions in fresh TRAIN full games; no optimization or live play."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sim.arena import MT
from sim.continuous_push import ContinuousPushEnv, CONTRACT
from sim.train_attack_ppo import initialize_attack_policy


def audit(checkpoint, output, games=4):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve earlier diagnostic')
    if games < 1:
        raise ValueError('Positive game count required')
    torch.set_num_threads(1)
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    net, _, _ = initialize_attack_policy(blob, device='cpu', num_scalars=309,
                                         observation_contract=CONTRACT)
    net.eval()
    records = []
    with torch.inference_mode():
        for index in range(games):
            seed = 6100000 + index
            env = ContinuousPushEnv(split='train')
            task = ('attack', 'defense')[index % 2]
            obs, info = env.reset(seed=seed, scenario_id=f'{task}::air/push')
            trace = []
            for _ in range(1242):
                logits, _ = net(torch.as_tensor(obs['planes'][None]),
                                torch.as_tensor(obs['scalars'][None]))
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite policy logits')
                mask = info['action_mask']
                action = int(logits.masked_fill(~torch.as_tensor(mask[None]), -torch.inf).argmax(-1))
                hand = list(env.match.players[1].hand)
                flying = [e for e in env.match.battle.entities.values()
                          if e.alive and not e.is_tower and e.side == -1 and e.flying]
                nearby = [e for e in flying if e.pos.y >= 14 * MT]
                chosen = 'hold' if action == 0 else hand[(action - 1) // 576]
                musk_slot = hand.index('musketeer') if 'musketeer' in hand else None
                musk_legal = musk_slot is not None and bool(mask[1 + musk_slot * 576:1 + (musk_slot + 1) * 576].any())
                friendly_musk = [e for e in env.match.battle.entities.values()
                                 if e.alive and e.side == 1 and e.name == 'musketeer']
                if nearby or action:
                    trace.append(dict(at_ms=env.match.elapsed_ms, hand=hand,
                        own_elixir=env.match.players[1].elixir / 1000,
                        air_near_river_or_own_half=[dict(name=e.name, x=e.pos.x / MT,
                            y=e.pos.y / MT, hp=e.hitpoints) for e in nearby],
                        musketeer_legal=musk_legal, own_musketeers=len(friendly_musk),
                        choice=chosen, action=action))
                obs, _, terminal, truncated, info = env.step(action)
                if terminal or truncated:
                    break
            else:
                raise RuntimeError('Finite full game did not terminate')
            threatened = [r for r in trace if r['air_near_river_or_own_half']]
            unprotected = [r for r in threatened if r['own_musketeers'] == 0]
            records.append(dict(seed=seed, scenario_id=env.scenario.scenario_id,
                metrics=info['metrics'], threatened_frames=len(threatened),
                unprotected_frames=len(unprotected),
                unprotected_bank_below_four=sum(r['own_elixir'] < 4 for r in unprotected),
                unprotected_musketeer_legal=sum(r['musketeer_legal'] for r in unprotected),
                choices_while_unprotected=dict(Counter(r['choice'] for r in unprotected)),
                trace=trace))
            print(json.dumps(dict(game=index + 1, outcome=info['metrics']['outcome'],
                                  unprotected_frames=len(unprotected))), flush=True)
    result = dict(schema='train-full-game-air-response-audit-v1', split='train_diagnostic',
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        games=records, training_performed=False, quality_accepted=False, live_games_started=False,
        limitations=['Consecutive frames are correlated, not independent opportunities',
                     'No nearby Musketeer does not prove that playing one is optimal',
                     'Fresh TRAIN trajectories, not validation or test win-rate'])
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--games', type=int, default=4)
    args = parser.parse_args()
    audit(args.checkpoint, args.output, args.games)
