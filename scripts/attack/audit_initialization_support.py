"""Separate exact demonstration matching from physical defense outcomes.

One selected action followed by no further own commands is intentionally a
local open-loop diagnostic, not an evaluation of the full policy or win rate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.attack.build_mixed_bank_sequences import support_opening, support_source
from scripts.attack.search_late_air_defense import rollout
from scripts.attack.sequence_execution import EXECUTION
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.cannon_placement import restrict_cannon
from sim.eval_defense_duel import mechanical_mask, policy_observation, JOINT_MODE
from sim.train_attack_ppo import initialize_attack_policy


def audit(verified, support, output):
    torch.set_num_threads(1)
    pins = physics_pins()
    proof = json.loads(Path(verified).read_text())
    if proof['schema'] != 'verified-bank-initialization-study-v1':
        raise ValueError('Verified initialization study required')
    contexts = [r for r in support_source(support) if r['seed'] == 354]
    rows = []
    for run in proof['runs']:
        path = Path(run['checkpoint'])
        if file_sha(path) != run['score']['checkpoint_sha256']:
            raise ValueError('Verified policy changed')
        blob = torch.load(path, map_location='cpu', weights_only=False)
        net, _, _ = initialize_attack_policy(blob, device='cpu', num_planes=20,
            num_scalars=309, observation_contract='joint_public_presence_continuous_slots_v2')
        net.eval()
        for context in contexts:
            safe = [r for r in context['trials'] if r['valid'] and r['clear_without_damage'] and r['commands']]
            # Mirror the original accepted support contexts, including pairs.
            if not safe:
                continue
            teacher = min(safe, key=lambda r: (r['elixir_spent'], json.dumps(r['commands'], sort_keys=True)))
            env = support_opening(context)
            obs = policy_observation(env.match, 1, JOINT_MODE)
            mask = restrict_cannon(mechanical_mask(env.match, env.cards, env.spells, 1),
                env.match.players[1].hand, EXECUTION['cannon_placement_mode'])
            with torch.inference_mode():
                logits, _ = net(torch.as_tensor(obs['planes'])[None], torch.as_tensor(obs['scalars'])[None])
                action = int(logits[0].masked_fill(~torch.as_tensor(mask), -torch.inf).argmax())
            first = None
            if action:
                slot, cell = divmod(action - 1, 576)
                y, x = divmod(cell, 18)
                first = dict(card=env.match.players[1].hand[slot], cell=[x, y])
            target = teacher['commands'][0]
            exact = first is not None and first['card'] == target['card'] and first['cell'] == target['cell']
            hp = {k: t.hitpoints for k, t in env.match.towers[1].items()}
            outcomes = {str(delay): rollout(env.match, hp, env.cards, env.spells, context['lane'],
                first, delay_ms=delay) for delay in (1350, 2025)}
            rows.append(dict(run=run['id'], lane=context['lane'], bank=context['bank'],
                hand=context['hand'], simulation_seed=context['seed'], chosen=first,
                matches_first_demonstration=exact, teacher_commands=len(teacher['commands']), outcomes=outcomes))
            print(json.dumps(dict(completed=len(rows), run=run['id'])), flush=True)
    if pins != physics_pins():
        raise ValueError('Physics changed during diagnostic')
    result = dict(schema='initialization-support-open-loop-audit-v1', physics_pins=pins,
        source_sha256=file_sha(__file__), verified_sha256=file_sha(verified), rows=rows,
        summary=[dict(run=run['id'], contexts=len(part),
            exact_first=sum(r['matches_first_demonstration'] for r in part),
            safe_1350=sum(r['outcomes']['1350']['clear_without_damage'] for r in part),
            safe_2025=sum(r['outcomes']['2025']['clear_without_damage'] for r in part),
            safe_but_not_exact=sum(r['outcomes']['1350']['clear_without_damage'] and
                not r['matches_first_demonstration'] for r in part))
            for run in proof['runs'] for part in [[r for r in rows if r['run'] == run['id']]]],
        quality_accepted=False, live_games_started=False,
        scope='Heldout support contexts, one action then no commands for 60 seconds. Pair teachers need further decisions; a failure here does not prove closed-loop failure.')
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('verified', 'support', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    audit(a.verified, a.support, a.output)
