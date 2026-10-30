"""Replay successful TRAIN search plans into public pending-aware BC examples.

Targets describe a complete local teacher continuation, not standalone optimal
placements. The held-out seed remains a local diagnostic, never a game test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scripts.attack.audit_air_label_latency import opening
from scripts.attack.sequence_physics import require_current_physics
from scripts.kaggle.local_artifact_transfer import file_sha
from scripts.attack.sequence_execution import EXECUTION,require_execution
from sim.adapter import grid_to_point
from sim.continuous_push import ContinuousPushEnv
from sim.defense_assets import load_defense_assets
from sim.env import ACTIONS, DECK_26
from sim.runner import resolve_deck


def replay(context, cards, spells, *, pending_inputs=False):
    match, hp = opening(context['opening'], cards, spells)
    return replay_match(match, hp, context['best'], cards, spells, pending_inputs=pending_inputs)


def replay_match(match, hp, plan, cards, spells, *, pending_inputs=False):
    """Replay a separately reconstructed paid opening through the same masks."""
    if not plan['valid'] or not plan['clear_without_damage'] or not plan['commands']:
        raise ValueError('Successful paid plan required')
    # Use the actual PPO environment observation/mask implementation unchanged.
    if pending_inputs:
        from sim.continuous_pending import PendingContinuousPushEnv
        env = object.__new__(PendingContinuousPushEnv)
    else:
        env = object.__new__(ContinuousPushEnv)
    env.match, env.cards, env.spells = match, cards, spells
    env.profile={'cannon_center_only':1}
    env.pending, env.slot_ready, env.reflected = [], {}, False
    commands = {c['requested_ms']: c for c in plan['commands']}
    if len(commands) != len(plan['commands']):
        raise ValueError('One command per decision required')
    start = match.elapsed_ms
    last = max(commands)
    if any((t-start) % 250 for t in commands):
        raise ValueError('Teacher requests must match 250ms policy decisions')
    rows, delivered = [], []
    while not match.finished and match.elapsed_ms < start + 60000:
        now = match.elapsed_ms
        if now <= last and (now-start) % 250 == 0:
            obs, mask = env._observe(), env._mask()
            command = commands.get(now)
            action = 0
            if command:
                slot = command['slot']; x, y = command['cell']
                if match.players[1].hand[slot] != command['card']:
                    raise ValueError('Replay hand diverged')
                action = 1 + slot*576 + y*18 + x
            if not mask[action]:
                raise ValueError('Teacher action illegal in actual PPO mask')
            label = np.zeros(ACTIONS, dtype=np.bool_); label[action] = True
            rows.append(dict(planes=obs['planes'], scalars=obs['scalars'], masks=mask,
                success_masks=label, decision_ms=now, action=action,
                pending_count=len(env.pending), reserved_elixir=sum(cards[p[1]].cost for p in env.pending)))
            if command:
                if command['due_ms'] != now+1350 or command['placed_ms'] != now+1350:
                    raise ValueError('1350ms delivery required')
                env.pending.append((now+1350, command['card'], slot, x, y))
        for event in list(env.pending):
            due, card, slot, x, y = event
            if due > now:
                continue
            if not match.play_card(1, card, grid_to_point(x, y, 1)):
                raise ValueError('Paid replay deployment failed')
            env.pending.remove(event); env.slot_ready[slot] = now+EXECUTION['action_cooldown_ms']
            delivered.append(dict(card=card, placed_ms=now))
        match.step(50)
    damage = {k:v-match.towers[1][k].hitpoints for k,v in hp.items()}
    enemies = sum(e.alive and e.side == -1 and not e.is_tower for e in match.battle.entities.values())
    if damage != plan['own_tower_damage'] or enemies != plan['remaining_enemies']:
        raise ValueError('Teacher outcome did not reproduce')
    if delivered != [dict(card=c['card'], placed_ms=c['placed_ms']) for c in plan['commands']]:
        raise ValueError('Teacher command coverage differs')
    return rows


def build(search_path, output, *, pending_inputs=False):
    search_path, output = Path(search_path), Path(output)
    result = json.loads(search_path.read_text())
    protocol = result['protocol']
    require_current_physics(protocol)
    require_execution(protocol)
    source = Path(protocol['source_data'])
    if (protocol['schema'] != 'late-air-native-delay-search-v1' or protocol['command_delay_ms'] != 1350
            or file_sha(source/'examples.npz') != protocol['data_sha256']
            or file_sha('scripts/attack/search_late_air_defense.py') != protocol['source_sha256']):
        raise ValueError('Pinned original search and data required')
    cards, spells, *_ = load_defense_assets(); cards.update(resolve_deck(cards, list(DECK_26)))
    arrays = {k:[] for k in ('planes','scalars','masks','success_masks')}
    records = []
    for context in result['contexts']:
        for row in replay(context, cards, spells, pending_inputs=pending_inputs):
            index = len(records)
            for key in arrays:
                arrays[key].append(row[key])
            records.append(dict(seed=context['opening']['seed'], accepted=True, array_index=index,
                skill='late_air_sequence', context=context['context'], lane=context['opening']['lane'],
                **{k:row[k] for k in ('decision_ms','action','pending_count','reserved_elixir')}))
    seeds = sorted({r['seed'] for r in records})
    if len(seeds) < 3:
        raise ValueError('Three seed groups required')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'examples.npz', **{k:np.stack(v) for k,v in arrays.items()})
    manifest = dict(schema='late-air-public-sequence-labels-v1', split='train', examples=len(records),
        physics_pins=protocol['physics_pins'],execution=EXECUTION, seeds=seeds, records=records, sequences=len(result['contexts']), command_delay_ms=1350,
        data_sha256=file_sha(output/'examples.npz'), search_sha256=file_sha(search_path),
        builder_sha256=file_sha(__file__), environment_sha256=file_sha('sim/continuous_push.py'),
        grouping='Maximum seed held out across complete sequences', quality_accepted=False,
        own_inputs='Exact ContinuousPushEnv._observe/_mask:20 planes,309 scalars; reserved elixir and busy slots',
        scope='Successful sequential teacher plans, not independently sufficient single actions',
        limitations=['Bounded search, not global optimum','No enemy reinforcement after fixture opening',
                     'WAIT is a valid successful continuation, not proof other actions are worse',
                     'Pending commands have no explicit spatial input in the current policy contract',
                     'Correlated TRAIN frames; local holdout is not a winrate; no full-game promotion'])
    manifest['pending_inputs']=pending_inputs
    if pending_inputs:
        from sim.pending_inputs import CONTRACT
        manifest.update(schema='late-air-pending-sequence-labels-v1',observation_contract=CONTRACT,
            own_inputs='Exact PendingContinuousPushEnv._observe/_mask:24 planes,313 scalars; own pending cells/times only',
            pending_source_sha256={name:file_sha(name)for name in ('sim/pending_inputs.py','sim/continuous_pending.py')})
        manifest['limitations']=[text for text in manifest['limitations']if not text.startswith('Pending commands have no explicit')]
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--search',required=True); p.add_argument('--output',required=True)
    p.add_argument('--pending-inputs',action='store_true')
    a=p.parse_args(); r=build(a.search,a.output,pending_inputs=a.pending_inputs)
    print(json.dumps({k:r[k] for k in ('examples','sequences','seeds','data_sha256')}))
