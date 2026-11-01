"""Closed-loop paid local air defenses using the actual PPO mask/observation.

Local seed354 is reused development holdout, not independent game validation.
The opponent supplies the initial paid push only; no match win rate is implied.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from scripts.attack.build_mixed_bank_sequences import support_opening, support_source
from scripts.attack.sequence_execution import EXECUTION
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.adapter import grid_to_point
from sim.continuous_push import ContinuousPushEnv
from sim.env import ClashEnv
from sim.train_attack_ppo import initialize_attack_policy
from sim.decision_economy import empty as empty_economy, record as record_economy
from sim.spell_economy import SpellEconomyTracker


def episode(network, context, delay, wave='none'):
    if wave not in ('none','same','opposite'):raise ValueError('Known wave mode required')
    opening = support_opening(context)
    env = object.__new__(ContinuousPushEnv)
    env.match, env.cards, env.spells = opening.match, opening.cards, opening.spells
    env.profile = {'cannon_center_only': 1}
    env.pending, env.slot_ready, env.reflected = [], {}, False
    start = env.match.elapsed_ms
    hp = {k: t.hitpoints for k, t in env.match.towers[1].items()}
    commands, rejected = [], []
    minimum_bank = env.match.players[1].elixir
    first_clear = None
    waits = 0
    enemy_wave=[]
    economy=empty_economy()
    spell_tracker=SpellEconomyTracker(env.match).attach()
    initial_overflow=env.match.players[1].natural_elixir_overflow_milli
    time_stats=dict(observed_ms=0,available_below4_ms=0,under_threat_ms=0,
        under_threat_below4_ms=0,under_threat_full_bank_ms=0,under_threat_overflow_milli=0)
    while not env.match.finished and env.match.elapsed_ms < start + 60000:
        now = env.match.elapsed_ms
        if wave!='none'and not enemy_wave and now>=start+12000:
            enemy=env.match.players[-1]
            affordable=[c for c in enemy.hand if c not in env.spells and env.match.can_play(-1,c)]
            priority=('hog_rider','musketeer','balloon','knight')
            if affordable:
                card=min(affordable,key=lambda c:(priority.index(c)if c in priority else len(priority),c))
                lane=context['lane']if wave=='same'else('right'if context['lane']=='left'else'left')
                x=3 if lane=='left'else 14
                before=enemy.elixir
                if env.match.play_card(-1,card,grid_to_point(x,12,1)):
                    cost=env.cards[card].cost*1000
                    if before-enemy.elixir!=cost:raise ValueError('Enemy wave not paid exactly')
                    enemy_wave.append(dict(card=card,at_ms=now,lane=lane,
                        elixir_before=before/1000,elixir_after=enemy.elixir/1000,cost=cost/1000))
        if (now - start) % 250 == 0:
            obs, mask = env._observe(), env._mask()
            with torch.inference_mode():
                logits, _ = network(torch.as_tensor(obs['planes'])[None], torch.as_tensor(obs['scalars'])[None])
                if not torch.isfinite(logits[0, mask]).all():
                    raise ValueError('Nonfinite legal logits')
                action = int(logits[0].masked_fill(~torch.as_tensor(mask), -torch.inf).argmax())
            native=env.match.players[1].elixir/1000
            available=max(0.,native-sum(env.cards[e[1]].cost for e in env.pending))
            record_economy(economy,hand=env.match.players[1].hand,mask=mask,action=action,
                native_elixir=native,available_elixir=available)
            if action:
                slot, x, y = ClashEnv.decode(action)
                env.pending.append((now + delay, env.match.players[1].hand[slot], slot, x, y))
            else:
                waits += 1
        for event in list(env.pending):
            due, card, slot, x, y = event
            if due > now:
                continue
            accepted = env.match.play_card(1, card, grid_to_point(x, y, 1))
            row = dict(card=card, requested_ms=due-delay, at_ms=now, cell=[x, y])
            (commands if accepted else rejected).append(row)
            env.slot_ready[slot] = now + EXECUTION['action_cooldown_ms']
            env.pending.remove(event)
        native=env.match.players[1].elixir
        available=max(0,native-sum(env.cards[e[1]].cost*1000 for e in env.pending))
        threat=any(e.alive and e.side==-1 and not e.is_tower for e in env.match.battle.entities.values())
        overflow=env.match.players[1].natural_elixir_overflow_milli
        env.match.step(50)
        dt=env.match.elapsed_ms-now
        time_stats['observed_ms']+=dt
        time_stats['available_below4_ms']+=dt*(available<4000)
        time_stats['under_threat_ms']+=dt*threat
        time_stats['under_threat_below4_ms']+=dt*(threat and available<4000)
        time_stats['under_threat_full_bank_ms']+=dt*(threat and native>=10000)
        if threat:time_stats['under_threat_overflow_milli']+=env.match.players[1].natural_elixir_overflow_milli-overflow
        minimum_bank = min(minimum_bank, env.match.players[1].elixir)
        enemies = sum(e.alive and e.side == -1 and not e.is_tower for e in env.match.battle.entities.values())
        if not enemies and first_clear is None:
            first_clear = env.match.elapsed_ms - start
    damage = {k: v-env.match.towers[1][k].hitpoints for k, v in hp.items()}
    enemies = sum(e.alive and e.side == -1 and not e.is_tower for e in env.match.battle.entities.values())
    return dict(seed=context['seed'], lane=context['lane'], bank=context['bank'], hand=context['hand'],
        command_delay_ms=delay, start_ms=start, elapsed_ms=env.match.elapsed_ms-start, first_clear_ms=first_clear,
        tower_damage=damage, total_damage=sum(damage.values()), enemies_remaining=enemies,
        clear_without_damage=enemies == 0 and sum(damage.values()) == 0,
        minimum_elixir=minimum_bank / 1000, wait_decisions=waits,wave_mode=wave,enemy_wave=enemy_wave,
        decision_economy=economy,elixir_time=time_stats,spell_economy=spell_tracker.metrics(),
        natural_elixir_overflow=(env.match.players[1].natural_elixir_overflow_milli-initial_overflow)/1000,
        cards=dict(Counter(c['card'] for c in commands)), commands=commands, rejected=rejected)


def evaluate(support, weights, output, wave='none'):
    torch.set_num_threads(1)
    pins = physics_pins()
    source_sha=file_sha(__file__)
    snapshot=Path(output).with_suffix('.source.py')
    if snapshot.exists()and file_sha(snapshot)!=source_sha:raise ValueError('Source snapshot already differs')
    if not snapshot.exists():snapshot.write_bytes(Path(__file__).read_bytes())
    contexts = [r for r in support_source(support) if r['seed'] == 354]
    if len(contexts) != 12:
        raise ValueError('Expected twelve complete heldout support contexts')
    models = []
    for path in weights:
        blob = torch.load(path, map_location='cpu', weights_only=False)
        net, _, _ = initialize_attack_policy(blob, device='cpu', num_planes=20, num_scalars=309,
            observation_contract='joint_public_presence_continuous_slots_v2')
        net.eval()
        rows = []
        for context in contexts:
            for delay in (1350, 2025):
                rows.append(episode(net, context, delay,wave))
                print(json.dumps(dict(checkpoint=str(path), episodes=len(rows))), flush=True)
        models.append(dict(checkpoint=str(path), checkpoint_sha256=file_sha(path), rows=rows,
            summaries=[dict(command_delay_ms=d, episodes=len(part),
                clear_without_damage=sum(r['clear_without_damage'] for r in part),
                mean_tower_damage=float(np.mean([r['total_damage'] for r in part])),
                rejected_commands=sum(len(r['rejected']) for r in part),
                mean_empty_spell_elixir=float(np.mean([r['spell_economy']['empty_spell_elixir']for r in part])),
                mean_elixir_overflow=float(np.mean([r['natural_elixir_overflow']for r in part])),
                mean_under_threat_overflow=float(np.mean([r['elixir_time']['under_threat_overflow_milli']/1000 for r in part])),
                available_below4_time_fraction=sum(r['elixir_time']['available_below4_ms']for r in part)/max(1,sum(r['elixir_time']['observed_ms']for r in part)),
                under_threat_below4_time_fraction=sum(r['elixir_time']['under_threat_below4_ms']for r in part)/max(1,sum(r['elixir_time']['under_threat_ms']for r in part)),
                cards_per_episode={c: sum(r['cards'].get(c, 0) for r in part)/len(part)
                    for c in sorted({c for r in part for c in r['cards']})})
                for d in (1350, 2025) for part in [[r for r in rows if r['command_delay_ms'] == d]]]))
    if pins != physics_pins()or source_sha!=file_sha(__file__)or source_sha!=file_sha(snapshot):
        raise ValueError('Physics or driver changed during evaluation')
    result = dict(schema='support-closed-loop-development-v1', physics_pins=pins,
        source_sha256=source_sha,source_snapshot=str(snapshot),models=models, quality_accepted=False, live_games_started=False,wave_mode=wave,
        scope='All twelve reused local heldout support contexts; 60-second closed-loop horizon. Optional one extra enemy troop after 12s uses the real hand and regenerated elixir. Actor continues after first clear. Not full-match wins or independent validation.')
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--support', required=True)
    p.add_argument('--weights', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--wave',choices=('none','same','opposite'),default='none')
    a = p.parse_args()
    evaluate(a.support, a.weights, a.output,a.wave)
