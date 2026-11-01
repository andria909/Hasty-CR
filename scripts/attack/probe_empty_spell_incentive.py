"""Paired idle-arena diagnostic of waste versus overflow shaping, not win rate."""
import argparse
import json
from pathlib import Path

from scripts.kaggle.local_artifact_transfer import file_sha
from sim.continuous_spell_economy import SpellEconomyContinuousPushEnv


class IdleArena(SpellEconomyContinuousPushEnv):
    def _enemy(self):
        # Isolate reward arithmetic: neither branch faces an enemy deployment.
        pass


def branch(trial, spell, weight, cast):
    seed = {'fireball': 0, 'zap': 3}[spell]
    keys = ('overflow_penalty', 'fireball_value_reward', 'reserve_potential_weight',
            'own_hp_penalty', 'barbarian_train_fraction', 'lane_reflection_fraction',
            'action_cooldown_ms', 'cannon_center_only', 'tower_damage_all',
            'source_radius_reach', 'target_lock_in_range', 'hog_sight_clipping')
    profile = {key: trial[key] for key in keys}
    profile['shaping_gamma'] = trial.get('gamma', .999)
    env = IdleArena(seed=seed, action_delay_ms=trial['action_delay_ms'],
                    continuous_profile=profile, spell_waste_weight=weight,
                    tower_support_weight=trial['tower_support_weight'],
                    crown_only_weight=0., dormant_king_weight=0.)
    env.reset(seed=seed)
    # Reach full bank through actual regeneration; no injected elixir or hand.
    while env.match.players[1].elixir < 10000:
        env.step(0)
    before = env._metrics()
    slot = env.match.players[1].hand.index(spell)
    action = 1 + slot * 576 + 16 * 18 + 8  # Empty river, far from every tower.
    if not env._mask()[action]:
        raise ValueError('Probe spell placement must be legal')
    rewards = []
    for index in range(120):  # 30 seconds: resolve spell and refill both banks.
        _, reward, terminated, truncated, _ = env.step(action if cast and index == 0 else 0)
        if terminated or truncated:
            raise ValueError('Unexpected terminal diagnostic')
        rewards.append(reward)
    after = env._metrics()
    if (after['own_fraction_loss'] or after['enemy_fraction_damage'] or
            after['own_tower_damage'] or env.pending or
            env.match.players[1].elixir != 10000 or
            after['empty_spell_counts'].get(spell, 0) != int(cast) or
            sum(after['accepted_card_counts'].values()) != int(cast)):
        raise ValueError('Diagnostic must contain exactly one resolved empty cast or WAIT')
    return dict(reward=sum(rewards),
                discounted_reward=sum(profile['shaping_gamma']**i*r for i, r in enumerate(rewards)),
                overflow=after['elixir_overflow']-before['elixir_overflow'],
                empty_spell_elixir=after['empty_spell_elixir'],
                final_bank=env.match.players[1].elixir/1000,
                decision_count=len(rewards), start_ms=before['elapsed_ms'], end_ms=after['elapsed_ms'])


def probe(protocol_path):
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    trial = protocol['cases'][0]
    rows = []
    for spell in ('fireball', 'zap'):
        for weight in (.1, .3, .5):
            wait, cast = (branch(trial, spell, weight, cast) for cast in (False, True))
            rows.append(dict(spell=spell, spell_waste_weight=weight, wait=wait, cast=cast,
                             cast_minus_wait=cast['reward']-wait['reward'],
                             discounted_cast_minus_wait=cast['discounted_reward']-wait['discounted_reward']))
    return dict(schema='empty-spell-incentive-probe-v1',
                protocol_path=str(protocol_path), protocol_sha256=file_sha(protocol_path),
                source_sha256=file_sha(__file__), rows=rows,
                reward_source_sha256={name:file_sha(name) for name in (
                    'sim/continuous_push.py', 'sim/continuous_spell_waste.py',
                    'sim/continuous_spell_economy.py', 'sim/tactical_rewards.py',
                    'sim/continuous_tower_support.py', 'sim/match.py')},
                overflow_penalty=trial['overflow_penalty'],
                scope='Controlled idle TRAIN arena; no enemies, no learned policy, no win-rate claim',
                interpretation='Positive cast-minus-WAIT means shaping favors wasting a spell at full bank in this isolated state',
                limitations=['Hand/cycle differs after cast', 'Does not prove the cause of learned behavior',
                             'Does not measure useful casts, threats, finishing damage or full-match quality'],
                active_training_changed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = probe(args.protocol)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([dict(spell=r['spell'], weight=r['spell_waste_weight'],
                           delta=r['cast_minus_wait']) for r in result['rows']]))
