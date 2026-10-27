"""Reconstruct undiscounted TRAIN returns without changing frozen experiments."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def timing(gamma, gae_lambda, decision_ms):
    if not 0 < gamma < 1 or not 0 < gae_lambda <= 1 or decision_ms <= 0:
        raise ValueError('Finite discount and positive decision interval required')
    return dict(decision_ms=decision_ms, gamma=gamma, gae_lambda=gae_lambda,
        discounted_terminal_weight_at_180s=gamma ** (180000 / decision_ms),
        discounted_terminal_weight_at_300s=gamma ** (300000 / decision_ms),
        discount_half_life_seconds=math.log(.5)/math.log(gamma)*decision_ms/1000,
        gae_residual_half_life_seconds=math.log(.5)/math.log(gamma*gae_lambda)*decision_ms/1000,
        interpretation='GAE residual trace, not total credit horizon: bootstrap value can carry later outcomes')


def components(row, profile):
    if row.get('split') != 'train' or row['outcome'] not in ('win', 'loss', 'draw'):
        raise ValueError('Known completed TRAIN outcomes required')
    result = dict(enemy_damage=profile['enemy_hp_reward']*row['enemy_fraction_damage'],
        own_damage=-profile['own_hp_penalty']*row['own_fraction_loss'],
        overflow=-profile['overflow_penalty']*row['elixir_overflow'],
        illegal=-.1*row['illegal_actions'],
        missed_fireball=-profile.get('missed_fireball_penalty',0.)*row.get('missed_fireball_elixir',0.),
        board_potential=row.get('board_shaping_return',0.),
        reserve_potential=row.get('reserve_shaping_return',0.),
        fireball_value=profile.get('fireball_value_reward',0.)*row.get('fireball_exchange_value',0.),
        terminal=profile['win_reward'] if row['outcome']=='win' else
            -profile['loss_penalty'] if row['outcome']=='loss' else 0.)
    if not math.isclose(sum(result.values()), row['episode_return'], abs_tol=1e-5):
        raise ValueError('Return does not match telescoped reward components')
    return result


def analyze(root, decision_ms, start_update=3000):
    root=Path(root); config_path=root/'config.json'; metrics_path=root/'metrics.jsonl'
    config=json.loads(config_path.read_text()); profile=config['continuous_profile']
    groups=defaultdict(list); digest=hashlib.sha256(); checked=0
    for raw in metrics_path.open('rb'):
        digest.update(raw); record=json.loads(raw)
        for row in record.get('episodes', []):
            terms=components(row, profile); checked+=1
            if record['update']>=start_update:
                groups[row['family_id']].append((row, terms))
    def summarize(rows):
        if not rows: raise ValueError('No completed training episodes in audit window')
        cards=Counter(); seconds=sum(r['elapsed_ms']/1000 for r,_ in rows)
        for row,_ in rows: cards.update(row['accepted_card_counts'])
        return dict(games=len(rows),
            mean_reward_components={k:statistics.mean(t[k] for _,t in rows) for k in rows[0][1]},
            mean_return=statistics.mean(r['episode_return'] for r,_ in rows),
            overflow_affected_games=sum(r['elixir_overflow']>0 for r,_ in rows),
            illegal_affected_games=sum(r['illegal_actions']>0 for r,_ in rows),
            accepted_plays_per_game={c:n/len(rows) for c,n in sorted(cards.items())},
            accepted_plays_per_minute={c:n*60/seconds for c,n in sorted(cards.items())},
            no_musketeer_fraction=statistics.mean(r['accepted_card_counts'].get('musketeer',0)==0 for r,_ in rows),
            no_hog_fraction=statistics.mean(r['accepted_card_counts'].get('hog_rider',0)==0 for r,_ in rows))
    return dict(source=str(root), config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        metrics_sha256=digest.hexdigest(), all_episode_returns_checked=checked,
        start_update=start_update, timing=timing(config['gamma'], config['gae_lambda'], decision_ms),
        overall=summarize([x for rows in groups.values() for x in rows]),
        by_family={k:summarize(v) for k,v in sorted(groups.items())},
        limits=['Undiscounted return decomposition; temporal reward sequence not stored',
            'Card-balance JS is an optimizer auxiliary loss, not part of environment return',
            'Completed TRAIN episodes; no checkpoint selection or live claim'])


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('roots', nargs='+')
    p.add_argument('--decision-ms', type=int, required=True,
        help='Interval in the frozen training source, not current working-tree source')
    p.add_argument('--start-update', type=int, default=3000); p.add_argument('--output', required=True)
    a=p.parse_args(); result=dict(schema='continuous-train-reward-audit-v1',
        models=[analyze(r,a.decision_ms,a.start_update) for r in a.roots],
        test_used_for_selection=False, remote_training_changed=False)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m['source']:m['overall'] for m in result['models']}))
