"""Stdlib summaries of recorded outcomes, independent of training/optimization.

Unknown metrics remain null and every rate carries a known-value denominator.
Compact stdout excludes sampler/config/trajectory arrays; raw JSONL stays intact.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from numbers import Real
import statistics

LANES = ('left', 'right', 'king')


def number(value):
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) else None


def mean(values):
    return statistics.mean(values) if values else None


def metrics_of(record):
    if not isinstance(record, dict):
        return {}
    metrics = record.get('metrics', record)
    return metrics if isinstance(metrics, dict) else {}


def known_no_damage(metrics):
    explicit = metrics.get('no_tower_damage')
    if isinstance(explicit, bool):
        return explicit
    for field in ('own_tower_hp_loss', 'own_tower_fraction_loss'):
        losses = metrics.get(field)
        if not isinstance(losses, dict):
            continue
        values = {lane: number(losses.get(lane)) for lane in LANES}
        if any(value is not None and value > 0 for value in values.values()):
            return False
        if all(value is not None and value == 0 for value in values.values()):
            return True
    return None


def summarize_episodes(records, returns=None, *, by_difficulty=True):
    """Summarize completed episode records without inferring return from HP.

    Accepts raw environment metrics or {'metrics':..., 'reward'/'return':...}.
    GRPO may provide explicit trajectory returns. PPO training metrics lack
    cumulative rewards, so their mean_return correctly remains None.
    """
    records = list(records or [])
    supplied = list(returns) if returns is not None else None
    no_damage = []; hp = []; fractions = []; princess = []; king = []; reasons = []
    observed_returns = []; groups = defaultdict(list); grouped_returns = defaultdict(list)
    for index, record in enumerate(records):
        metrics = metrics_of(record)
        no_hit = known_no_damage(metrics)
        if no_hit is not None:
            no_damage.append(no_hit)
        for field, target in (('own_tower_hp_loss', hp), ('own_tower_fraction_loss', fractions)):
            losses = metrics.get(field)
            if isinstance(losses, dict):
                values = [number(losses.get(lane)) for lane in LANES]
                if all(value is not None and value >= 0 for value in values):
                    target.append(sum(values))
        alive = metrics.get('own_towers_alive')
        if isinstance(alive, dict):
            princess.extend(alive[lane] for lane in ('left', 'right') if isinstance(alive.get(lane), bool))
            if isinstance(alive.get('king'), bool):
                king.append(alive['king'])
        reason = metrics.get('terminal_reason')
        if isinstance(reason, str) and reason:
            reasons.append(reason)
        if supplied is not None:
            value = number(supplied[index]) if index < len(supplied) else None
        elif isinstance(record, dict) and 'metrics' in record:
            value = number(record.get('return', record.get('reward')))
        else:
            value = None
        if value is not None:
            observed_returns.append(value)
        difficulty = metrics.get('difficulty')
        difficulty = difficulty if isinstance(difficulty, str) and difficulty else 'unknown'
        groups[difficulty].append(record)
        grouped_returns[difficulty].append(value)
    if supplied is not None:
        observed_returns = [number(value) for value in supplied if number(value) is not None]
    result = dict(n=len(records), no_damage=sum(no_damage) if no_damage else None,
                  no_damage_pct=100*sum(no_damage)/len(no_damage) if no_damage else None,
                  no_damage_known=len(no_damage),
                  mean_hp_loss=mean(hp), hp_loss_known=len(hp),
                  mean_total_tower_fraction_loss=mean(fractions), fraction_loss_known=len(fractions),
                  princess_deaths=princess.count(False) if princess else None,
                  princess_known=len(princess),
                  princess_survival_pct=100*sum(princess)/len(princess) if princess else None,
                  king_deaths=king.count(False) if king else None, king_known=len(king),
                  king_survival_pct=100*sum(king)/len(king) if king else None,
                  timeouts=reasons.count('timeout') if reasons else None,
                  invalid=sum(reason.startswith('invalid_script') for reason in reasons) if reasons else None,
                  terminal_known=len(reasons), mean_return=mean(observed_returns),
                  return_std=statistics.pstdev(observed_returns) if observed_returns else None,
                  return_known=len(observed_returns),
                  return_alignment_complete=supplied is None or len(supplied) == len(records))
    if by_difficulty:
        result['by_difficulty'] = {difficulty: summarize_episodes(group, grouped_returns[difficulty], by_difficulty=False)
                                   for difficulty, group in sorted(groups.items())}
    return result


def summarize_update(record, algorithm):
    """Train and validation summaries kept separate; never claim paired curricula."""
    algorithm = algorithm.upper()
    if algorithm not in ('PPO', 'GRPO'):
        raise ValueError('Expected PPO or GRPO')
    episodes = record.get('episodes' if algorithm == 'PPO' else 'metrics', [])
    train = summarize_episodes(episodes, record.get('returns') if algorithm == 'GRPO' else None)
    validation = None
    if isinstance(record.get('validation'), dict):
        raw = record['validation']
        validation = summarize_episodes(raw.get('episodes', raw.get('metrics', [])))
        # GRPO validation stores only the aggregate return, not individual
        # returns. Preserve it without inventing a return distribution.
        if validation['mean_return'] is None:
            validation['mean_return'] = number(raw.get('mean_return', raw.get('mean_reward')))
    return dict(train=train, validation=validation)


def _compact_episode_summary(summary):
    fields = ('n', 'no_damage', 'no_damage_pct', 'no_damage_known', 'mean_hp_loss',
              'hp_loss_known', 'princess_deaths', 'princess_known', 'king_deaths',
              'king_known', 'timeouts', 'invalid', 'terminal_known')
    return {field: summary[field] for field in fields}


def _rounded(value):
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    return round(value, 4) if isinstance(value, float) else value


def compact_stdout_lines(record, algorithm):
    """One short global train line, plus a separate validation line if present.

    Full per-difficulty details are available through summarize_update, while
    complete original episodes/returns/sampler remain in the raw JSONL record.
    """
    summaries = summarize_update(record, algorithm)
    train = summaries['train']
    head = dict(event='train', algorithm=algorithm.upper(), update=record.get('update'),
                steps=record.get('step', record.get('environment_steps')),
                mean_return=train['mean_return'], return_std=train['return_std'],
                loss=number(record.get('loss')), kl=number(record.get('kl')),
                degenerate=record.get('degenerate') if isinstance(record.get('degenerate'), bool) else None,
                episodes=_compact_episode_summary(train),
                elapsed_seconds=number(record.get('elapsed_seconds')))
    lines = [json.dumps(_rounded(head), separators=(',', ':'), allow_nan=False)]
    validation = summaries['validation']
    if validation is not None:
        value = dict(event='validation', algorithm=algorithm.upper(), update=record.get('update'),
                     mean_return=validation['mean_return'], return_std=validation['return_std'],
                     episodes=_compact_episode_summary(validation))
        lines.append(json.dumps(_rounded(value), separators=(',', ':'), allow_nan=False))
    return lines
