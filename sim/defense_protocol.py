"""Frozen, auditable defense acceptance; reward is never a quality metric."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

V25_PATH = 'checkpoints/portable_v25/portable_v25_cannon_economy_best.pt'
V25_SHA256 = 'd65ab55893592c0e4655a1882db459c4a7fb1ebd9e10afaf2daa4d2b369fc972'


@dataclass(frozen=True)
class EvaluationProtocol:
    test_scenario_ids: tuple[str, ...]
    baseline_ids: tuple[str, ...] = ('v25',)
    paired_seeds: tuple[int, ...] = (710001, 710002, 710003, 710004, 710005)
    defense_seeds: tuple[int, ...] = (810001, 810002, 810003)
    medium_max_hp_loss: float = .02
    medium_threshold_confirmed: bool = False
    required_observed_winrate: float = .8
    schema: str = 'defense-evaluation-v1'

    def __post_init__(self):
        for name in ('test_scenario_ids', 'baseline_ids', 'paired_seeds', 'defense_seeds'):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not value or len(set(value)) != len(value):
                raise ValueError(f'{name} must be a nonempty tuple without duplicates')
        if 'v25' not in self.baseline_ids:
            raise ValueError('v25 is always a required baseline')
        if not 10 <= 2 * len(self.paired_seeds) <= 50:
            raise ValueError('Require 10–50 games per baseline, paired across both sides')
        if not 0 <= self.medium_max_hp_loss <= 1 or not 0 < self.required_observed_winrate <= 1:
            raise ValueError('Invalid quality threshold')

    def to_dict(self):
        # JSON roundtrip yields only portable types, not tuples/dataclasses.
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ('test_scenario_ids', 'baseline_ids', 'paired_seeds', 'defense_seeds'):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def paired_schedule(protocol):
    return [dict(baseline_id=baseline, seed=seed, candidate_side=side)
            for baseline in protocol.baseline_ids for seed in protocol.paired_seeds for side in (-1, 1)]


def wilson_interval(wins, games, z=1.959963984540054):
    if games < 0 or not 0 <= wins <= games:
        raise ValueError('Invalid win counts')
    if games == 0:
        return [0., 1.]
    p = wins / games
    denominator = 1 + z*z/games
    center = (p + z*z/(2*games)) / denominator
    half = z * math.sqrt(p*(1-p)/games + z*z/(4*games*games)) / denominator
    return [max(0., center-half), min(1., center+half)]


def load_baseline_allowlist(manifest, *, project_root, verify_hashes=True):
    """Explicit IDs only; never choose models from names, mtimes or scan results.

    Format: {schema: 'defense-baselines-v1', checkpoints:
      [{id: 'v25', path: 'checkpoints/...pt', sha256: '<64hex>'}, ...]}.
    No weight deserialization or credential access is performed.
    """
    root = Path(project_root).resolve()
    data = json.loads(Path(manifest).read_text(encoding='utf-8'))
    if data.get('schema') != 'defense-baselines-v1':
        raise ValueError('Unsupported baseline allowlist schema')
    result = {}
    for row in data['checkpoints']:
        identifier, relative, digest = row['id'], row['path'], row['sha256']
        path = (root / relative).resolve()
        if identifier in result or not path.is_relative_to(root / 'checkpoints'):
            raise ValueError('Duplicate baseline ID or path outside checkpoints')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid checkpoint SHA256')
        if not path.is_file():
            raise ValueError(f'Missing allowlisted checkpoint: {identifier}')
        if verify_hashes:
            hasher = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    hasher.update(chunk)
            if hasher.hexdigest() != digest:
                raise ValueError(f'Checkpoint hash mismatch: {identifier}')
        result[identifier] = dict(id=identifier, path=str(path), sha256=digest)
    if ('v25' not in result or Path(result['v25']['path']) != root / V25_PATH
            or result['v25']['sha256'] != V25_SHA256):
        raise ValueError('Canonical preserved v25 baseline is mandatory')
    return result


def acceptance_gate(protocol, defense_results, head_to_head, *, coverage_complete=False):
    """Compare a frozen TEST protocol; validation tuning belongs elsewhere.

    Defense results are environment metric dicts, or records containing metrics.
    Draws count as nonwins. Passing 80% here is only the observed finite suite
    criterion: a Wilson interval is reported, not a population-rate guarantee.
    """
    failures, unknown = [], []
    if not coverage_complete:
        unknown.append('missing_coverage')
    expected = {(sid, seed) for sid in protocol.test_scenario_ids for seed in protocol.defense_seeds}
    seen = set()
    summaries = {name: {'episodes': 0, 'no_damage': 0, 'unavoidable_episodes': 0,
                        'princess_survived': 0, 'king_survived': 0, 'timeouts': 0,
                        'hp_loss_fractions': [], 'elixir_spent': []}
                 for name in ('simple', 'medium', 'complex', 'impossible')}
    for record in defense_results:
        metrics = record.get('metrics', record)
        key = (metrics.get('scenario_id'), metrics.get('seed'))
        if key not in expected or key in seen:
            unknown.append('unexpected_or_duplicate_defense_episode')
            continue
        seen.add(key)
        if metrics.get('split') != 'test':
            unknown.append('non_test_defense_episode')
        reason = str(metrics.get('terminal_reason') or '')
        if reason.startswith('invalid_script'):
            unknown.append('invalid_script_episode')
            continue
        flags = set(metrics.get('coverage_flags', ()))
        if flags & {'missing_spawn_profiles', 'unmodelled_projectiles'}:
            unknown.append('unresolved_simulator_mechanics')
        difficulty = metrics.get('difficulty')
        if difficulty not in summaries:
            unknown.append('unknown_difficulty')
            continue
        losses = metrics.get('own_tower_fraction_loss')
        alive = metrics.get('own_towers_alive')
        if (not isinstance(losses, dict) or not isinstance(alive, dict)
                or any(lane not in losses or lane not in alive for lane in ('left','right','king'))
                or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in losses.values())):
            unknown.append('missing_or_invalid_quality_metrics')
            continue
        if reason == 'timeout':
            unknown.append('timeouts_require_review')
        elif reason not in ('cleared', 'own_king_destroyed', 'enemy_king_destroyed', 'both_kings_destroyed'):
            unknown.append('unknown_terminal_reason')
        summary = summaries[difficulty]
        summary['episodes'] += 1
        summary['timeouts'] += reason == 'timeout'
        no_damage = all(float(x) == 0 for x in losses.values())
        summary['no_damage'] += no_damage
        unavoidable = bool(metrics.get('unavoidable_tower_damage', False))
        summary['unavoidable_episodes'] += unavoidable
        lane = metrics.get('lane')
        princess = bool(alive.get(lane, False)) if lane in ('left','right') else bool(alive['left'] and alive['right'])
        summary['princess_survived'] += princess
        summary['king_survived'] += bool(alive['king'])
        summary['hp_loss_fractions'].append(dict(losses))
        summary['elixir_spent'].append(metrics.get('elixir_spent'))
        if difficulty == 'simple' and not unavoidable and not no_damage:
            failures.append('simple_avoidable_tower_damage')
        if difficulty == 'medium' and not unavoidable:
            if not protocol.medium_threshold_confirmed:
                unknown.append('untuned_medium_threshold')
            elif max(float(x) for x in losses.values()) > protocol.medium_max_hp_loss:
                failures.append('medium_hp_loss_above_threshold')
    if expected - seen:
        unknown.append('not_enough_defense_episodes')
    planned = {(r['baseline_id'], r['seed'], r['candidate_side']) for r in paired_schedule(protocol)}
    matches = {}
    for row in head_to_head:
        key = (row.get('baseline_id'), row.get('seed'), row.get('candidate_side'))
        if key not in planned or key in matches or row.get('result') not in ('win','loss','draw'):
            unknown.append('unexpected_duplicate_or_invalid_match')
            continue
        matches[key] = row['result']
    h2h = {}
    for baseline in protocol.baseline_ids:
        selected = [outcome for (b, _, _), outcome in matches.items() if b == baseline]
        wins = selected.count('win')
        n = len(selected)
        complete = all(key in matches for key in planned if key[0] == baseline)
        rate = wins/n if n else None
        h2h[baseline] = dict(games=n, wins=wins, draws=selected.count('draw'),
                             observed_winrate=rate, wilson_95=wilson_interval(wins, n),
                             complete=complete, population_80_percent_established=False,
                             interval_assumption='Wilson treats games as independent; paired seeds may correlate outcomes.')
        if not complete:
            unknown.append('not_enough_paired_games:' + baseline)
        elif rate < protocol.required_observed_winrate:
            failures.append('observed_winrate_below_target:' + baseline)
    return dict(status='fail' if failures else 'undetermined' if unknown else 'pass',
                failures=sorted(set(failures)), undetermined_reasons=sorted(set(unknown)),
                protocol_fingerprint=protocol.fingerprint, defense=summaries, head_to_head=h2h,
                interpretation='Finite observed suite; not proof of >=80% population winrate. Complex/stress survival are soft targets.')
