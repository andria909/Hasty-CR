"""Verify every precommitted game before reporting independent evaluation results."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scripts.attack.evaluate_full_candidates import schedule, sha, summarize_test, validate_protocol
from sim.env import DECK_26


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, float):
        return isinstance(actual, (float, int)) and not isinstance(actual, bool) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    return actual == expected


def nonnegative(value, integer=False):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 and (not integer or isinstance(value, int))


def check_cards(metrics):
    counts = metrics['accepted_card_counts']
    lanes = metrics['accepted_card_lane_counts']
    require(set(counts) <= set(DECK_26) and set(lanes) == {'left', 'right'}, 'Unknown card or lane')
    for values in (counts, *lanes.values()):
        require(set(values) <= set(DECK_26) and all(nonnegative(v, True) for v in values.values()), 'Invalid card count')
    require(all(counts.get(c, 0) == sum(lanes[l].get(c, 0) for l in lanes) for c in DECK_26), 'Card lane totals differ')
    losses = metrics['own_tower_fraction_loss']
    require(set(losses) == {'left', 'right', 'king'} and all(nonnegative(v) and v <= 1 for v in losses.values()), 'Invalid tower loss')
    for key in ('missed_fireballs', 'cluster_fireballs'):
        require(nonnegative(metrics[key], True), 'Invalid fireball count')
    require(metrics['missed_fireballs'] + metrics['cluster_fireballs'] <= counts.get('fireball', 0), 'Fireball outcomes exceed plays')


def validate_part(report, job, protocol, blueprint):
    require(report['kind'] == job['kind'], 'Wrong part kind')
    rows = report['rows']
    require(len(rows) == job['games'], 'Incomplete part')
    require(all(r['outcome'] in ('win', 'loss', 'draw') for r in rows), 'Unknown outcome')
    if job['kind'] == 'test':
        entry = next(m for m in protocol['models'] if m['id'] == job['model'])
        test = next(t for t in blueprint['test_sets'] if t['id'] == job['test'])
        require(report['model'] == entry and report['test_set'] == test['id'], 'Wrong model or test identity')
        expected = set(itertools.product(blueprint['tasks'], blueprint['families'], test['seeds']))
        actual = [(r['task'], r['family_id'], r['seed']) for r in rows]
        require(len(actual) == len(set(actual)) and set(actual) == expected, 'Test context coverage differs')
        for row in rows:
            require(row['test_set'] == test['id'], 'Mixed test sets')
            check_cards(row)
            require(nonnegative(row['elixir_overflow']) and nonnegative(row['illegal_actions'], True), 'Invalid resource metrics')
        require(same(report['summary'], summarize_test(rows)), 'Test summary differs from rows')
        grouped = {f'{task}/{family}': summarize_test([r for r in rows if r['task'] == task and r['family_id'] == family]) for task in blueprint['tasks'] for family in blueprint['families']}
        require(same(report['by_task_family'], grouped), 'Family summary differs from rows')
    else:
        require(report['pair'] == job['pair'] and report['games'] == len(rows), 'Wrong duel identity')
        expected = set(itertools.product(blueprint['tournament']['seeds'], blueprint['tournament']['seats']))
        actual = [(r['seed'], r['candidate_side']) for r in rows]
        require(len(actual) == len(set(actual)) and set(actual) == expected, 'Duel seed/seat coverage differs')
        require(report['counts'] == dict(Counter(r['outcome'] for r in rows)), 'Duel counts differ')
        for row in rows:
            require(set(row['public_metrics']) == {'1', '-1'}, 'Both player metrics required')
            require(nonnegative(row['elapsed_ms']) and row['elapsed_ms'] <= 310000, 'Invalid game duration')
            for side in ('1', '-1'):
                check_cards(row['public_metrics'][side])
                require(row['accepted_cards'][side] == row['public_metrics'][side]['accepted_card_counts'], 'Duel card totals differ')
                require(nonnegative(row['elixir_overflow'][side]), 'Invalid overflow')
            # In-memory duel counters use integer sides; JSON encodes the same keys as strings.
            rejected=row['rejected_commands']
            require(all((type(k)is int and k in (1,-1)) or (type(k)is str and k in ('1','-1')) for k in rejected)
                    and len({str(k)for k in rejected})==len(rejected)
                    and all(nonnegative(v,True)for v in rejected.values()), 'Invalid rejected commands')
    return rows


def seed_interval(rows, *, replicates=10000, seed=270010):
    """Resample whole seed blocks, retaining correlated families and seats."""
    groups = {}
    for row in rows:
        groups.setdefault(row['seed'], []).append({'win': 1., 'draw': .5, 'loss': 0.}[row['outcome']])
    sizes = {len(v) for v in groups.values()}
    require(len(sizes) == 1 and len(groups) >= 2, 'Balanced seed blocks required')
    values = np.array([np.mean(groups[k]) for k in sorted(groups)])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return dict(points=float(values.mean()), interval95=[float(v) for v in np.quantile(samples, [.025, .975])],
                games=len(rows), independent_seed_blocks=len(values), bootstrap_replicates=replicates)


def verify(protocol_path, weights, output, *, expected_source_sha256=None):
    protocol_path, output = Path(protocol_path), Path(output)
    protocol = json.loads(protocol_path.read_text())
    blueprint = validate_protocol(protocol, weights)
    jobs = schedule(protocol, blueprint)
    from scripts.attack.full_evaluation_recovery import identity
    source_digest = sha(output / 'source_manifest.json')
    if expected_source_sha256 is not None:
        require(source_digest == expected_source_sha256, 'Submitted evaluation source differs')
    require(json.loads((output / 'run_identity.json').read_text()) == identity(protocol_path, source_digest, jobs), 'Run identity differs')
    result = json.loads((output / 'result.json').read_text())
    require(result['source_manifest_sha256'] == source_digest, 'Result source differs')
    require(result['schema'] == 'full-candidate-independent-evaluation-v1' and result['status'] == 'complete', 'Complete evaluation required')
    require(result['protocol_sha256'] == sha(protocol_path) and result['blueprint_sha256'] == protocol['blueprint_sha256'] and result['models'] == protocol['models'], 'Evaluation provenance differs')
    require(all(result[k] is False for k in ('test_used_for_selection', 'quality_accepted', 'live_checkpoint_changed')), 'Unexpected selection or promotion')
    require(set(result['files']) == {j['file'] for j in jobs}, 'Scheduled part manifest differs')
    tests, duels = {}, []
    total = 0
    for job in jobs:
        path = output / job['file']
        require(sha(path) == result['files'][job['file']], 'Part digest differs')
        receipt = json.loads((output / (job['file'] + '.receipt.json')).read_text())
        require(receipt == dict(schema='full-evaluation-part-receipt-v1', file=job['file'], sha256=result['files'][job['file']], games=job['games'], protocol_sha256=sha(protocol_path), source_manifest_sha256=source_digest), 'Part receipt differs')
        part = json.loads(path.read_text())
        rows = validate_part(part, job, protocol, blueprint)
        total += len(rows)
        interval = seed_interval(rows)
        if job['kind'] == 'test':
            tests.setdefault(job['test'], {})[job['model']] = dict(summary=part['summary'], uncertainty=interval, by_task_family=part['by_task_family'])
        else:
            duels.append(dict(pair=job['pair'], counts=part['counts'], uncertainty=interval,
                              by_seat={str(s): dict(Counter(r['outcome'] for r in rows if r['candidate_side'] == s)) for s in (1, -1)}))
    require(total == result['total_games'] == sum(j['games'] for j in jobs), 'Total budget differs')
    return dict(schema='verified-full-candidate-evaluation-v1', result_sha256=sha(output / 'result.json'),
                source_manifest_sha256=source_digest, total_games_verified=total, tests=tests, duels=duels, quality_accepted=False,
                limitations=['Simulator results are not live win rates', 'Test sets reported separately', 'Intervals resample whole seed blocks; no multiple-comparison correction'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('protocol', 'weights', 'output', 'report'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = verify(args.protocol, args.weights, args.output)
    with Path(args.report).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
