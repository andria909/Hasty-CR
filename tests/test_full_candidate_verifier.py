from collections import Counter
from copy import deepcopy
import itertools
import json

import pytest
from scripts.attack import verify_full_candidate_evaluation as module
from scripts.attack.evaluate_full_candidates import schedule, sha, summarize_test
from tests.test_full_candidate_evaluator import fixture


def metric():
    return dict(accepted_card_counts={'hog_rider': 1},
                accepted_card_lane_counts={'left': {'hog_rider': 1}, 'right': {}},
                own_tower_fraction_loss=dict(left=0., right=.1, king=0.),
                missed_fireballs=0, cluster_fireballs=0, elixir_overflow=1., illegal_actions=0)


def part(job, protocol, blueprint):
    if job['kind'] == 'test':
        test = next(t for t in blueprint['test_sets'] if t['id'] == job['test'])
        rows = [dict(metric(), task=task, family_id=family, seed=seed, test_set=test['id'], outcome='win' if seed % 2 else 'loss')
                for task, family, seed in itertools.product(blueprint['tasks'], blueprint['families'], test['seeds'])]
        return dict(kind='test', model=next(m for m in protocol['models'] if m['id'] == job['model']),
                    test_set=test['id'], rows=rows, summary=summarize_test(rows),
                    by_task_family={f'{task}/{family}': summarize_test([r for r in rows if r['task'] == task and r['family_id'] == family])
                                    for task in blueprint['tasks'] for family in blueprint['families']})
    rows = [dict(seed=seed, candidate_side=seat, outcome='win' if seat == 1 else 'loss', elapsed_ms=200000,
                 public_metrics={s: metric() for s in ('1', '-1')}, accepted_cards={s: metric()['accepted_card_counts'] for s in ('1', '-1')},
                 elixir_overflow={'1': 1., '-1': 2.}, rejected_commands={})
            for seed, seat in itertools.product(blueprint['tournament']['seeds'], blueprint['tournament']['seats'])]
    return dict(kind='duel', pair=job['pair'], games=len(rows), rows=rows, counts=dict(Counter(r['outcome'] for r in rows)))


def setup(tmp_path):
    protocol = fixture(tmp_path)
    blueprint = module.validate_protocol(protocol, tmp_path)
    return protocol, blueprint, schedule(protocol, blueprint)


def test_complete_synthetic_manifest_verified_not_real_quality(tmp_path):
    protocol, blueprint, jobs = setup(tmp_path)
    path = tmp_path / 'protocol.json'; path.write_text(json.dumps(protocol))
    output = tmp_path / 'output'; output.mkdir()
    for job in jobs:
        (output / job['file']).write_text(json.dumps(part(job, protocol, blueprint)))
    result = dict(schema='full-candidate-independent-evaluation-v1', status='complete', protocol_sha256=sha(path),
                  blueprint_sha256=protocol['blueprint_sha256'], models=protocol['models'], total_games=4200,
                  files={j['file']: sha(output / j['file']) for j in jobs},
                  test_used_for_selection=False, quality_accepted=False, live_checkpoint_changed=False)
    (output / 'result.json').write_text(json.dumps(result))
    verified = module.verify(path, tmp_path, output)
    assert verified['total_games_verified'] == 4200
    assert set(verified['tests']) == {'scripted_stress', 'legal_random_compatibility'}
    assert len(verified['duels']) == 10
    assert verified['duels'][0]['uncertainty']['interval95'] == [.5, .5]
    assert verified['quality_accepted'] is False
    (output / jobs[0]['file']).write_text('{}')
    with pytest.raises(ValueError, match='digest'): module.verify(path, tmp_path, output)


@pytest.mark.parametrize('kind', ['test', 'duel'])
def test_duplicate_context_cannot_replace_missing_game(tmp_path, kind):
    protocol, blueprint, jobs = setup(tmp_path)
    job = next(j for j in jobs if j['kind'] == kind)
    report = part(job, protocol, blueprint)
    report['rows'][0] = deepcopy(report['rows'][1])
    with pytest.raises(ValueError, match='coverage'): module.validate_part(report, job, protocol, blueprint)


@pytest.mark.parametrize('change,pattern', [
    ('summary', 'summary'), ('lane', 'lane totals'), ('nan', 'resource'), ('fireball', 'exceed'), ('model', 'identity')])
def test_reject_inconsistent_test_metrics(tmp_path, change, pattern):
    protocol, blueprint, jobs = setup(tmp_path)
    job = jobs[0]; report = deepcopy(part(job, protocol, blueprint))
    if change == 'summary': report['summary']['points'] = .99
    if change == 'lane': report['rows'][0]['accepted_card_lane_counts']['left'] = {}
    if change == 'nan': report['rows'][0]['elixir_overflow'] = float('nan')
    if change == 'fireball': report['rows'][0]['missed_fireballs'] = 1
    if change == 'model': report['model']['sha256'] = 'foreign'
    with pytest.raises(ValueError, match=pattern): module.validate_part(report, job, protocol, blueprint)


def test_uncertainty_uses_seed_blocks_not_duplicate_games():
    rows = [dict(seed=s, outcome='win' if s % 2 else 'loss') for s in range(30)]
    single = module.seed_interval(rows)
    repeated = module.seed_interval(rows * 12)
    assert single['interval95'] == repeated['interval95']
    assert repeated['independent_seed_blocks'] == 30
    assert repeated['games'] == 360
