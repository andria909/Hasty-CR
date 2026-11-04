from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.kaggle.prepare_credit_assignment_pilot import validate_protocol, VARIANT
from scripts.kaggle.preflight_global_full_pair import validate_specialized_preflight


def protocol():
    p = json.loads(Path('data/ppo_tuning/credit_assignment_design_20261104.json').read_text())
    p['study_variant'] = VARIANT
    return p


@pytest.mark.parametrize('key,value', [
    ('gae_lambda', .999), ('card_balance_weight', .1), ('gamma', .99),
    ('tower_support_weight', 0.), ('spell_waste_weight', .1),
    ('own_hp_penalty', 10.), ('focus', 'air_golem'), ('action_delay_ms', 225),
])
def test_credit_factorial_rejects_confounding_changes(key, value):
    p = protocol()
    validate_protocol(p)
    p['cases'][3][key] = value
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_credit_factorial_requires_two_distinct_seeds_and_fixed_budget():
    for key, value in [('pilot_steps_each', 102400), ('pilot_seeds', [472, 472]),
                       ('paired_controls', {})]:
        p = protocol()
        p[key] = value
        with pytest.raises(ValueError):
            validate_protocol(p)


@pytest.mark.parametrize('key', ['gae_lambda', 'card_balance_weight'])
def test_full_preflight_requires_effective_optimizer_treatment(key):
    trial = protocol()['cases'][3]
    model = dict(id='parent', sha256='fixed')
    entry = dict(config=dict(models=[model]), selected=dict(trial=trial))
    report = dict(explicit_initial_model=model, economy_metrics_verified=True,
                  long_economy_full_verified=True, actual_focus='uniform', full_steps=4096000,
                  actual_dormant_king_weight=0., actual_spell_waste_weight=.3,
                  actual_tower_support_weight=.15, crown_validation_games=12,
                  actual_gae_lambda=.98, actual_card_balance_weight=0.)
    validate_specialized_preflight(entry, report, VARIANT)
    wrong = deepcopy(report)
    wrong['actual_' + key] = None
    with pytest.raises(ValueError, match='credit-assignment'):
        validate_specialized_preflight(entry, wrong, VARIANT)


def test_old_air_grid_still_rejects_disabled_balance():
    from scripts.kaggle.prepare_air_exposure_pilot import validate_protocol as validate_air
    p = json.loads(Path('data/ppo_tuning/air_exposure_design_20261102.json').read_text())
    p['study_variant'] = 'air_exposure_v1'
    for row in p['cases']:
        row['card_balance_weight'] = 0.
    with pytest.raises(ValueError):
        validate_air(p)


def test_credit_dashboard_does_not_invent_progress(tmp_path):
    from scripts.defense.current_dashboard import render_credit_assignment
    state = dict(root=str(tmp_path/'job'),kernel='owner/credit',submitted=True)
    root = tmp_path/'job/live_stream'
    root.mkdir(parents=True)
    (root/'progress.json').write_text(json.dumps(dict(kernel='owner/credit',trials={
        'pilot_longer_s472':dict(steps=204800,target_steps=409600)})))
    text = render_credit_assignment(tmp_path,state)
    assert '| GAE 0,98; баланс 0,03 | ≥50.0% | нет счётчика |' in text
    assert '100% проверено' not in text
    (root/'progress.json').write_text(json.dumps(dict(kernel='wrong/job',trials={
        'pilot_longer_s472':dict(steps=409600,target_steps=409600)})))
    assert '100%' not in render_credit_assignment(tmp_path,state)
