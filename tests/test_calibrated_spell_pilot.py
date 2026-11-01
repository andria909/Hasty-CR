from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest
import requests

from scripts.kaggle.prepare_calibrated_spell_pilot import CASES, VARIANT, validate_protocol
from scripts.kaggle import collect_final_evaluation as collector
from tests.test_policy_adaptation_pilot import protocol as adaptation_protocol


def protocol():
    p = adaptation_protocol()
    base = dict(p['cases'][0], overflow_penalty=.3, missed_fireball_penalty=0.)
    p.update(study_variant=VARIANT, pilot_seeds=[422, 423],
             cases=[dict(base, id=name, initial_id='parent', spell_waste_weight=weight)
                    for name, weight in CASES])
    return p


def test_grid_rejects_confounding_or_duplicate_penalty():
    p = protocol()
    validate_protocol(p)
    for key, value in [('overflow_penalty', .5), ('spell_waste_weight', .2),
                       ('initial_id', 'bc8000'), ('missed_fireball_penalty', .1),
                       ('supervised_replay_weight', .1)]:
        bad = deepcopy(p)
        bad['cases'][2][key] = value
        with pytest.raises(ValueError):
            validate_protocol(bad)


def test_collector_preserves_all_selected_and_last_weights(monkeypatch, tmp_path):
    prefix = 'continuous_focus/'
    data = {prefix+'result.json': b'{}', prefix+'parent_reference.json': b'{}'}
    for name, _ in CASES:
        for seed in (422, 423):
            run = f'pilot_{name}_s{seed}'
            data[prefix+run+'_summary.json'] = json.dumps(dict(id=run, checkpoint=run+'/update_000050.pt')).encode()
            for artifact in ('config.json', 'metrics.jsonl', 'training_result.json', 'update_000050.pt', 'update_000100.pt'):
                data[prefix+run+'/'+artifact] = b'{}'
    data[prefix+'pilot_control_s424/update_000100.pt'] = b'exclude'
    monkeypatch.setattr(collector, 'list_outputs', lambda *a: ({name: NS(url=name) for name in data}, []))
    monkeypatch.setattr(requests, 'get', lambda url, **kw: NS(content=data[url], raise_for_status=lambda: None))
    root = tmp_path/'collected'
    receipt = collector.collect(NS(kernels_status=lambda k: 'COMPLETE'), 'owner/job', root, 'calibrated-spell-pilot')
    assert receipt['complete'] and len(list(root.glob('*/update_*.pt'))) == 16
    assert not (root/'pilot_control_s424').exists()


def test_actual_spell_incentive_reverses_when_waste_exceeds_overflow():
    from scripts.attack.probe_empty_spell_incentive import branch
    p = dict(protocol()['cases'][0], fireball_value_reward=.25, reserve_potential_weight=.1,
             own_hp_penalty=6., barbarian_train_fraction=.35, lane_reflection_fraction=.5)
    for spell, cost in [('fireball', 4), ('zap', 2)]:
        for weight in (.1, .5):
            wait = branch(p, spell, weight, False)
            cast = branch(p, spell, weight, True)
            assert cast['reward']-wait['reward'] == pytest.approx(cost*(.3-weight), abs=1e-8)
            assert cast['empty_spell_elixir'] == cost
            assert cast['final_bank'] == wait['final_bank'] == 10


def test_dashboard_does_not_turn_remote_acceptance_into_training_completion(tmp_path):
    from scripts.defense.current_dashboard import render_economy_experiments
    state = dict(root=str(tmp_path/'job'), kernel='owner/cost', status='accepted')
    (tmp_path/'job/live_stream').mkdir(parents=True)
    stream = tmp_path/'job/live_stream/progress.json'
    stream.write_text(json.dumps(dict(kernel='owner/other', trials={'pilot_control_s422': dict(steps=102400)})))
    text = render_economy_experiments(tmp_path, state, None)
    assert '100%' not in text and 'нет счётчика' in text
    stream.write_text(json.dumps(dict(kernel='owner/cost', trials={'pilot_control_s422': dict(steps=25600)})))
    text = render_economy_experiments(tmp_path, state, None)
    assert '≥25.0%' in text and '100% проверено' not in text
