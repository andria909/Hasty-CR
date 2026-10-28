from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import requests

from sim import decision_economy as economy
from sim.continuous_push import ContinuousPushEnv
from scripts.kaggle.prepare_mana_campaign import GRIDS, validate_protocol
from scripts.attack.verify_mana_campaign import audit_economy
from scripts.kaggle import collect_final_evaluation as collector

DESIGN = json.loads((Path(__file__).resolve().parents[1]/
                    'data/ppo_tuning/mana_campaign_20261029.json').read_text())


def test_three_paired_grids_have_identical_controls_and_exact_budget():
    controls = []
    for name, protocol in DESIGN['protocols'].items():
        validate_protocol(protocol)
        controls.append(protocol['cases'][0])
        changed = GRIDS[name][0]
        assert all({k for k in row if row[k] != controls[-1][k]} <= {'id', changed}
                   for row in protocol['cases'])
    assert controls[0] == controls[1] == controls[2]
    assert 3*4*2*102400 == DESIGN['total_steps'] == 2457600


@pytest.mark.parametrize('key,value', [('action_delay_ms', 225), ('hog_sight_clipping', 0),
    ('cannon_center_only', 0), ('tower_support_weight', 0), ('card_balance_weight', .2)])
def test_silent_confound_or_legacy_physics_is_rejected(key, value):
    protocol = deepcopy(DESIGN['protocols']['reserve'])
    protocol['cases'][1][key] = value
    with pytest.raises(ValueError):
        validate_protocol(protocol)


def test_economy_counts_distinguish_wait_availability_and_reservations():
    stats = economy.empty(); mask = np.zeros(2321, dtype=bool); mask[0] = True
    hand = ['hog_rider', 'fireball', 'knight', 'zap']
    economy.record(stats, hand=hand, mask=mask, action=0, native_elixir=5, available_elixir=1)
    mask[1+3*576] = True
    economy.record(stats, hand=hand, mask=mask, action=0, native_elixir=3, available_elixir=3)
    economy.record(stats, hand=hand, mask=mask, action=1+3*576, native_elixir=3, available_elixir=3)
    assert stats['forced_wait'] == stats['voluntary_wait'] == stats['plays'] == 1
    assert stats['native_below4'] == 2 and stats['available_below4'] == 3
    assert stats['cards']['zap'] == dict(in_hand=3, legal=2, selected=1)
    audit_economy([dict(decision_economy=stats)])
    saved = economy.snapshot(stats); stats['cards']['zap']['legal'] += 1
    assert saved['cards']['zap']['legal'] == 2


def test_real_environment_pending_queue_and_snapshot_keep_counts():
    env = ContinuousPushEnv(action_delay_ms=1350)
    env.reset(seed=3)
    env.match.players[1].hand = ['fireball', 'hog_rider', 'musketeer', 'knight']
    env.match.players[1].queue = ['cannon', 'zap', 'skeletons', 'electro_spirit']
    env.match.players[1].elixir = 5000
    action = int(np.flatnonzero(env._mask()[1:577])[0])+1
    env.step(action)
    saved = env.snapshot()
    env.step(0)
    counts = env._metrics()['decision_economy']
    assert counts['decisions'] == 2 and counts['plays'] == 1 and counts['forced_wait'] == 1
    assert counts['available_below4'] == 1 and counts['native_below4'] == 0
    env.restore(saved)
    assert env._metrics()['decision_economy']['decisions'] == 1
    env.step(0)
    assert env._metrics()['decision_economy'] == counts


def test_collector_gets_both_checkpoint_boundaries_but_no_other_campaign(monkeypatch, tmp_path):
    run = 'pilot_medium_s342'; prefix = 'continuous_focus/'
    data = {prefix+'result.json': b'{}', prefix+run+'_summary.json': json.dumps(
        dict(id=run, checkpoint=run+'/update_000050.pt')).encode(),
        prefix+run+'/update_000050.pt': b'selected', prefix+run+'/update_000100.pt': b'last',
        prefix+'pilot_medium_s322/update_000100.pt': b'wrong campaign'}
    monkeypatch.setattr(collector, 'list_outputs', lambda api, k: ({n: NS(url=n) for n in data}, []))
    monkeypatch.setattr(requests, 'get', lambda url, **kw: NS(content=data[url], raise_for_status=lambda: None))
    out = tmp_path/'out'
    result = collector.collect(NS(kernels_status=lambda k: 'COMPLETE'), 'owner/job', out,
                               profile='mana-decision-pilot')
    assert result['complete']
    assert (out/run/'update_000050.pt').read_bytes() == b'selected'
    assert (out/run/'update_000100.pt').read_bytes() == b'last'
    assert not (out/'pilot_medium_s322').exists()


def test_shared_control_does_not_supply_two_full_candidates_or_choose_luckiest_copy():
    from scripts.attack.select_pilot_pool import unique_mana_profiles
    rows = [dict(study=owner, trial=dict(id='control', coef=.1), score=score, minimum_family_points=.2)
            for owner, score in [('c', .9), ('a', .4), ('b', .7)]]
    rows.append(dict(study='b', trial=dict(id='high', coef=2.), score=.5, minimum_family_points=.2))
    unique = unique_mana_profiles(rows)
    assert len(unique) == 2
    control = next(r for r in unique if r['trial']['id'] == 'control')
    assert control['study'] == 'a' and control['score'] == .4
    assert len(control['replicated_controls']) == 2


def test_campaign_does_not_select_before_all_three_verified(monkeypatch):
    from scripts.kaggle import watch_mana_campaign as watcher
    monkeypatch.setattr(watcher, 'ready_studies', lambda c: [{}, {}])
    monkeypatch.setattr(watcher, 'select', lambda *a, **kw: pytest.fail('Partial pool must not select'))
    assert watcher.rank_complete(dict(studies=[{}, {}, {}])) is None
    assert watcher.continue_full({}, dict(gate_met=False))['full_submitted'] is False


def test_capacity_queue_only_reacts_to_matching_terminal_job(tmp_path):
    from scripts.kaggle.queue_capacity_pilot import capacity_available
    root = tmp_path/'running'; root.mkdir()
    (root/'preparation.json').write_text(json.dumps(dict(kernel='owner/our-job')))
    p = root/'collected.status.json'
    for kernel, status, expected in [('owner/other', 'COMPLETE', False),
                                    ('owner/our-job', 'RUNNING', False),
                                    ('owner/our-job', 'COMPLETE', True)]:
        p.write_text(json.dumps(dict(kernel=kernel, remote_status=status)))
        assert capacity_available([root]) is expected


def test_capacity_queue_refuses_duplicate_or_quota_rejected_submission(tmp_path):
    from scripts.kaggle.queue_capacity_pilot import retry
    for receipt in [dict(ref='/code/owner/our-job', versionNumber=1),
                    dict(error='Maximum weekly GPU quota of 30.00 hours reached.')]:
        (tmp_path/'launch_receipt.json').write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match='capacity rejection'):
            retry(tmp_path, 1)


def test_account_relocation_preserves_frozen_code_inputs_and_original(tmp_path):
    from scripts.kaggle.rehome_quota_pilot import relocate
    from scripts.kaggle.prepare_motionfix_pilot import sha
    source=tmp_path/'source'; (source/'dataset').mkdir(parents=True); (source/'job').mkdir()
    (source/'dataset/protocol.json').write_text('{}')
    config=dict(owner='andriashadrin',slug='data',pins={'protocol.json':sha(source/'dataset/protocol.json')})
    for name in ('dataset/dataset-metadata.json','job/kernel-metadata.json'):
        (source/name).write_text(json.dumps(dict(id='andriashadrin/job',is_private=True)))
    (source/'job/train.py').write_text('def execute(config):\n    pass\nexecute('+repr(config)+')\n')
    original=(source/'job/train.py').read_bytes()
    (source/'preparation.json').write_text(json.dumps(dict(config=config,kernel='andriashadrin/job',bootstrap_sha256=sha(source/'job/train.py'))))
    (source/'preflight.json').write_text(json.dumps(dict(frozen_sources_verified=True,exact_resume_16_to_32=True)))
    (source/'launch_receipt.json').write_text(json.dumps(dict(error='Maximum weekly GPU quota of 30.00 hours reached.')))
    target=tmp_path/'target'; result=relocate(source,target,'vertel0')
    assert result['kernel']=='vertel0/job'
    assert (source/'job/train.py').read_bytes()==original
    assert (target/'dataset/protocol.json').read_bytes()==(source/'dataset/protocol.json').read_bytes()
    assert json.loads((target/'preflight.json').read_text())['account_relocation']['identical_training_ast']
    assert not (target/'launch_receipt.json').exists()
