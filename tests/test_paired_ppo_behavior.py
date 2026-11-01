from copy import deepcopy

import pytest

from scripts.attack.compare_ppo_pilot_behavior import compare_records, FAMILIES, METRICS


def records():
    rows = []
    for family in FAMILIES:
        for task in ('attack', 'defense'):
            for i in range(6):
                rows.append(dict(scenario_id=f'{task}::{family}/heldout_push', episode_seed=len(rows),
                                 match_seed=1000000000+len(rows), family_id=family, task=task,
                                 split='validation', outcome='loss', elapsed_ms=60000,
                                 accepted_card_counts={}, **{k:0. for k in METRICS}))
    return rows


def test_pairing_uses_context_identity_not_file_order():
    before = records()
    before[0]['outcome'] = before[1]['outcome'] = 'win'
    after = deepcopy(before)
    after[0]['outcome'] = 'draw'
    after[2]['outcome'] = after[6]['outcome'] = 'win'
    after[2]['empty_spell_elixir'] = 4.
    result = compare_records(before, list(reversed(after)))
    assert result['all']['gained_wins'] == 2
    assert result['all']['lost_wins'] == 1
    assert result['all']['wins_delta'] == 1
    assert result['tasks']['attack']['wins_delta'] == 0
    assert result['tasks']['defense']['wins_delta'] == 1
    assert result['all']['metrics']['empty_spell_elixir']['delta'] == pytest.approx(4/72)


@pytest.mark.parametrize('mutation', ['duplicate', 'seed', 'split', 'nonfinite', 'task', 'outcome'])
def test_unpaired_or_invalid_evidence_is_rejected(mutation):
    a = records(); b = deepcopy(a)
    if mutation == 'duplicate': b[1] = deepcopy(b[0])
    if mutation == 'seed': b[0]['match_seed'] += 9000
    if mutation == 'split': b[0]['split'] = 'test'
    if mutation == 'nonfinite': b[0]['empty_spell_elixir'] = float('nan')
    if mutation == 'task': b[0]['task'] = 'defense'
    if mutation == 'outcome': b[0]['outcome'] = 'unknown'
    with pytest.raises(ValueError): compare_records(a, b)


def test_study_pairs_each_candidate_with_same_seed_and_rejects_changed_artifact(tmp_path):
    import json
    from pathlib import Path
    from scripts.kaggle.local_artifact_transfer import file_sha
    from scripts.attack.compare_ppo_pilot_behavior import compare_study
    from tests.test_calibrated_spell_pilot import protocol
    p = protocol(); kernel = 'owner/paired-fixture'
    (tmp_path/'dataset').mkdir(); collected = tmp_path/'collected'; collected.mkdir()
    (tmp_path/'preparation.json').write_text(json.dumps(dict(kernel=kernel)))
    (tmp_path/'dataset/protocol.json').write_text(json.dumps(p))
    models = []; files = {}
    for case in p['cases']:
        for seed in p['pilot_seeds']:
            name = f"pilot_{case['id']}_s{seed}"
            (collected/name).mkdir()
            weights = collected/name/'policy.pt'; weights.write_bytes(b'unit-test-placeholder')
            rows = records()
            if seed == 422: rows[0]['outcome'] = 'win'
            if case['id'] != 'control': rows[1]['outcome'] = 'win'
            path = collected/(name+'_summary.json')
            path.write_text(json.dumps(dict(id=name, checkpoint=name+'/policy.pt', selected_update=100,
                                            validation=dict(records=rows))))
            files[path.name] = dict(sha256=file_sha(path))
            models.append(dict(id=name, checkpoint=str(weights), checkpoint_sha256=file_sha(weights)))
    (collected/'collection_receipt.json').write_text(json.dumps(dict(complete=True,kernel=kernel,files=files)))
    proof_path = tmp_path/'proof.json'
    proof_path.write_text(json.dumps(dict(kernel=kernel,calibrated_spell_cost_verified=True,
                                         pilot_steps_verified=819200,pilot_models=models)))
    report = compare_study(tmp_path, proof_path)
    assert len(report['rows']) == 8
    for row in report['rows']:
        assert row['reference'] == f"pilot_control_s{row['seed']}"
        assert row['comparison']['all']['wins_delta'] == (0 if row['case'] == 'control' else 1)
    changed = collected/'pilot_medium_s422_summary.json'
    changed.write_text(changed.read_text()+' ')
    with pytest.raises(ValueError, match='measurements changed'): compare_study(tmp_path, proof_path)
