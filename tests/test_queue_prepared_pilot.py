import ast
import json
from pathlib import Path
import pytest
from scripts.kaggle.queue_prepared_pilot import rebind,first_verified_free,sha


def fixture(root):
    (root/'dataset').mkdir(parents=True);(root/'job').mkdir()
    for name in('parent.pt','protocol.json','source.tar.gz'):(root/'dataset'/name).write_bytes(b'fixed input')
    config=dict(owner='oldowner',slug='demo-inputs',source_slug='demo-inputs',pins={n:sha(root/'dataset'/n)for n in('parent.pt','protocol.json')})
    source='def execute(config):\n    return config\nexecute('+repr(config)+')\n'
    (root/'job/train.py').write_text(source)
    (root/'preparation.json').write_text(json.dumps(dict(kernel='oldowner/demo',config=config,bootstrap_sha256=sha(root/'job/train.py'))))
    (root/'preflight.json').write_text(json.dumps(dict(exact_resume_16_to_32=True)))
    (root/'dataset/dataset-metadata.json').write_text(json.dumps(dict(id='oldowner/demo-inputs')))
    (root/'job/kernel-metadata.json').write_text(json.dumps(dict(id='oldowner/demo',dataset_sources=['oldowner/demo-inputs'])))


def test_rebinding_changes_owner_and_mount_only_preserving_frozen_algorithm(tmp_path):
    source=tmp_path/'source';fixture(source);before=(source/'job/train.py').read_bytes()
    target=tmp_path/'target';result=rebind(source,target,'newowner')
    assert result['kernel']=='newowner/demo'
    assert result['algorithm_ast_unchanged']
    assert (source/'job/train.py').read_bytes()==before
    old=ast.parse(before).body[0];new=ast.parse((target/'job/train.py').read_text()).body[0]
    assert ast.dump(old)==ast.dump(new)
    for name in('parent.pt','protocol.json','source.tar.gz'):
        assert (source/'dataset'/name).read_bytes()==(target/'dataset'/name).read_bytes()
    assert json.loads((target/'dataset/dataset-metadata.json').read_text())['id']=='newowner/demo-inputs'


def test_changed_inputs_cannot_rebind_or_create_output(tmp_path):
    source=tmp_path/'source';fixture(source);(source/'dataset/parent.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='input changed'):rebind(source,tmp_path/'target','newowner')
    assert not(tmp_path/'target').exists()


def test_slot_requires_matching_verified_budget_and_terminal_remote_receipt(tmp_path):
    slot=dict(owner='newowner',kernel='newowner/job',root=str(tmp_path/'run'),verified_report=str(tmp_path/'proof.json'))
    Path(slot['verified_report']).write_text(json.dumps(dict(kernel=slot['kernel'],pilot_steps_verified=819200)))
    path=Path(slot['root'])/'collected/collection_receipt.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(kernel=slot['kernel'],remote_status='RUNNING',complete=True)))
    assert first_verified_free([slot])is None
    path.write_text(json.dumps(dict(kernel=slot['kernel'],remote_status='COMPLETE',complete=True)))
    assert first_verified_free([slot])==slot
    Path(slot['verified_report']).write_text(json.dumps(dict(kernel='other/job',pilot_steps_verified=819200)))
    assert first_verified_free([slot])is None


def test_optional_priority_keeps_successful_pair_account_free_for_full_training(tmp_path):
    slot=dict(owner='newowner',kernel='newowner/job',root=str(tmp_path/'run'),verified_report=str(tmp_path/'proof.json'))
    Path(slot['verified_report']).write_text(json.dumps(dict(kernel=slot['kernel'],pilot_steps_verified=819200,
        pilot_selection=dict(gate_met=True))))
    path=Path(slot['root'])/'collected/collection_receipt.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(kernel=slot['kernel'],remote_status='COMPLETE',complete=True)))
    assert first_verified_free([slot])==slot
    assert first_verified_free([slot],reserve_eligible_pairs=True)is None


def test_new_full_job_keeps_owner_busy_after_its_pilot_finished(tmp_path):
    from scripts.kaggle.queue_prepared_pilot import full_training_reservations
    reports=tmp_path/'reports';reports.mkdir()
    path=reports/'global_full_pair0_training_20261027.json'
    report=dict(kernel='owner/full',launch_receipt={'versionNumber':1},children=[{},{}],status='full_training_steps_observed')
    path.write_text(json.dumps(report))
    busy,count=full_training_reservations(reports)
    assert busy=={'owner'}and count==2
    slot=dict(owner='owner',kernel='owner/pilot',root=str(tmp_path/'run'),verified_report=str(tmp_path/'proof.json'))
    Path(slot['verified_report']).write_text(json.dumps(dict(kernel=slot['kernel'],pilot_steps_verified=819200)))
    receipt=Path(slot['root'])/'collected/collection_receipt.json';receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(dict(kernel=slot['kernel'],remote_status='COMPLETE',complete=True)))
    assert first_verified_free([slot],busy_owners=busy)is None
    report['status']='local_wait_expired_remote_job_unchanged';path.write_text(json.dumps(report))
    assert full_training_reservations(reports)[0]=={'owner'}
    report['status']='full_complete_budgets_verified';path.write_text(json.dumps(report))
    assert not full_training_reservations(reports)[0]
