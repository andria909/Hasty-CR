import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts.kaggle import run_defense_recovery as recovery


def test_ppo_additional_updates_and_commands(tmp_path):
    selected={'ppo_latest':tmp_path/'ppo.pt','grpo_boundary':tmp_path/'grpo.pt'}
    plan=recovery.commands(selected,{'update':500,'step':512000},tmp_path/'out',tmp_path/'project')
    ppo=plan['ppo'];grpo=plan['grpo']
    assert ppo[ppo.index('--updates')+1]=='1500'
    assert ppo[ppo.index('--init')+1]==ppo[ppo.index('--resume')+1]
    for flag,value in (('--envs','8'),('--rollout','128'),('--epochs','3'),
                       ('--lr','0.00005'),('--max-ms','240000'),('--device','cuda:0')):
        assert ppo[ppo.index(flag)+1]==value
    assert '--init' not in grpo
    assert grpo[grpo.index('--target-environment-steps')+1]=='1000000'
    assert grpo[grpo.index('--resume-code-migration')+1]==str(tmp_path/'project/data/defense/grpo_code_migration.json')
    assert 'ppo' not in recovery.commands(selected,{'update':2000,'step':2048000},tmp_path/'out',tmp_path/'project')


@pytest.mark.parametrize('blob',[{'update':1,'step':1},{'update':2001,'step':2049024},
                                 {'update':-1,'step':-1024}])
def test_bad_progress_rejected(blob):
    with pytest.raises(ValueError):recovery.ppo_remaining(blob)


def make_receipt(tmp_path):
    names={'ppo_latest':'ppo/update_000500.pt','grpo_boundary':'grpo/boundary_u000375_s000010.pt'}
    receipt=dict(complete=True,remote_status='ERROR',selected_roles=names,files={})
    for name in names.values():
        path=tmp_path/name;path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'opaque fake no torch')
        receipt['files'][name]={'sha256':recovery.sha(path),'remote_hash_verified':False}
    target=tmp_path/'recovery_receipt.json'
    target.write_text(json.dumps(receipt))
    return target,receipt


def test_receipt_and_each_weight_hash_guard(tmp_path):
    target,receipt=make_receipt(tmp_path)
    _,selected=recovery.verify_recovered(tmp_path,recovery.sha(target))
    assert len(selected)==2
    with pytest.raises(ValueError,match='receipt SHA'):
        recovery.verify_recovered(tmp_path,'0'*64)
    (tmp_path/receipt['selected_roles']['ppo_latest']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='weight SHA'):
        recovery.verify_recovered(tmp_path,recovery.sha(target))


def test_real_source_v2_all_file_hashes_verified_without_training(tmp_path):
    source=Path('tmp/kaggle/defense_recovery_20261022/source_v2')
    if not source.exists():pytest.skip('Local recovery source artifact unavailable')
    proof=recovery.prepare_source(source,tmp_path/'project')
    manifest=json.loads((source/'manifest.json').read_text())
    assert proof['source_file_count']==len(manifest['files'])
    assert proof['baseline_sha256']==recovery.BASELINE_SHA
    assert not proof['baseline_file_verified']
    assert proof['migration_sha256']==manifest['recovery_migration_sha256']


def test_failure_result_preserves_artifact_status_without_torch(monkeypatch,tmp_path):
    def fail(*args,**kwargs):raise ValueError('synthetic source hash error')
    monkeypatch.setattr(recovery,'prepare_source',fail)
    output=tmp_path/'new'
    with pytest.raises(ValueError,match='synthetic'):
        recovery.execute(NS(output=output,source_input=tmp_path,recovered_input=tmp_path,
                            recovery_receipt_sha256='0'*64))
    report=json.loads((output/'recovery_result.json').read_text())
    assert report['status']=='failed' and report['not_exact_resume']
    assert not report['quality_accepted']
    assert report['original_stage2_status_not_overridden']


def test_fresh_selection_same_fixed_validation_and_no_testset(monkeypatch,tmp_path):
    paths=[tmp_path/'final.pt',tmp_path/'best.pt']
    for path in paths:path.write_bytes(b'opaque')
    commands=[]
    def fake_run(command,project,log):
        commands.append(command)
        Path(command[4]).write_text(json.dumps({'mean_reward':2 if 'best' in command[3] else 1}))
    monkeypatch.setattr(recovery,'run_logged',fake_run)
    report=recovery.choose_fresh_validation(paths,'ppo',tmp_path,tmp_path)
    assert report['selected']['checkpoint']==str(paths[1])
    assert report['episodes']==32 and report['seed']==900000
    assert report['split']=='validation' and not report['quality_accepted']
    assert len(commands)==2
    assert all('episodes=32,seed=900000' in c[2] for c in commands)


def test_resume_configuration_and_all_optimizer_states_required():
    shared=dict(seed=42,max_ms=240000,initial_elixir=10,action_delay_ms=150,
                difficulty=None,scenario_ids=['original-train-case'])
    ppo=dict(update=500,step=512000,config=dict(shared,envs=8,rollout=128,epochs=3,lr=5e-5),
             optimiser={},sampler={},rng={},environment_states=[{}],observations=[],infos=[],episode_counters=[])
    grpo=dict(config=dict(shared,group_size=8,epochs=2,lr=3e-5),reference_state_dict={},
              optimizer={},sampler={},torch_rng=[],cuda_rng=[],action_rng=[],numpy_rng=[],python_rng=[],
              checkpoint_phase='boundary',resume_supported=True,environment_steps=290000)
    recovery.validate_resume_states(ppo,grpo)
    ppo['config']['lr']=1
    with pytest.raises(ValueError,match='lr'):
        recovery.validate_resume_states(ppo,grpo)
    ppo['config']['lr']=5e-5
    del grpo['reference_state_dict']
    with pytest.raises(ValueError,match='boundary'):
        recovery.validate_resume_states(ppo,grpo)


def test_kaggle_autoextracted_source_uses_explicit_root_and_checks_every_hash(tmp_path):
    import tarfile
    source=Path('tmp/kaggle/defense_recovery_20261022/source_v2')
    if not source.exists():pytest.skip('Local recovery source artifact unavailable')
    mount=tmp_path/'mount';mount.mkdir()
    (mount/'manifest.json').write_bytes((source/'manifest.json').read_bytes())
    extracted=mount/'source';extracted.mkdir()
    with tarfile.open(source/'source.tar.gz') as archive:
        archive.extractall(extracted,filter='data')
    proof=recovery.prepare_source(mount,tmp_path/'project',extracted_root=extracted)
    assert proof['source_file_count']==142
    (extracted/'sim/defense_env.py').write_text('corrupted source')
    with pytest.raises(ValueError,match='Source file SHA mismatch'):
        recovery.prepare_source(mount,tmp_path/'refused',extracted_root=extracted)


def test_validation_selected_artifact_is_selfcontained_and_never_replaced(tmp_path):
    source=tmp_path/'old_input.pt';source.write_bytes(b'validated weights')
    output=tmp_path/'new_output';output.mkdir()
    selection={'selected':{'checkpoint':str(source),'sha256':recovery.sha(source)}}
    artifact=recovery.materialize_selected_artifact(selection,'ppo',output)
    assert artifact['path']=='selected/ppo.pt'
    assert recovery.sha(output/artifact['path'])==selection['selected']['sha256']
    with pytest.raises(FileExistsError):
        recovery.materialize_selected_artifact(selection,'ppo',output)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed after validation'):
        recovery.materialize_selected_artifact(selection,'grpo',output)
