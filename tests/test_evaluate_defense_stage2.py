import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts.defense.evaluate_stage2 import (aggregate_candidate,commands,frozen_members,
                                            selected_candidates,sha)
from sim.defense_protocol import EvaluationProtocol,acceptance_gate


def protocol():
    return EvaluationProtocol(test_scenario_ids=('test/s',),paired_seeds=tuple(range(710001,710011)),
                              defense_seeds=(810001,810002,810003))


def frozen_fixture(tmp_path,archive=True):
    root=tmp_path/'inputs';root.mkdir()
    p=protocol();data=(json.dumps(p.to_dict(),sort_keys=True)+'\n').encode()
    members={'data/defense/evaluation_protocol.json':data,'sim/no_execute.py':b'invalid python never imported'}
    manifest=dict(files={name:sha(contents)for name,contents in members.items()},
        evaluation_protocol_fingerprint=p.fingerprint,evaluation_protocol_sha256=sha(data),
        archive_sha256='unused',checkpoint_sha256='v25',warmstarts={'ppo':{'sha256':'warm'}})
    if archive:
        with tarfile.open(root/'source.tar.gz','w:gz')as stream:
            for name,contents in members.items():
                entry=tarfile.TarInfo(name);entry.size=len(contents);stream.addfile(entry,io.BytesIO(contents))
        manifest['archive_sha256']=sha((root/'source.tar.gz').read_bytes())
    else:
        for name,contents in members.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(contents)
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root,manifest,members,p


@pytest.mark.parametrize('archive',[True,False])
def test_archive_and_explicit_extracted_mounts_verify_identical_bytes(tmp_path,archive):
    root,manifest,members,p=frozen_fixture(tmp_path,archive)
    actual,_,data=frozen_members(root,expected_protocol=p.fingerprint)
    assert actual==members and data==p.to_dict()
    if archive:
        path=root/'source.tar.gz';path.write_bytes(path.read_bytes()+b'corrupt')
    else:
        (root/'sim/no_execute.py').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch|SHA mismatch'):
        frozen_members(root,expected_protocol=p.fingerprint)


def candidate_fixture(tmp_path,manifest):
    source=tmp_path/'stage2';source.mkdir()
    result=dict(status='passed',input_checkpoint_sha256=manifest['checkpoint_sha256'],warmstarts=manifest['warmstarts'],targets={'ppo_steps':2048000,'grpo_steps':1000000})
    for name in ('ppo','grpo'):
        folder=source/name;folder.mkdir();checkpoint=folder/'best_validation_u000100.pt';checkpoint.write_bytes(name.encode())
        # Other checkpoints never affect selection.
        (folder/'final_later.pt').write_bytes(b'not-selected')
        result[name]=dict(returncode=0,step=result['targets'][name+'_steps'],selected_checkpoint=checkpoint.relative_to(source).as_posix(),selected_sha256=sha(checkpoint.read_bytes()))
        result[name+'_validation']=dict(returncode=0)
    path=source/'stage2_result.json';path.write_text(json.dumps(result))
    return source,result,path


def test_selection_is_only_pinned_stage2_validation_choice_not_latest_or_test(tmp_path):
    _,manifest,_,_=frozen_fixture(tmp_path)
    source,result,path=candidate_fixture(tmp_path,manifest)
    selected,_=selected_candidates(source,sha(path.read_bytes()),manifest)
    assert all(row['path'].name.startswith('best_validation')for row in selected.values())
    with pytest.raises(ValueError,match='Pinned stage2 result SHA'):
        selected_candidates(source,'bad',manifest)
    result['ppo']['selected_checkpoint']='ppo/final_later.pt'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='validation-selected'):
        selected_candidates(source,sha(path.read_bytes()),manifest)


def test_selected_checkpoint_corruption_and_wrong_provenance_fail_closed(tmp_path):
    _,manifest,_,_=frozen_fixture(tmp_path)
    source,result,path=candidate_fixture(tmp_path,manifest)
    (source/result['grpo']['selected_checkpoint']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Selected validation checkpoint SHA'):
        selected_candidates(source,sha(path.read_bytes()),manifest)
    result['warmstarts']={};path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='warmstart provenance'):
        selected_candidates(source,sha(path.read_bytes()),manifest)


def test_commands_pin_test_seeds_two_gpus_and_all_manifest_baselines(tmp_path):
    p=protocol()
    selected={name:{'path':Path('/inputs')/name/'best.pt'}for name in ('ppo','grpo')}
    plans=commands(tmp_path,tmp_path/'out',selected,p.to_dict())
    for gpu,name in enumerate(('ppo','grpo')):
        defense,duel=plans[name]
        assert defense[defense.index('--device')+1]==f'cuda:{gpu}'
        assert '--all-scenarios'in defense and defense[defense.index('--split')+1]=='test'
        assert defense[defense.index('--seeds')+1]=='810001,810002,810003'
        assert duel[duel.index('--matches')+1]=='20'
        assert '--baseline-manifest'in duel
        assert not any('train_defense'in part for cmd in plans[name]for part in cmd)


def test_complete_observed_evaluation_with_quality_flags_remains_undetermined():
    p=protocol();candidate='selected-sha'
    records=[dict(scenario_id='test/s',seed=seed,split='test',difficulty='simple',lane='left',
        terminal_reason='cleared',own_tower_fraction_loss={'left':0,'right':0,'king':0},
        own_towers_alive={'left':True,'right':True,'king':True},elixir_spent=1)for seed in p.defense_seeds]
    rows=[dict(baseline_id='v25',seed=seed,candidate_side=side,result='unknown',observed_result='win',
               coverage_flags=['simulator_mechanics_not_fully_validated'])for seed in p.paired_seeds for side in (-1,1)]
    defense=dict(checkpoint_sha256=candidate,records=records,coverage=dict(metrics_complete=True,
        tower_metrics_complete=True,scenario_catalog_known=True,all_expected_scenarios_seen=True))
    duels=dict(candidate_sha256=candidate,head_to_head=rows,coverage_complete_for_acceptance=False,
        baselines=[dict(label='v25',summary={'wins':20})])
    result=aggregate_candidate(p,defense,duels,candidate,acceptance_gate)
    assert result['defense_episodes']==3 and result['duel_games']==20
    assert result['acceptance']['status']=='undetermined'
    assert 'missing_coverage'in result['acceptance']['undetermined_reasons']
    assert result['observed_duel_summaries']['v25']['wins']==20
    duels['head_to_head']=rows[:-1]
    with pytest.raises(ValueError,match='off-protocol duel'):
        aggregate_candidate(p,defense,duels,candidate,acceptance_gate)


def test_budget_stop_without_requested_steps_cannot_touch_final_test(tmp_path):
    _,manifest,_,_=frozen_fixture(tmp_path)
    source,result,path=candidate_fixture(tmp_path,manifest)
    result['grpo']['step']=999999
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='step target has not been reached'):
        selected_candidates(source,sha(path.read_bytes()),manifest)


def recovery_fixture(tmp_path):
    source=tmp_path/'recovery-output';(source/'selected').mkdir(parents=True)
    manifest=dict(recovery_source_archive_sha256='a'*64,
        files={'sim/defense_env.py':'b'*64},evaluation_environment_patch={'to_sha256':'b'*64},
        evaluation_repaired_files=['sim/defense_env.py'])
    report=dict(schema='defense-semantic-recovery-v1',status='training_targets_completed_validation_selected',
        not_exact_resume=True,original_stage2_status_not_overridden=True,
        source_provenance={'archive_sha256':'a'*64},targets={'ppo_steps':2048000,'grpo_steps':1000000},
        training_counters={},validation_selection={},selected_artifacts={})
    for name,target in [('ppo',2048000),('grpo',1000000)]:
        artifact=source/'selected'/f'{name}.pt';artifact.write_bytes(('self-contained-'+name).encode())
        digest=sha(artifact.read_bytes())
        choice=dict(checkpoint='/old/runtime/final.pt',sha256=digest,mean_reward=1.5,
                    validation_report='/old/runtime/selection.json',validation_sha256='c'*64)
        report['training_counters'][name]=dict(steps=target,target_met=True,not_exact_resume=True)
        report['validation_selection'][name]=dict(split='validation',episodes=32,seed=900000,
            selected=choice,candidates=[choice],quality_accepted=False)
        report['selected_artifacts'][name]=dict(path=f'selected/{name}.pt',sha256=digest)
    result=source/'recovery_result.json';result.write_text(json.dumps(report))
    return source,manifest,report,result


def test_semantic_recovery_selects_exact_self_contained_fresh_validation_artifacts(tmp_path):
    from scripts.defense.evaluate_stage2 import selected_recovery_candidates
    source,manifest,report,result=recovery_fixture(tmp_path)
    selected,loaded=selected_recovery_candidates(source,sha(result.read_bytes()),manifest)
    assert loaded['not_exact_resume']is True
    assert loaded['original_stage2_status_not_overridden']is True
    assert {name:row['path'].relative_to(source).as_posix()for name,row in selected.items()}=={
        'ppo':'selected/ppo.pt','grpo':'selected/grpo.pt'}
    assert all(row['source']['checkpoint']=='/old/runtime/final.pt'for row in selected.values())
    assert all(row['selection_mode']=='semantic_recovery_fresh_validation'for row in selected.values())
    # The original failed parent status is neither read as passed nor overwritten.
    (source/'stage2_result.json').write_text(json.dumps({'status':'failed'}))
    selected_recovery_candidates(source,sha(result.read_bytes()),manifest)
    assert json.loads((source/'stage2_result.json').read_text())['status']=='failed'


@pytest.mark.parametrize('mutation',[
    'schema','status','nonexact','parent-status','provenance','target-counts','target-steps','target-met',
    'validation-split','validation-episodes','validation-seed','validation-nan','nonbest','artifact-sha','artifact-path'])
def test_semantic_recovery_malformed_targets_provenance_validation_artifacts_fail_closed(tmp_path,mutation):
    from scripts.defense.evaluate_stage2 import selected_recovery_candidates
    source,manifest,report,result=recovery_fixture(tmp_path)
    if mutation=='schema':report['schema']='wrong'
    elif mutation=='status':report['status']='failed'
    elif mutation=='nonexact':report['not_exact_resume']=False
    elif mutation=='parent-status':report['original_stage2_status_not_overridden']=False
    elif mutation=='provenance':report['source_provenance']['archive_sha256']='x'*64
    elif mutation=='target-counts':report['targets']['grpo_steps']=10
    elif mutation=='target-steps':report['training_counters']['grpo']['steps']=999999
    elif mutation=='target-met':report['training_counters']['grpo']['target_met']=False
    elif mutation=='validation-split':report['validation_selection']['ppo']['split']='test'
    elif mutation=='validation-episodes':report['validation_selection']['ppo']['episodes']=3
    elif mutation=='validation-seed':report['validation_selection']['ppo']['seed']=1
    elif mutation=='validation-nan':report['validation_selection']['ppo']['selected']['mean_reward']=float('nan')
    elif mutation=='nonbest':report['validation_selection']['ppo']['candidates'].append(dict(mean_reward=99.,sha256='f'*64))
    elif mutation=='artifact-sha':report['selected_artifacts']['ppo']['sha256']='f'*64
    elif mutation=='artifact-path':report['selected_artifacts']['ppo']['path']='selected/../ppo.pt'
    result.write_text(json.dumps(report))
    with pytest.raises(ValueError):selected_recovery_candidates(source,sha(result.read_bytes()),manifest)


def test_semantic_recovery_rejects_pin_or_content_changes_and_unrepaired_eval(tmp_path):
    from scripts.defense.evaluate_stage2 import selected_recovery_candidates
    source,manifest,report,result=recovery_fixture(tmp_path)
    with pytest.raises(ValueError,match='Pinned recovery'):
        selected_recovery_candidates(source,'bad',manifest)
    with pytest.raises(ValueError,match='repaired evaluation'):
        selected_recovery_candidates(source,sha(result.read_bytes()),dict(manifest,evaluation_repaired_files=[]))
    (source/'selected/grpo.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='content SHA'):
        selected_recovery_candidates(source,sha(result.read_bytes()),manifest)
