import json
from pathlib import Path

import pytest

from scripts.defense.evaluate_stage2 import selected_account2_candidate, sha, commands
from scripts.kaggle import prepare_defense_evaluation_job as builder
from tests.test_evaluate_defense_stage2 import recovery_fixture, protocol


def fixture(tmp_path):
    source,manifest,old,_=recovery_fixture(tmp_path)
    report=dict(schema='defense-grpo-account2-v1',status='grpo_target_completed_validation_selected',
        not_exact_resume=True,target_steps=1000000,steps=1000100,
        source_provenance=old['source_provenance'],validation_selection=old['validation_selection']['grpo'],
        selected_artifact=old['selected_artifacts']['grpo'])
    path=source/'account2_result.json';path.write_text(json.dumps(report))
    return source,manifest,report,path


def test_separate_grpo_needs_no_ppo_or_fabricated_parent_success(tmp_path):
    source,manifest,report,path=fixture(tmp_path)
    selected,loaded=selected_account2_candidate(source,sha(path.read_bytes()),manifest)
    assert set(selected)=={'grpo'}
    assert loaded==report and 'original_stage2_status_not_overridden'not in loaded
    plan=commands(Path('/source'),Path('/output'),selected,protocol().to_dict())
    assert set(plan)=={'grpo'}
    assert all(cmd[cmd.index('--device')+1]=='cuda:0'for cmd in plan['grpo'])
    assert '--all-scenarios'in plan['grpo'][0]
    assert '--baseline-manifest'in plan['grpo'][1]


@pytest.mark.parametrize('fault',['short','bool','status','source','split','nan','nonbest','unsafe','sha'])
def test_incomplete_or_unproven_account2_cannot_enter_test(tmp_path,fault):
    source,manifest,report,path=fixture(tmp_path)
    if fault=='short':report['steps']=999999
    elif fault=='bool':report['steps']=True
    elif fault=='status':report['status']='training'
    elif fault=='source':report['source_provenance']['archive_sha256']='d'*64
    elif fault=='split':report['validation_selection']['split']='test'
    elif fault=='nan':report['validation_selection']['selected']['mean_reward']=float('nan')
    elif fault=='nonbest':report['validation_selection']['candidates'].append(dict(mean_reward=100))
    elif fault=='unsafe':report['selected_artifact']['path']='../grpo.pt'
    elif fault=='sha':report['selected_artifact']['sha256']='e'*64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):selected_account2_candidate(source,sha(path.read_bytes()),manifest)


def test_builder_embeds_single_algorithm_and_exact_account2_provenance(monkeypatch,tmp_path):
    source,manifest,report,path=fixture(tmp_path)
    evaluation=tmp_path/'eval';evaluation.mkdir()
    (evaluation/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(builder,'frozen_members',lambda path:({},manifest,{}))
    output=tmp_path/'prepared'
    config=builder.prepare(source,evaluation,output,kernel='owner/account2-eval',
        selected_dataset='owner/account2-selected',evaluation_dataset='owner/frozen-source',result_mode='account2')
    assert config['algorithms']==['grpo']
    assert (output/'selected_dataset/grpo.pt').is_file()
    assert not (output/'selected_dataset/ppo.pt').exists()
    mounts=tmp_path/'mounts';mounts.mkdir()
    (mounts/'frozen-source').symlink_to(evaluation,target_is_directory=True)
    (mounts/'account2-selected').symlink_to(output/'selected_dataset',target_is_directory=True)
    ns={'__name__':'builder_test'}
    exec(compile((output/'job/evaluate.py').read_text(),'<job>','exec'),ns)
    captured=[];ns['main']=lambda argv:captured.extend(argv)
    ns['launch_evaluation'](ns['EVALUATION_LAUNCH_CONFIG'],mounts,tmp_path/'working')
    assert '--account2-result-sha256'in captured
    assert '--recovery-result-sha256'not in captured
    assert (tmp_path/'working/selected_input/account2_result.json').is_file()
    assert not (tmp_path/'working/selected_input/selected/ppo.pt').exists()
