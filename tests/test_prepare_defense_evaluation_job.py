import json
from pathlib import Path

import pytest

from scripts.kaggle import prepare_defense_evaluation_job as builder
from tests.test_evaluate_defense_stage2 import recovery_fixture


def prepared(monkeypatch,tmp_path):
    source,manifest,report,result=recovery_fixture(tmp_path)
    evaluation=tmp_path/'evaluation';evaluation.mkdir()
    (evaluation/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(builder,'frozen_members',lambda path:({},manifest,{}))
    out=tmp_path/'prepared'
    config=builder.prepare(source,evaluation,out,kernel='owner/fixed-evaluation',
                           selected_dataset='owner/fixed-selection',evaluation_dataset='owner/fixed-source')
    return source,evaluation,out,config


def test_prepared_job_is_selfcontained_private_and_uses_exact_slug(monkeypatch,tmp_path):
    source,evaluation,out,config=prepared(monkeypatch,tmp_path)
    metadata=json.loads((out/'job/kernel-metadata.json').read_text())
    assert metadata['title']=='fixed evaluation'
    assert metadata['is_private'] and not metadata['enable_internet']
    assert metadata['dataset_sources']==['owner/fixed-source','owner/fixed-selection']
    ns={'__name__':'prepared_test'}
    exec(compile((out/'job/evaluate.py').read_text(),'<prepared>','exec'),ns)
    assert ns['EVALUATION_LAUNCH_CONFIG']['recovery_result_sha256']==config['recovery_result_sha256']
    root=tmp_path/'input';root.mkdir()
    (root/'fixed-source').symlink_to(evaluation,target_is_directory=True)
    (root/'fixed-selection').symlink_to(out/'selected_dataset',target_is_directory=True)
    ns['main']=lambda argv:argv
    argv=ns['launch_evaluation'](ns['EVALUATION_LAUNCH_CONFIG'],root,tmp_path/'working')
    assert '--recovery-result-sha256' in argv
    assert argv[argv.index('--extracted-root')+1]==str(root/'fixed-source/source')
    for name in ('ppo','grpo'):
        assert (tmp_path/f'working/selected_input/selected/{name}.pt').read_bytes()==(source/f'selected/{name}.pt').read_bytes()


def test_tampered_flat_selection_refused_before_controller(monkeypatch,tmp_path):
    source,evaluation,out,config=prepared(monkeypatch,tmp_path)
    (out/'selected_dataset/ppo.pt').write_bytes(b'changed')
    root=tmp_path/'input';root.mkdir()
    (root/'fixed-source').symlink_to(evaluation,target_is_directory=True)
    (root/'fixed-selection').symlink_to(out/'selected_dataset',target_is_directory=True)
    ns={'__name__':'prepared_test'}
    exec(compile((out/'job/evaluate.py').read_text(),'<prepared>','exec'),ns)
    ns['main']=lambda argv:pytest.fail('Controller must not run with unverified weights')
    with pytest.raises(ValueError,match='Selected input SHA'):
        ns['launch_evaluation'](ns['EVALUATION_LAUNCH_CONFIG'],root,tmp_path/'working')
    assert not (tmp_path/'working/selected_input').exists()


def test_incomplete_training_cannot_prepare_test_job(monkeypatch,tmp_path):
    source,manifest,report,result=recovery_fixture(tmp_path)
    report['training_counters']['ppo']['steps']=1177600
    result.write_text(json.dumps(report))
    evaluation=tmp_path/'evaluation';evaluation.mkdir()
    (evaluation/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(builder,'frozen_members',lambda path:({},manifest,{}))
    with pytest.raises(ValueError,match='step target'):
        builder.prepare(source,evaluation,tmp_path/'refused',kernel='owner/evaluation',selected_dataset='owner/selection')
    assert not (tmp_path/'refused').exists()


def test_existing_preparation_is_never_overwritten(monkeypatch,tmp_path):
    source,evaluation,out,config=prepared(monkeypatch,tmp_path)
    with pytest.raises(FileExistsError):
        builder.prepare(source,evaluation,out,kernel='owner/fixed-evaluation',selected_dataset='owner/fixed-selection')
