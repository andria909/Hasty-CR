import json

import pytest

from scripts.kaggle import watch_bank_correction_pair as module


def setup(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    root=tmp_path/'job';(root/'collected').mkdir(parents=True)
    (root/'collected/collection_receipt.json').write_text('{"complete":true}')
    status=tmp_path/'status.json';status.write_text('{"kernel":"owner/job"}')
    monkeypatch.setattr(module,'render',lambda *a:'dashboard')
    return root,status


def test_full_matches_start_only_after_verification_and_include_all_candidates(tmp_path,monkeypatch):
    root,status=setup(tmp_path,monkeypatch);calls=[]
    def verify(path):calls.append('verify');return dict(runs=[dict(checkpoint=f'candidate{i}.pt')for i in range(8)])
    def validate(weights,config,out):
        calls.append('validate');assert len(weights)==9 and weights[0]==str(root/'dataset/parent.pt')
        assert weights[1:]==[f'candidate{i}.pt'for i in range(8)]
    def summarize(*args):calls.append('summarize');return dict(games_verified=648)
    monkeypatch.setattr(module,'verify',verify);monkeypatch.setattr(module,'full_validation',validate)
    monkeypatch.setattr(module,'summarize',summarize)
    module.watch(root,status,tmp_path/'verified.json','summary.json','report.md','config.json')
    assert calls==['verify','validate','summarize']
    state=json.loads(status.read_text());assert state['status']=='full_validation_verified'
    assert state['full_training_accepted']is False and state['full_validation_games_verified']==648


def test_failed_artifact_verification_does_not_start_matches(tmp_path,monkeypatch):
    root,status=setup(tmp_path,monkeypatch)
    def reject(*args):raise ValueError('incomplete actual budget')
    monkeypatch.setattr(module,'verify',reject)
    monkeypatch.setattr(module,'full_validation',lambda *a:pytest.fail('Invalid candidate must not be evaluated'))
    with pytest.raises(ValueError,match='incomplete'):
        module.watch(root,status,tmp_path/'verified.json','summary.json','report.md','config.json')
    assert json.loads(status.read_text())['status']=='inspection_required'


def test_partially_written_observer_json_is_not_a_failed_training_job(tmp_path):
    path=tmp_path/'progress.json';path.write_text('{"complete":')
    assert module.read_if_complete(path)is None
    path.write_text('{"complete":true}')
    assert module.read_if_complete(path)==dict(complete=True)


def test_isolated_fitting_diagnostic_does_not_promote_or_start_full_games(tmp_path,monkeypatch):
    from scripts.attack import summarize_fit_depth
    root,status=setup(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(module,'verify',lambda *a:dict(schema='verified-bank-fit-depth-v1'))
    monkeypatch.setattr(summarize_fit_depth,'summarize',lambda *a:calls.append('labels'))
    monkeypatch.setattr(module,'full_validation',lambda *a:pytest.fail('Isolated diagnostic must not auto-promote'))
    module.watch(root,status,tmp_path/'verified.json','summary.json','report.md','config.json')
    assert calls==['labels']
    state=json.loads(status.read_text());assert state['status']=='fitting_diagnostic_verified'
    assert state['full_training_accepted']is False
