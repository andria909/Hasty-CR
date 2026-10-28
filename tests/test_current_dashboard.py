import json
from scripts.defense.current_dashboard import progress, render


def test_failed_upload_is_not_claimed_as_submitted_training():
    assert progress(dict(status='upload_failed_before_submission',submitted=False)) == (
        '0%', 'проверено локально; загрузка Kaggle не принята')


def test_newer_matching_log_is_a_lower_bound(tmp_path):
    (tmp_path/'live_stream').mkdir()
    path=tmp_path/'live_stream/progress.json'
    path.write_text(json.dumps(dict(kernel='owner/job',checked_host_ns=20,
        trials={'one':dict(steps=102400)})))
    state=dict(root=str(tmp_path),kernel='owner/job',pilot_target_steps=819200,
        progress_snapshot=dict(checked_host_ns=10,trials={}))
    assert progress(state)[0]=='≥12.5%'
    path.write_text(json.dumps(dict(kernel='owner/other',checked_host_ns=30,
        trials={'one':dict(steps=819200)})))
    assert progress(state)[0]=='—'


def test_no_claim_of_verified_completion_from_progress_alone(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    state=dict(kernel='owner/job',pilot_target_steps=819200,
        progress_snapshot=dict(trials={'one':dict(steps=819200)}))
    (tmp_path/'fireball_reward_ablation_training_20261027.json').write_text(json.dumps(state))
    content=render(tmp_path)
    assert '≥100.0%' in content and 'шаги видны в логах' in content
    assert 'порог продолжения пройден' not in content
