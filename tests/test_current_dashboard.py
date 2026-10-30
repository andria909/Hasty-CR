import json
from scripts.defense.current_dashboard import progress, render


def test_invalidated_teacher_is_never_shown_as_an_accepted_candidate():
    pct,note=progress(dict(status='invalidated_teacher',promotion_blocked=True))
    assert pct=='исключён' and 'остановка GPU пока не подтверждена' in note


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


def test_remote_complete_does_not_invent_missing_training_steps(tmp_path):
    (tmp_path/'collected.status.json').write_text(json.dumps(dict(kernel='owner/job',remote_status='COMPLETE')))
    state=dict(root=str(tmp_path),kernel='owner/job',pilot_target_steps=819200,
               progress_snapshot=dict(trials={'one':dict(steps=102400)}))
    assert progress(state)==('≥12.5%','Kaggle завершён; проверяем файлы')
    state['progress_snapshot']={}
    assert progress(state)==('—','Kaggle завершён; проверяем файлы')


def test_current_crown_dashboard_keeps_lower_bound_and_hides_unverified_scores(tmp_path):
    state=dict(kernel='owner/crown',root=str(tmp_path/'job'),progress_snapshot=dict(trials={
        'pilot_control_s382':dict(steps=101376)}))
    (tmp_path/'crown_economy_pilot_training_20261030.json').write_text(json.dumps(state))
    (tmp_path/'crown_economy_pilot_summary_20261030.json').write_text(json.dumps(dict(
        kernel='owner/crown',rows=[dict(id='control',wins=144,games=144)])))
    (tmp_path/'job/live_stream').mkdir(parents=True)
    (tmp_path/'job/live_stream/progress.json').write_text(json.dumps(dict(kernel='owner/crown',trials={})))
    content=render(tmp_path)
    assert '≥99.0%' in content and '≥12.4%' in content
    assert '144/144' not in content and '100% бюджета проверено' not in content
    assert len(content.splitlines())<30
