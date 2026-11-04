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
def test_bank_campaign_ignores_foreign_progress_and_unverified_completion(tmp_path):
    from scripts.defense.current_dashboard import render_bank_campaign
    job=tmp_path/'job';(job/'live_stream').mkdir(parents=True)
    campaign=dict(studies=[dict(label='v25',kernel='owner/new',root=str(job),
        status_file='state.json',verified_file='proof.json')])
    (tmp_path/'state.json').write_text(json.dumps(dict(status='принят Kaggle')))
    (job/'live_stream/progress.json').write_text(json.dumps(dict(kernel='owner/old',
        initializations={'retain01_s392':dict(updates=800)})))
    (tmp_path/'proof.json').write_text(json.dumps(dict(kernel='owner/new',supervised_updates_verified=6400,runs=[])))
    text=render_bank_campaign(tmp_path,campaign)
    assert '| v25 | — | принят Kaggle |'in text
    assert '100% проверено'not in text
    (job/'live_stream/progress.json').write_text(json.dumps(dict(kernel='owner/new',
        initializations={'retain01_s392':dict(updates=25)})))
    assert '| v25 | ≥0.4%'in render_bank_campaign(tmp_path,campaign)


def test_correction_dashboard_does_not_count_old_or_cpu_updates(tmp_path):
    from scripts.defense.current_dashboard import render_bank_campaign
    job=tmp_path/'job';(job/'live_stream').mkdir(parents=True)
    state=dict(root=str(job),kernel='owner/corrections',status='RUNNING')
    (tmp_path/'bank_correction_pair_training_20261101.json').write_text(json.dumps(state))
    snapshot=dict(kernel=state['kernel'],initializations={'retain01_s392':dict(updates=800)},
        corrections={'control_kl001_s402':dict(updates=400,actual_cuda=True),
            'corrections_kl001_s402':dict(updates=800,actual_cuda=False)})
    (job/'live_stream/progress.json').write_text(json.dumps(snapshot))
    text=render_bank_campaign(tmp_path,dict(studies=[]))
    assert '| Исправления защиты: с примерами / без | ≥6.2% | RUNNING |'in text
    assert '100% проверено'not in text
    snapshot['kernel']='owner/other'
    (job/'live_stream/progress.json').write_text(json.dumps(snapshot))
    assert '| Исправления защиты: с примерами / без | — | RUNNING |'in render_bank_campaign(tmp_path,dict(studies=[]))


def test_choice_preparation_does_not_claim_gpu_training(tmp_path):
    from scripts.defense.current_dashboard import render_credit_assignment
    state=dict(root=str(tmp_path/'job'),kernel='owner/credit',submitted=True)
    (tmp_path/'policy_choice_preparation_20261104.json').write_text(json.dumps(dict(
        frozen_pipeline_verified=True,submitted=False)))
    content=render_credit_assignment(tmp_path,state)
    assert 'Следующий опыт подготовлен' in content
    assert 'проверены на CPU' in content and 'Запуск после сравнения текущей серии' in content
    assert content.count('нет счётчика')==8


def test_launched_choice_study_has_separate_progress_and_no_invented_cuda(tmp_path):
    from scripts.defense.current_dashboard import policy_choice_progress
    root=tmp_path/'job';(root/'live_stream').mkdir(parents=True)
    state=dict(root=str(root),kernel='owner/choice',submitted=True,actual_cuda_ready=False)
    (tmp_path/'policy_choice_training_20261104.json').write_text(json.dumps(state))
    progress=root/'live_stream/progress.json'
    progress.write_text(json.dumps(dict(kernel='owner/credit',trials={'pilot_control_s482':dict(steps=409600,target_steps=409600)})))
    content='\n'.join(policy_choice_progress(tmp_path))
    assert 'CUDA ещё не подтверждена' in content and content.count('нет счётчика')==8
    progress.write_text(json.dumps(dict(kernel='owner/choice',trials={'pilot_control_s482':dict(steps=102400,target_steps=409600)})))
    content='\n'.join(policy_choice_progress(tmp_path))
    assert '≥25.0%'in content and '100% проверено'not in content
