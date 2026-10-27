import json
from scripts.kaggle.start_global_full_observers import commands,start
from scripts.defense.update_done import render_global_full_training


def test_full_observers_use_full_collector_and20minute_remote_interval(tmp_path):
    (tmp_path/'preparation.json').write_text(json.dumps(dict(kernel='owner/full')))
    result=commands(tmp_path,'global_full_pair0','owner','sdk','python')
    assert 'scripts.kaggle.collect_global_full'in result['collector']
    assert result['collector'][result['collector'].index('--poll-seconds')+1]=='1200'
    assert result['stream'][result['stream'].index('--reconnect-seconds')+1]=='1200'
    assert 'scripts.kaggle.watch_global_full_pair'in result['postprocess']


def test_full_progress_is_per_child_and_missing_counter_does_not_become_zero(tmp_path):
    report=tmp_path/'global_full_pair0_training_20261027.json'
    report.write_text(json.dumps(dict(status='full_training_steps_observed',children=[
        dict(id='global0_full_a_s64'),dict(id='global1_full_a_s64')],progress_snapshot=dict(trials={
        'global0_full_a_s64':dict(steps=1024000,target_steps=4096000)}))))
    rendered='\n'.join(render_global_full_training(tmp_path))
    assert '| global0_full_a_s64 | 25.0% |'in rendered
    assert '| global1_full_a_s64 | — |'in rendered
    proof=tmp_path/'global_full_pair0_verified_20261027.json'
    proof.write_text(json.dumps(dict(full_steps_verified=8192000)))
    assert '\n'.join(render_global_full_training(tmp_path)).count('100%')==2
