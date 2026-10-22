import json
import sys

import pytest

from scripts.kaggle.run_defense_recovery import live_summary,run_logged


def test_live_summary_keeps_progress_before_large_grpo_sampler_and_validation():
    record=dict(update=351,environment_steps=270000,loss=1.5,
                sampler={'scenario_ids':['large']*1000},
                validation={'mean_reward':-2.3,'episodes':[{'large':True}]*1000})
    summary=json.loads(live_summary(json.dumps(record)))
    assert summary==dict(update=351,environment_steps=270000,loss=1.5,
                         validation={'mean_reward':-2.3})


def test_real_subprocess_streams_individual_child_progress_and_preserves_full_log(tmp_path,capsys):
    line=json.dumps(dict(update=1151,step=1178624,episodes=['full']*1000))
    log=tmp_path/'ppo.log'
    run_logged([sys.executable,'-c','print('+repr(line)+')'],tmp_path,log)
    output=capsys.readouterr().out
    assert 'child_started' in output and 'child_completed' in output
    progress=json.loads(next(line.removeprefix('ppo: ')for line in output.splitlines()if line.startswith('ppo: ')))
    assert progress['update']==1151 and progress['step']==1178624
    assert progress['training_percent']==57.55 and progress['target_steps']==2048000
    assert log.read_text()==line+'\n'
    with pytest.raises(FileExistsError):
        run_logged([sys.executable,'-c','print("must not overwrite")'],tmp_path,log)


def test_failing_child_is_visible_and_not_successfully_completed(tmp_path,capsys):
    with pytest.raises(RuntimeError,match='Child failed'):
        run_logged([sys.executable,'-c','print("specific error"); raise SystemExit(3)'],tmp_path,tmp_path/'grpo.log')
    output=capsys.readouterr().out
    assert 'grpo: specific error' in output
    assert '"returncode": 3' in output


@pytest.mark.parametrize('child,record,expected',[
    ('ppo',{'event':'START','step':1177600},57.5),
    ('grpo',{'update':350,'environment_steps':269282},26.93),
    ('grpo',{'update':5000,'environment_steps':1000100},100.0)])
def test_percentage_uses_actual_total_steps_including_restored_progress(child,record,expected):
    progress=json.loads(live_summary(json.dumps(record),child=child))
    assert progress['training_percent']==expected


def test_no_fake_percentage_from_grpo_updates_or_validation_phase():
    assert 'training_percent'not in json.loads(live_summary('{"update": 350}',child='grpo'))
    assert 'training_percent'not in json.loads(live_summary('{"step": 2048000}',child='ppo_selection_0'))
