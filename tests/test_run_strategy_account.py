import json
from pathlib import Path

import pytest

from scripts.kaggle.run_strategy_account import commands, progress_line


@pytest.mark.parametrize('key', ['step', 'steps', 'environment_steps'])
def test_progress_uses_actual_training_counters(key):
    row=json.loads(progress_line(json.dumps({'update':5,key:250}), 'grpo',1000))
    assert row['environment_steps']==250
    assert row['training_percent']==25
    assert 'environment_steps' not in json.loads(progress_line(
        '{"event":"validation","update":5}', 'grpo',1000))


@pytest.mark.parametrize('algorithm,target', [('ppo',2048000),('grpo',1000000)])
def test_one_joint_gpu_policy_per_job(algorithm,target):
    args=commands(dict(algorithm=algorithm,target_steps=target),Path('/source'),Path('/input'),Path('/output'))
    for flag,value in [('--task-mode','joint'),('--device','cuda:0'),('--deadline-scale','3'),
                       ('--initial-hand-mode','shuffle'),('--target-environment-steps',str(target))]:
        assert args[args.index(flag)+1]==value
    assert args.count('-m')==1
