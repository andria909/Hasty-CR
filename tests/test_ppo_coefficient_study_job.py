from pathlib import Path

from scripts.kaggle.run_ppo_coefficient_study import training_command


def test_paired_trials_have_exactly_one_twentieth_of_full_training_steps():
    for mode in ('defense','joint'):
        args=training_command(Path('/project'),Path('/input'),Path('/output'),mode,'baseline','cuda:1',42)
        value=lambda flag:args[args.index(flag)+1]
        assert int(value('--updates'))*int(value('--envs'))*int(value('--rollout'))==102400
        assert value('--eval-every')=='0'
        assert value('--reward-profile').endswith('/baseline.json')
        assert value('--device')=='cuda:1'
        assert value('--init').endswith('teacher_'+mode+'.pt')
        if mode=='joint':
            assert value('--task-mode')=='joint'and value('--target-environment-steps')=='102400'
            assert value('--deadline-scale')=='3'
        else:assert value('--coordinate-mode')=='physical_live'
