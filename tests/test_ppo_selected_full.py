from pathlib import Path
import pytest
from scripts.kaggle.run_ppo_selected_full import full_command,TARGET,UPDATES


def test_full_training_budget_and_resume_have_mode_specific_update_semantics():
    for mode in ('defense','joint'):
        for completed in (0,300,1950):
            command=full_command(Path('/project'),Path('/inputs'),Path('/new-output'),mode,'baseline','cuda:0',42,
                                 resume=Path('/saved.pt') if completed else None,completed_updates=completed)
            value=lambda flag:command[command.index(flag)+1]
            updates=int(value('--updates'))
            assert (completed+updates if mode=='defense' else updates)*8*128==TARGET
            assert value('--eval-every')=='0'
            assert value('--checkpoint-every')=='100'
            if mode=='joint':
                assert value('--target-environment-steps')==str(TARGET)
                if completed:assert '--init' not in command
            if completed:assert value('--resume')=='/saved.pt'
    assert UPDATES==2000


def test_resume_cannot_silently_restart_budget_or_overrun():
    for args in ((None,20),(Path('/saved.pt'),0),(Path('/saved.pt'),2000)):
        with pytest.raises(ValueError):full_command(Path('/p'),Path('/i'),Path('/o'),'defense','baseline','cuda:0',42,resume=args[0],completed_updates=args[1])


def test_full_job_preparation_preserves_four_distinct_selected_models(monkeypatch,tmp_path):
    import json
    import runpy
    from scripts.kaggle import prepare_ppo_selected_full as builder
    study=tmp_path/'study';shared=study/'shared';shared.mkdir(parents=True)
    for name in ('manifest.json','teacher_defense.pt','teacher_joint.pt'):(shared/name).write_text(name)
    for owner in builder.OWNERS:
        folder=study/owner;folder.mkdir()
        (folder/'launch_config.json').write_text(json.dumps(dict(dataset=owner+'/private-study',pins={p.name:builder.sha(p) for p in shared.iterdir()})))
    selection=dict(stage='confirmed',source_sha256='source',test_used_for_selection=False,
                   modes=dict(defense=dict(full_training_profiles=['baseline','economy_soft']),joint=dict(full_training_profiles=['economy_strong','opportunity'])))
    monkeypatch.setattr(builder,'select',lambda *args:selection)
    report=builder.prepare(study,tmp_path/'confirmation',tmp_path/'full')
    assert len(report['jobs'])==4 and report['submitted'] is False
    identities=[]
    for job in report['jobs']:
        identities.append((job['mode'],job['profile']))
        code=tmp_path/'full'/job['owner']/'job/train_full.py'
        namespace=runpy.run_path(str(code),run_name='prepared_full')
        assert namespace['LAUNCH_CONFIG']['selection']==selection
        assert namespace['TARGET']==2048000
        assert namespace['LAUNCH_CONFIG']['dataset'].startswith(job['owner']+'/')
    assert identities==[('defense','baseline'),('defense','economy_soft'),('joint','economy_strong'),('joint','opportunity')]


def test_full_preparation_refuses_missing_confirmation(tmp_path):
    from scripts.kaggle.prepare_ppo_selected_full import prepare
    with pytest.raises((FileNotFoundError,ValueError)):prepare(tmp_path/'missing',tmp_path/'confirmation',tmp_path/'out')
    assert not (tmp_path/'out').exists()
