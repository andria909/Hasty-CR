from pathlib import Path
import pytest
from scripts.kaggle.run_selected_ppo_evaluation import verify_selection,duel_command


def test_only_completed_final_same_source_policy_can_enter_test():
    selected=dict(schema='selected-full-ppo-evaluation-input-v1',test_used_for_selection=False,training_steps=2048000,training_update=2000,
                  selected_sha256='policy',mode='defense',num_scalars=306,source_sha256='source',policy_optimizer_finite=True)
    config=dict(policy_sha256='policy');manifest=dict(training_source_archive_sha256='source')
    assert verify_selection(selected,config,manifest)=='defense'
    for key,value in [('training_steps',102400),('training_update',100),('test_used_for_selection',True),('num_scalars',309),('source_sha256','other')]:
        bad=dict(selected,**{key:value})
        with pytest.raises(ValueError):verify_selection(bad,config,manifest)
    selected.update(mode='joint',num_scalars=309)
    assert verify_selection(selected,config,manifest)=='joint'


def test_fifteen_opponents_use_same_paired_schedule_and_explicit_input_mode():
    for mode,expected in [('defense','physical_live'),('joint','joint_enemy_presence_v1')]:
        command=duel_command(Path('/source'),Path('/policy.pt'),Path('/output'),mode)
        value=lambda key:command[command.index(key)+1]
        assert value('--candidate-mode')==expected
        assert value('--matches')=='20' and value('--seed')=='710001'
        assert value('--delay-ms')=='150' and value('--device')=='cuda:1'
        assert value('--max-ms')=='400000'


def test_generated_evaluation_has_no_local_script_import_dependency(tmp_path):
    import ast
    import runpy
    from scripts.kaggle.prepare_selected_ppo_evaluation import standalone_source
    config=dict(policy_sha256='fixed',evaluation_dataset='owner/evaluation',selected_dataset='owner/weights')
    code=standalone_source(config)
    assert not [node for node in ast.walk(ast.parse(code)) if isinstance(node,ast.ImportFrom) and (node.module or '').startswith('scripts.')]
    path=tmp_path/'evaluate.py';path.write_text(code)
    values=runpy.run_path(str(path),run_name='prepared_evaluation')
    assert values['LAUNCH_CONFIG']==config
    command=values['duel_command'](Path('/project'),Path('/policy'),Path('/output'),'joint')
    assert command[command.index('--candidate-mode')+1]=='joint_enemy_presence_v1'


def test_private_input_owners_and_existing_outputs_are_guarded(tmp_path):
    from scripts.kaggle.prepare_selected_ppo_evaluation import prepare
    marker=tmp_path/'existing';marker.mkdir();(marker/'keep.txt').write_text('user file')
    args=[tmp_path/'missing']*4
    with pytest.raises(ValueError,match='owner'):prepare(*args,tmp_path/'new',kernel='alice/eval',evaluation_dataset='bob/input',selected_dataset='alice/weights')
    with pytest.raises(ValueError,match='new'):prepare(*args,marker,kernel='alice/eval',evaluation_dataset='alice/input',selected_dataset='alice/weights')
    assert (marker/'keep.txt').read_text()=='user file'
