from pathlib import Path
import pytest
from scripts.kaggle import start_full_evaluation_observers as module


def test_commands_separate_sdk_and_torch_environments_and_poll_20min():
    commands=module.commands(Path('/job'),Path('/binding'),Path('/report.json'),'vertel0','sdk-python','project-python','vertel0/eval')
    assert commands['stream'][0]==commands['collector'][0]=='sdk-python'
    assert commands['postprocess'][0]=='project-python'
    assert commands['collector'][commands['collector'].index('--poll-seconds')+1]=='1200'
    assert commands['stream'][commands['stream'].index('--reconnect-seconds')+1]=='1200'
    assert '--watch' in commands['postprocess'] and '--binding' in commands['postprocess']
    assert 'submit' not in ' '.join(sum(commands.values(),[]))


def test_duplicate_observers_rejected_before_credentials(tmp_path):
    (tmp_path/'monitoring.json').write_text('{}')
    with pytest.raises(ValueError,match='no duplicates'):module.start(tmp_path,tmp_path,tmp_path/'report.json',1)


def test_fourth_account_is_not_retried(tmp_path):
    with pytest.raises(ValueError,match='working accounts'):module.start(tmp_path,tmp_path,tmp_path/'report.json',3)
