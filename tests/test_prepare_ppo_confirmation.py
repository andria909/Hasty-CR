import json
from pathlib import Path
import runpy
import inspect

import pytest
from scripts.kaggle import prepare_ppo_confirmation as module



def test_unfinished_study_cannot_prepare_second_seed(tmp_path):
    # No fabricated screening winner can bypass verification of collected outputs.
    with pytest.raises((FileNotFoundError,ValueError)):
        module.prepare(tmp_path/'missing-study',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_confirmation_preserves_submitted_driver_and_input_pins(monkeypatch,tmp_path):
    from scripts.kaggle import run_five_round_robin as helpers
    study=tmp_path/'study';shared=study/'shared';shared.mkdir(parents=True)
    for name in ('manifest.json','teacher_defense.pt','teacher_joint.pt'):(shared/name).write_text(name)
    driver=Path('scripts/kaggle/run_ppo_coefficient_study.py').read_text()
    driver=driver.replace('from scripts.kaggle.run_five_round_robin import sha,prepare_source',inspect.getsource(helpers.sha)+'\n'+inspect.getsource(helpers.prepare_source))
    for owner in module.OWNERS:
        folder=study/owner;job=folder/'job';job.mkdir(parents=True)
        config=dict(dataset=owner+'/private-study',pins={p.name:module.sha(p) for p in shared.iterdir()},workers=[[],[]])
        (folder/'launch_config.json').write_text(json.dumps(config))
        (job/'study.py').write_text(driver.replace('LAUNCH_CONFIG=None','LAUNCH_CONFIG='+repr(config)))
    profiles=['baseline','economy_soft','economy_strong']
    selected=dict(source_sha256='frozen',modes={mode:dict(confirmation_profiles=profiles) for mode in ('defense','joint')})
    monkeypatch.setattr(module,'select',lambda *a:selected)
    result=module.prepare(study,tmp_path/'out')
    assert result['submitted'] is False and result['trials']==6
    identities=[]
    for job in result['jobs']:
        folder=tmp_path/'out'/job['owner']
        config=json.loads((folder/'launch_config.json').read_text())
        old=json.loads((study/job['owner']/'launch_config.json').read_text())
        assert config['pins']==old['pins'] and config['dataset']==old['dataset']
        standalone=runpy.run_path(str(folder/'job/study.py'),run_name='prepared_confirmation')
        assert standalone['LAUNCH_CONFIG']==config
        for i,worker in enumerate(config['workers']):
            assert len(worker)==1
            trial=worker[0];identities.append((trial['mode'],trial['profile'],trial['seed']))
            command=standalone['training_command'](Path('/project'),Path('/input'),Path('/out'),trial['mode'],trial['profile'],f'cuda:{i}',trial['seed'])
            assert command[command.index('--seed')+1]=='43'
            assert command[command.index('--updates')+1]=='100'
    assert len(set(identities))==6 and {i[2] for i in identities}=={43}
