import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.kaggle.start_pilot_observers import start


def test_current_date_budget_and_private_credentials_are_preserved(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);root=tmp_path/'job';(root/'dataset').mkdir(parents=True)
    (tmp_path/'reports/defense_plan').mkdir(parents=True)
    (root/'preparation.json').write_text(json.dumps(dict(kernel='owner/job',config=dict(owner='owner'),bootstrap_sha256='pin')))
    (root/'launch_receipt.json').write_text(json.dumps(dict(ref='/code/owner/job',versionNumber=1)))
    (root/'dataset/protocol.json').write_text(json.dumps(dict(total_pilot_steps=3276800)))
    calls=[]
    def popen(args,**kwargs):
        calls.append((args,kwargs));return SimpleNamespace(pid=100+len(calls))
    monkeypatch.setattr('subprocess.Popen',popen)
    monkeypatch.setenv('KAGGLE_API_TOKEN','must-not-inherit')
    start(root,'policy_choice','policy-choice-pilot',0,report_date='20261104')
    report=json.loads(Path('reports/defense_plan/policy_choice_training_20261104.json').read_text())
    assert report['pilot_target_steps']==3276800 and not report['actual_cuda_ready']
    assert len(calls)==3
    assert all('KAGGLE_API_TOKEN'not in options['env']for _,options in calls)
    assert 'scripts.kaggle.run_observer'in calls[0][0]
    assert 'reports/defense_plan/policy_choice_verified_20261104.json'in calls[2][0]
    assert json.loads((root/'monitoring.json').read_text())['complete']
    with pytest.raises(ValueError,match='duplicate'):start(root,'policy_choice','policy-choice-pilot',0,report_date='20261104')
    assert len(calls)==3
