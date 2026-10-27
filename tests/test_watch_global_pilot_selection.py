import json
from pathlib import Path

from scripts.attack import watch_global_pilot_selection as module


def setup(tmp_path, complete=True, kernel='owner/job'):
    root=tmp_path/'prepared';(root/'collected').mkdir(parents=True)
    proof=tmp_path/'proof.json';proof.write_text(json.dumps(dict(kernel=kernel,pilot_steps_verified=819200)))
    (root/'collected/collection_receipt.json').write_text(json.dumps(dict(kernel='owner/job',complete=complete,remote_status='COMPLETE')))
    return dict(studies=[dict(proof=str(proof),prepared=str(root))],reports=str(tmp_path),
                pointer=str(tmp_path/'pointer.json'),limit=4)


def test_missing_or_incomplete_or_wrong_kernel_receipt_does_not_trigger_selection(tmp_path):
    config=setup(tmp_path,complete=False)
    assert module.ready_studies(config)==[]
    receipt=Path(config['studies'][0]['prepared'])/'collected/collection_receipt.json'
    row=json.loads(receipt.read_text());row['complete']=True;row['kernel']='other/job';receipt.write_text(json.dumps(row))
    assert module.ready_studies(config)==[]


def test_refresh_invokes_full_verifier_via_selector_only_on_new_verified_cohorts(monkeypatch,tmp_path):
    config=setup(tmp_path);calls=[]
    def select(roots,limit):
        calls.append((roots,limit))
        return dict(eligible_profiles=0,selected=[],gate_met=False)
    monkeypatch.setattr(module,'select',select)
    signature,pointer=module.refresh(config)
    assert pointer['verified_models']==8 and pointer['selected_profiles']==0
    assert not pointer['training_submitted'] and not pointer['ready_for_full_preflight']
    assert len(calls)==1 and calls[0][1]==4
    assert module.refresh(config,signature)==(signature,None)
    assert len(calls)==1
    snapshot=Path(pointer['report']);original=snapshot.read_bytes()
    assert json.loads(Path(config['pointer']).read_text())==pointer
    assert snapshot.read_bytes()==original


def test_restart_reuses_only_identical_snapshot_without_overwriting(monkeypatch,tmp_path):
    config=setup(tmp_path)
    result=dict(eligible_profiles=0,selected=[],gate_met=False)
    monkeypatch.setattr(module,'select',lambda *a,**kw:result)
    signature,pointer=module.refresh(config)
    path=Path(pointer['report']);before=path.read_bytes()
    assert module.refresh(config)[1]==pointer
    assert path.read_bytes()==before
    result['eligible_profiles']=1
    import pytest
    with pytest.raises(ValueError,match='snapshot differs'):module.refresh(config)
    assert path.read_bytes()==before
