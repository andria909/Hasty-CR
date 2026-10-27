import numpy as np
import pytest
import torch
from scripts.kaggle.preflight_global_full_pair import smoke_args,equal_state


def command():
    return ['python','-c','trainer','--init','parent.pt','--device','cuda:1','--output','full',
        '--envs','8','--rollout','128','--updates','4000','--target-environment-steps','4096000',
        '--minibatch-size','256','--eval-every','250','--eval-episodes','36','--checkpoint-every','250',
        '--value-warmup-steps','20480','--wallclock-seconds','34000','--gamma','.999',
        '--supervised-replay-data','own_data','--supervised-replay-weight','1']


def test_smoke_reduces_budget_but_keeps_own_data_and_coefficients():
    original=command();modified=smoke_args(original,'test_output',32,resume='first.pt')
    assert original==command()
    for flag,value in [('--updates','2'),('--target-environment-steps','32'),('--device','cpu'),
                       ('--gamma','.999'),('--supervised-replay-data','own_data'),('--supervised-replay-weight','1')]:
        assert modified[modified.index(flag)+1]==value
    assert '--init'not in modified and modified[modified.index('--resume')+1]=='first.pt'


def test_resume_equality_checks_optimizer_tensors_and_numpy_rng():
    a={'optimizer':{'moments':[torch.tensor([1.,2.])]},'rng':np.arange(4)}
    b={'optimizer':{'moments':[torch.tensor([1.,2.])]},'rng':np.arange(4)}
    assert equal_state(a,b)
    b['optimizer']['moments'][0][1]=3
    assert not equal_state(a,b)


def test_technical_smoke_fixture_cannot_enable_submission(tmp_path):
    import json,hashlib
    from scripts.kaggle.preflight_global_full_pair import enable_submission
    (tmp_path/'job').mkdir();(tmp_path/'job/train.py').write_text('driver')
    (tmp_path/'preparation.json').write_text('{}')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(frozen_sources_verified=True,exact_resume_16_to_32=True,
        preparation_sha256=digest(tmp_path/'preparation.json'),bootstrap_sha256=digest(tmp_path/'job/train.py'),
        children=[dict(technical_fixture=True,optimizer_and_rng_equal=True)]*2,validation_games=24)
    (tmp_path/'preflight.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='technical fixtures'):enable_submission(tmp_path)
    assert not(tmp_path/'job/kernel-metadata.json').exists()
