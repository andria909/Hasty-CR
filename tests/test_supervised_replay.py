import json
from pathlib import Path
import numpy as np
import pytest
import torch
from sim.supervised_replay import SuccessfulActionReplay,successful_set_loss

DATA=Path('data/air_position_examples_20261027')


def test_replay_excludes_entire_local_holdout_group():
    replay=SuccessfulActionReplay(DATA,'cpu')
    manifest=json.loads((DATA/'manifest.json').read_text())
    ids=[r['array_index']for r in manifest['records']if r['accepted']and r['seed']!=max(manifest['seeds'])]
    with np.load(DATA/'examples.npz')as arrays:
        assert np.array_equal(replay.planes.numpy(),arrays['planes'][ids])
        assert np.array_equal(replay.scalars.numpy(),arrays['scalars'][ids])
    assert replay.metadata['train_examples']==32
    assert replay.metadata['heldout_examples']==16
    assert replay.metadata['heldout_seed']not in replay.metadata['train_seeds']


def test_auxiliary_recovers_saturated_wrong_choice_without_illegal_gradients():
    logits=torch.tensor([[200.,0.,-20.]],requires_grad=True)
    loss=successful_set_loss(logits,torch.tensor([[True,True,False]]),torch.tensor([[False,True,False]]))
    loss.backward()
    assert loss.item()==pytest.approx(200.)
    assert logits.grad[0,1].item()==pytest.approx(-1.)
    assert logits.grad[0,2].item()==0.


def test_changed_or_validation_data_are_rejected(tmp_path):
    import shutil
    for name in ('examples.npz','manifest.json'):shutil.copyfile(DATA/name,tmp_path/name)
    p=tmp_path/'manifest.json';m=json.loads(p.read_text());m['split']='validation';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='TRAIN'):SuccessfulActionReplay(tmp_path,'cpu')
    m['split']='train';p.write_text(json.dumps(m));(tmp_path/'examples.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='TRAIN'):SuccessfulActionReplay(tmp_path,'cpu')
