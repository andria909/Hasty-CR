import json
from pathlib import Path
import pytest

from scripts.kaggle.prepare_continuous_focus import freeze_template,bind


def test_frozen_template_compiles_and_cannot_bind_global_block_weights(tmp_path):
    template=tmp_path/'template';manifest=freeze_template(template)
    assert 'sim/continuous_sampling.py'in manifest['extensions']
    compile((template/'bootstrap.txt').read_text(),'bootstrap','exec')
    report=tmp_path/'old.json';report.write_text(json.dumps(dict(schema='verified-continuous-ppo-training-v1',
        all_training_budgets_verified=True,kernel='andriashadrin/hastycr-continuous-ppo-20261026',models=[])))
    with pytest.raises(ValueError,match='Corrected-slot'):bind(template,report,tmp_path/'job')
    assert not(tmp_path/'job').exists()


def test_changed_template_is_rejected_before_weights_or_upload(tmp_path):
    template=tmp_path/'template';freeze_template(template)
    (template/'protocol.json').write_text('{}')
    with pytest.raises(ValueError,match='template changed'):bind(template,tmp_path/'absent.json',tmp_path/'job')


def test_report_label_cannot_disguise_a_global_block_checkpoint(tmp_path):
    import hashlib,torch
    template=tmp_path/'template';freeze_template(template)
    weight=tmp_path/'wrong.pt';torch.save(dict(config={'observation_contract':'joint_public_presence_continuous_v1'},
        state_dict={'scalar.0.weight':torch.zeros(1,309)}),weight)
    digest=hashlib.sha256(weight.read_bytes()).hexdigest()
    report=tmp_path/'claimed_corrected.json';report.write_text(json.dumps(dict(schema='verified-continuous-ppo-training-v1',
        all_training_budgets_verified=True,kernel='andreiperelaiko/hastycr-continuous-slots-20261026',
        models=[dict(id=n,checkpoint=str(weight),checkpoint_sha256=digest)for n in('a','b')])) )
    with pytest.raises(ValueError,match='slot input contract'):bind(template,report,tmp_path/'job')
    assert not(tmp_path/'job').exists()
