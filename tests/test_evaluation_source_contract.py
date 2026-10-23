import hashlib
import json

import pytest

from sim.evaluation_source_contract import JOINT_FILES,verify_joint_source


def fixture(tmp_path):
    digest=lambda data:hashlib.sha256(data).hexdigest()
    original={}
    for name in JOINT_FILES:
        path=tmp_path/name;path.parent.mkdir(exist_ok=True,parents=True)
        raw=b'return chosen.pos\n'if name=='sim/defense_env.py'else name.encode()
        path.write_bytes(raw);original[name]=digest(raw)
    (tmp_path/'sim/defense_env.py').write_text('return arena.clamp_to_arena(chosen.pos)\n')
    current={name:digest((tmp_path/name).read_bytes())for name in JOINT_FILES}
    fingerprint=lambda files:digest(json.dumps(files,sort_keys=True).encode())
    return original,fingerprint(original),fingerprint(current)


def test_declared_target_migration_accepts_original_joint_weights(tmp_path):
    original,old,new=fixture(tmp_path)
    verify_joint_source(old,new,original,tmp_path)
    with pytest.raises(ValueError,match='semantics'):verify_joint_source(old,new)


@pytest.mark.parametrize('mutation',['engine','environment','fingerprint','original'])
def test_migration_does_not_disable_source_integrity(tmp_path,mutation):
    original,old,new=fixture(tmp_path)
    if mutation=='engine':(tmp_path/'sim/engine.py').write_text('altered physics')
    elif mutation=='environment':(tmp_path/'sim/defense_env.py').write_text('unrelated environment')
    elif mutation=='fingerprint':old='untrusted'
    else:original['sim/engine.py']='untrusted'
    with pytest.raises(ValueError):verify_joint_source(old,new,original,tmp_path)


def test_eval_lineage_rejects_unrelated_source_files():
    from scripts.kaggle.prepare_selected_ppo_evaluation import verify_evaluation_lineage
    manifest=dict(training_source_archive_sha256='old',original_training_files={'sim/engine.py':'engine','sim/defense_env.py':'old-env'},
        environment_migration={'migrated_training_source_sha256':'new','evaluation_defense_env_sha256':'new-env'})
    verify_evaluation_lineage(manifest,dict(archive_sha256='old',files=manifest['original_training_files']))
    files=dict(manifest['original_training_files'],**{'sim/defense_env.py':'new-env'})
    verify_evaluation_lineage(manifest,dict(archive_sha256='new',files=files))
    files['sim/engine.py']='other'
    with pytest.raises(ValueError):verify_evaluation_lineage(manifest,dict(archive_sha256='new',files=files))
