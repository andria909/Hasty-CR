import hashlib
import json

import pytest

from scripts.kaggle.run_ppo_selected_full import verify_script_target_migration
from scripts.kaggle.prepare_ppo_target_recovery import original_full_config


def test_recovery_inherits_full_slot_instead_of_screening_configuration(tmp_path):
    full=dict(mode='defense',profile='economy_soft',selection={'stage':'confirmed'})
    path=tmp_path/'launch_config.json';path.write_text(json.dumps({'original':full}))
    assert original_full_config(tmp_path)==full
    path.write_text(json.dumps({'original':{'dataset':'screening'}}))
    with pytest.raises(ValueError):original_full_config(tmp_path)


def fixture(tmp_path):
    source=tmp_path/'project';(source/'sim').mkdir(parents=True)
    old='def target(chosen):\n    return chosen.pos\n'
    new=old.replace('chosen.pos','arena.clamp_to_arena(chosen.pos)')
    (source/'sim/defense_env.py').write_text(new)
    digest=lambda value:hashlib.sha256(value).hexdigest()
    original=dict(archive_sha256='original-source',files={'sim/defense_env.py':digest(old.encode()),'sim/engine.py':'engine-pin'})
    (tmp_path/'original_manifest.json').write_text(json.dumps(original))
    (tmp_path/'resume.pt').write_bytes(b'checkpoint')
    manifest=dict(files=dict(original['files'],**{'sim/defense_env.py':digest(new.encode())}))
    config=dict(migration=dict(schema='script-target-clamp-migration-v1',
        original_manifest_sha256=digest((tmp_path/'original_manifest.json').read_bytes()),
        original_source_sha256='original-source',replacement='arena.clamp_to_arena(chosen.pos)',
        resume_update=796,resume_step=815104,resume_checkpoint_sha256=digest(b'checkpoint')))
    return config,source,manifest


def test_declared_one_expression_migration_accepts_saved_boundary(tmp_path):
    config,source,manifest=fixture(tmp_path)
    assert verify_script_target_migration(config,tmp_path,source,manifest)==tmp_path/'resume.pt'


@pytest.mark.parametrize('tamper',['engine','extra_environment_change','checkpoint','boundary','expression'])
def test_migration_refuses_unrelated_semantics_or_boundary_substitution(tmp_path,tamper):
    config,source,manifest=fixture(tmp_path)
    if tamper=='engine':manifest['files']['sim/engine.py']='changed'
    elif tamper=='extra_environment_change':
        with (source/'sim/defense_env.py').open('a')as stream:stream.write('extra_change=True\n')
    elif tamper=='checkpoint':(tmp_path/'resume.pt').write_bytes(b'other')
    elif tamper=='boundary':config['migration']['resume_update']=700
    else:config['migration']['replacement']='chosen.pos'
    with pytest.raises(ValueError):verify_script_target_migration(config,tmp_path,source,manifest)
