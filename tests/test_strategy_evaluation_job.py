import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts.kaggle.run_strategy_evaluation import prepare_source, commands


def package(tmp_path,extracted=False):
    source=tmp_path/'source';source.mkdir()
    raw=b'print("verified")\n';members={'sim/probe.py':raw}
    archive=source/'source.tar.gz'
    with tarfile.open(archive,'w:gz')as stream:
        info=tarfile.TarInfo('sim/probe.py');info.size=len(raw);stream.addfile(info,io.BytesIO(raw))
    manifest=dict(files={name:hashlib.sha256(data).hexdigest()for name,data in members.items()},
                  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (source/'manifest.json').write_text(json.dumps(manifest))
    return source,manifest,hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()


def test_pinned_source_materializes_and_tampering_is_rejected(tmp_path):
    source,manifest,pin=package(tmp_path)
    out=tmp_path/'verified';prepare_source(source,out,pin)
    assert (out/'sim/probe.py').read_bytes()==b'print("verified")\n'
    (source/'source.tar.gz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='archive SHA'):
        prepare_source(source,tmp_path/'other',pin)


def test_explicit_autoextracted_source_is_verified(tmp_path):
    source,manifest,pin=package(tmp_path)
    # Build a separate input mount instead of deleting the original archive.
    mount=tmp_path/'auto';(mount/'source/sim').mkdir(parents=True)
    (mount/'manifest.json').write_bytes((source/'manifest.json').read_bytes())
    (mount/'source/sim/probe.py').write_bytes(b'print("verified")\n')
    prepare_source(mount,tmp_path/'verified',pin)
    (mount/'source/sim/probe.py').write_bytes(b'changed')
    with pytest.raises(ValueError,match='member SHA'):
        prepare_source(mount,tmp_path/'other',pin)


def test_path_escape_rejected_before_output_creation(tmp_path):
    source,manifest,pin=package(tmp_path)
    manifest['files']['../outside.py']='a'*64
    (source/'manifest.json').write_text(json.dumps(manifest))
    pin=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='Unsafe evaluation source path'):prepare_source(source,out,pin)
    assert not out.exists()


def test_full_skill_splits_and_all_baselines_use_distinct_gpus(tmp_path):
    policy=tmp_path/'policy.pt';policy.write_bytes(b'weights')
    plan=commands(tmp_path/'project',policy,tmp_path/'outputs','protocol-sha')
    assert [name for name,_ in plan['skills']]==['validation','test']
    assert all(args[args.index('--device')+1]=='cuda:0'for _,args in plan['skills'])
    args=plan['duels'][0][1]
    assert args[args.index('--device')+1]=='cuda:1'
    assert args[args.index('--candidate-mode')+1]=='joint_enemy_presence_v1'
    assert args[args.index('--matches')+1]=='20'and '--baseline-manifest'in args
    assert not any('train_attack'in item for chain in plan.values()for _,args in chain for item in args)
