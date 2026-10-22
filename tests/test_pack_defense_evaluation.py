import hashlib
import io
import json
import tarfile

import pytest

from scripts.defense.pack_evaluation import build_package,digest,safe_name,verified_frozen_package


def fixture_package(tmp_path,monkeypatch):
    import sim.defense_protocol as protocol
    root=tmp_path/'project';root.mkdir()
    canonical=root/protocol.V25_PATH;canonical.parent.mkdir(parents=True);canonical.write_bytes(b'canonical-fixture-v25')
    sha=digest(canonical.read_bytes());monkeypatch.setattr(protocol,'V25_SHA256',sha)
    other=root/'checkpoints/portable_old/best.pt';other.parent.mkdir();other.write_bytes(b'explicit-old-model')
    baseline=dict(schema='defense-baselines-v1',checkpoints=[
        dict(id='v25',path=protocol.V25_PATH,sha256=sha,observation_mode='physical_live'),
        dict(id='old',path=other.relative_to(root).as_posix(),sha256=digest(other.read_bytes()),observation_mode='legacy_training')])
    baseline_bytes=json.dumps(baseline).encode();manifest_path=root/'baselines.json';manifest_path.write_bytes(baseline_bytes)
    scenarios=dict(schema='defense-v1',scenarios=[
        dict(scenario_id='test/z',split='test'),dict(scenario_id='train/a',split='train'),
        dict(scenario_id='test/a',split='test')])
    members={'sim/frozen.py':b'THIS IS NOT EXECUTABLE PYTHON',
        'data/defense/scenarios.json':json.dumps(scenarios).encode(),
        'data/defense/baselines.json':baseline_bytes}
    source=tmp_path/'source';source.mkdir()
    write_tar(source/'source.tar.gz',members)
    (source/'v25.pt').write_bytes(canonical.read_bytes())
    (source/'ppo_init.pt').write_bytes(b'explicit-warmstart')
    frozen=dict(files={name:digest(data)for name,data in members.items()},
        archive_sha256=digest((source/'source.tar.gz').read_bytes()),checkpoint_sha256=sha,
        warmstarts={'ppo':{'file':'ppo_init.pt','sha256':digest(b'explicit-warmstart'),'source_update':200}})
    (source/'manifest.json').write_text(json.dumps(frozen))
    return root,source,manifest_path,members


def write_tar(path,members):
    with tarfile.open(path,'w:gz')as stream:
        for name,data in members.items():
            entry=tarfile.TarInfo(name);entry.size=len(data);stream.addfile(entry,io.BytesIO(data))


def test_frozen_to_evaluation_preserves_bytes_adds_only_allowlist_and_freezes_protocol(tmp_path,monkeypatch):
    root,source,baseline,members=fixture_package(tmp_path,monkeypatch)
    output=tmp_path/'new-evaluation'
    result=build_package(source,baseline,output,project_root=root,owner='owner',slug='frozen-eval')
    assert result['baseline_ids']==['old','v25'] and result['test_scenario_count']==2
    assert result['warmstarts']['ppo']['source_update']==200
    assert result['warmstart_files_included'] is False and not(output/'ppo_init.pt').exists()
    assert result['private_dataset'] is True and result['quality_passed'] is None
    assert result['medium_threshold_confirmed'] is False
    # Output deliberately retains warmstart provenance without claiming external files exist.
    with tarfile.open(output/'source.tar.gz')as stream:
        actual={item.name:stream.extractfile(item).read()for item in stream}
    assert all(actual[name]==data for name,data in members.items())
    assert set(actual)-set(members)=={
        'checkpoints/portable_v25/portable_v25_cannon_economy_best.pt',
        'checkpoints/portable_old/best.pt','data/defense/evaluation_protocol.json'}
    assert {name:digest(data)for name,data in actual.items()}==result['files']
    protocol=json.loads(actual['data/defense/evaluation_protocol.json'])
    assert protocol['test_scenario_ids']==['test/a','test/z']
    assert protocol['paired_seeds']==list(range(710001,710011))
    assert protocol['defense_seeds']==[810001,810002,810003]
    assert protocol['baseline_ids']==['old','v25']
    assert protocol['medium_threshold_confirmed'] is False
    assert digest((output/'source.tar.gz').read_bytes())==result['archive_sha256']
    assert json.loads((output/'dataset-metadata.json').read_text())['isPrivate']is True
    with pytest.raises(ValueError,match='already exists'):
        build_package(source,baseline,output,project_root=root,owner='owner',slug='frozen-eval')


def test_external_warmstarts_verified_and_optionally_copied(tmp_path,monkeypatch):
    root,source,baseline,_=fixture_package(tmp_path,monkeypatch)
    output=tmp_path/'copied'
    result=build_package(source,baseline,output,project_root=root,owner='owner',slug='frozen-eval',copy_warmstarts=True)
    assert result['external_files']['ppo_init.pt']==digest((output/'ppo_init.pt').read_bytes())
    (source/'ppo_init.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='External frozen weight'):
        build_package(source,baseline,tmp_path/'refused',project_root=root,owner='owner',slug='frozen-eval')
    assert not(tmp_path/'refused').exists()


@pytest.mark.parametrize('bad',['../escape','/absolute','a/../b','./name','a//b','a\\b','.'])
def test_unsafe_members_are_rejected(bad):
    with pytest.raises(ValueError):safe_name(bad)


def test_file_hash_corruption_and_duplicate_conflicts_fail_before_output(tmp_path,monkeypatch):
    root,source,baseline,members=fixture_package(tmp_path,monkeypatch)
    members['sim/frozen.py']=b'corrupt'
    write_tar(source/'source.tar.gz',members)
    manifest=json.loads((source/'manifest.json').read_text());manifest['archive_sha256']=digest((source/'source.tar.gz').read_bytes())
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Frozen file SHA256 mismatch'):
        build_package(source,baseline,tmp_path/'bad',project_root=root,owner='owner',slug='frozen-eval')
    assert not(tmp_path/'bad').exists()


def test_baseline_outside_checkpoint_root_rejected_without_importing_core(tmp_path,monkeypatch):
    root,source,baseline,_=fixture_package(tmp_path,monkeypatch)
    data=json.loads(baseline.read_text());data['checkpoints'][1]['path']='elsewhere.pt'
    baseline.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='under checkpoints'):
        build_package(source,baseline,tmp_path/'bad',project_root=root,owner='owner',slug='frozen-eval')


def test_conflicting_frozen_baseline_path_is_not_silently_replaced(tmp_path,monkeypatch):
    root,source,baseline,members=fixture_package(tmp_path,monkeypatch)
    members['checkpoints/portable_old/best.pt']=b'conflicting-frozen-weights'
    write_tar(source/'source.tar.gz',members)
    manifest=json.loads((source/'manifest.json').read_text())
    manifest['files']={name:digest(data)for name,data in members.items()}
    manifest['archive_sha256']=digest((source/'source.tar.gz').read_bytes())
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Conflicting duplicate archive path'):
        build_package(source,baseline,tmp_path/'bad',project_root=root,owner='owner',slug='frozen-eval')
    assert not(tmp_path/'bad').exists()


def test_noncanonical_frozen_v25_is_refused_even_with_consistent_source_hashes(tmp_path,monkeypatch):
    root,source,baseline,_=fixture_package(tmp_path,monkeypatch)
    (source/'v25.pt').write_bytes(b'wrong-canonical-v25')
    manifest=json.loads((source/'manifest.json').read_text())
    manifest['checkpoint_sha256']=digest((source/'v25.pt').read_bytes())
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='canonical v25'):
        build_package(source,baseline,tmp_path/'bad',project_root=root,owner='owner',slug='frozen-eval')


def test_archive_symlinks_are_rejected_even_when_manifest_names_match(tmp_path,monkeypatch):
    root,source,baseline,_=fixture_package(tmp_path,monkeypatch)
    with tarfile.open(source/'source.tar.gz','w:gz')as stream:
        entry=tarfile.TarInfo('sim/frozen.py');entry.type=tarfile.SYMTYPE;entry.linkname='/etc/passwd'
        stream.addfile(entry)
    manifest=json.loads((source/'manifest.json').read_text())
    manifest['archive_sha256']=digest((source/'source.tar.gz').read_bytes())
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Only regular frozen'):
        build_package(source,baseline,tmp_path/'bad',project_root=root,owner='owner',slug='frozen-eval')
