import json
import tarfile
import io

import pytest

import scripts.defense.pack_repaired_evaluation as pack


def write_package(root,members,metadata):
    root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(root/'source.tar.gz','w:gz')as stream:
        for name,data in members.items():
            item=tarfile.TarInfo(name);item.size=len(data);stream.addfile(item,io.BytesIO(data))
    manifest=dict(metadata,files={name:pack.sha(data)for name,data in members.items()},
                  archive_sha256=pack.sha((root/'source.tar.gz').read_bytes()))
    (root/'manifest.json').write_text(json.dumps(manifest))
    return manifest


def inputs(tmp_path,monkeypatch):
    old=b'exact old frozen env';new=b'exact repaired escort env'
    monkeypatch.setattr(pack,'OLD_ENV_SHA256',pack.sha(old))
    monkeypatch.setattr(pack,'NEW_ENV_SHA256',pack.sha(new))
    names=['v25']+['old'+str(i)for i in range(14)]
    rows=[];members={pack.ENV:old,'sim/train_defense_grpo.py':b'old trainer stays unchanged',
                     'sim/arena.py':b'old geometry stays unchanged','data/defense/scenarios.json':b'original scenario bytes'}
    for name in names:
        path=pack.V25_PATH if name=='v25'else 'checkpoints/'+name+'/model.pt'
        data=('weight-'+name).encode();members[path]=data
        rows.append(dict(id=name,path=path,sha256=pack.sha(data)))
    monkeypatch.setattr(pack,'V25_SHA256',rows[0]['sha256'])
    protocol=dict(baseline_ids=names,medium_threshold_confirmed=False,paired_seeds=list(range(710001,710011)))
    protocol_bytes=(json.dumps(protocol,sort_keys=True)+'\n').encode()
    fingerprint=pack.sha(json.dumps(protocol,sort_keys=True).encode())
    monkeypatch.setattr(pack,'PROTOCOL_BYTES_SHA256',pack.sha(protocol_bytes))
    members[pack.PROTOCOL]=protocol_bytes
    members['data/defense/baselines.json']=json.dumps(dict(schema='defense-baselines-v1',checkpoints=rows)).encode()
    metadata=dict(evaluation_protocol_sha256=pack.sha(protocol_bytes),evaluation_protocol_fingerprint=fingerprint,
        baseline_manifest_sha256=pack.sha(members['data/defense/baselines.json']),checkpoint_sha256=rows[0]['sha256'])
    evaluation=tmp_path/'evaluation';old_manifest=write_package(evaluation,members,metadata)
    patch=dict(from_sha256=pack.sha(old),to_sha256=pack.sha(new))
    recovery=tmp_path/'recovery'
    repaired=dict(members);repaired[pack.ENV]=new;repaired['sim/train_defense_grpo.py']=b'patched TRAINER MUST NOT TRANSPLANT'
    repaired[pack.REGRESSION]=b'actual escort regression test bytes'
    recovery_manifest=write_package(recovery,repaired,dict(recovery_migration=dict(changed_environment_files={pack.ENV:patch}),
        recovery_changed_files={pack.ENV:patch,'sim/train_defense_grpo.py':{'changed':True}}))
    options=dict(expected_evaluation_sha256=old_manifest['archive_sha256'],
        expected_recovery_sha256=recovery_manifest['archive_sha256'],expected_protocol=fingerprint)
    return evaluation,recovery,members,metadata,options


def archive_members(path):
    with tarfile.open(path/'source.tar.gz')as stream:
        return {item.name:stream.extractfile(item).read()for item in stream}


def test_repaired_bundle_changes_only_env_keeps_all_weights_scenarios_protocol_and_private_metadata(tmp_path,monkeypatch):
    evaluation,recovery,original,_,options=inputs(tmp_path,monkeypatch)
    original_sha=pack.sha((evaluation/'source.tar.gz').read_bytes())
    output=tmp_path/'new'
    manifest=pack.build_repaired_evaluation(evaluation,recovery,output,**options)
    actual=archive_members(output)
    assert {name for name,data in original.items()if actual[name]!=data}=={pack.ENV}
    assert set(actual)-set(original)=={pack.REGRESSION}
    assert actual['sim/train_defense_grpo.py']==b'old trainer stays unchanged'
    assert actual['sim/arena.py']==b'old geometry stays unchanged'
    assert actual[pack.PROTOCOL]==original[pack.PROTOCOL]
    assert actual['data/defense/scenarios.json']==original['data/defense/scenarios.json']
    assert all(actual[name]==data for name,data in original.items()if name.startswith('checkpoints/'))
    assert len(manifest['baseline_sha256'])==15
    assert manifest['quality_passed']is None and manifest['not_quality_accepted']is True
    assert manifest['unchanged_original_files']==len(original)-1
    assert manifest['original_evaluation_archive_sha256']==original_sha
    assert pack.sha((evaluation/'source.tar.gz').read_bytes())==original_sha
    assert json.loads((output/'dataset-metadata.json').read_text())['isPrivate']is True
    assert manifest['dataset_id'].endswith('/hastycr-defense-eval-repaired-20261022')
    with pytest.raises(ValueError,match='new output'):
        pack.build_repaired_evaluation(evaluation,recovery,output,**options)


def test_optional_regression_does_not_replace_any_other_member(tmp_path,monkeypatch):
    evaluation,recovery,original,_,options=inputs(tmp_path,monkeypatch)
    output=tmp_path/'without-test'
    pack.build_repaired_evaluation(evaluation,recovery,output,include_regression=False,**options)
    assert set(archive_members(output))==set(original)


@pytest.mark.parametrize('name,data',[
    ('.kaggle/access_token',b'fake'),('normal.py',b'KGAT_'+b'a'*32),
    ('normal.py',b'-----BEGIN PRIVATE KEY-----'),('../escape',b'fake')])
def test_credentials_and_unsafe_paths_refused_even_with_consistent_manifest_hashes(tmp_path,name,data):
    source=tmp_path/'source'
    manifest=write_package(source,{name:data},{})
    with pytest.raises(ValueError):pack.verified_package(source,manifest['archive_sha256'])


def test_archive_and_member_hash_corruption_fail_closed(tmp_path):
    source=tmp_path/'source'
    manifest=write_package(source,{'safe.py':b'bytes'},{})
    manifest['files']['safe.py']='0'*64
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='member SHA'):
        pack.verified_package(source,manifest['archive_sha256'])
    with pytest.raises(ValueError,match='archive SHA'):
        pack.verified_package(source,'0'*64)


def test_changed_baseline_is_rejected_before_output(tmp_path,monkeypatch):
    evaluation,recovery,original,metadata,options=inputs(tmp_path,monkeypatch)
    original[pack.V25_PATH]=b'changed model'
    manifest=write_package(evaluation,original,metadata)
    options['expected_evaluation_sha256']=manifest['archive_sha256']
    output=tmp_path/'refused'
    with pytest.raises(ValueError,match='weight SHA'):
        pack.build_repaired_evaluation(evaluation,recovery,output,**options)
    assert not output.exists()
