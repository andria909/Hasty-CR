"""Offline evaluation-only transplant of the pinned frozen escort repair.

All old evaluation bytes except defense_env.py remain identical. No deserialization,
training, model selection, credentials, remote calls or broad directory copying.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.defense.pack_evaluation import safe_name

EVALUATION_ARCHIVE_SHA256='87fc3e495ec1486b3927fe64ce31bb302c0b448bbc656304fb719ea272a18016'
RECOVERY_ARCHIVE_SHA256='4375e1b1650e82c5b4114eda7872ef79e6321887b837a07356af48aeb5c7c5a1'
PROTOCOL_FINGERPRINT='c03b49396e95de983419211e476d732e1841ad97bb8d6a79c9b74ed97397437f'
PROTOCOL_BYTES_SHA256='8b43e0925ee2d94b413cd6b1894c1db0780a5abf4674c62ed99d643c9c3fc358'
OLD_ENV_SHA256='58e305a098a068e13f53d5372da05d0d90aefedc68dc35a2194032e0eb4b3472'
NEW_ENV_SHA256='defc163b61c0fdae7e6c10e0ac57d5bf371f185a33c26662be7db97ae3dadf9c'
V25_SHA256='d65ab55893592c0e4655a1882db459c4a7fb1ebd9e10afaf2daa4d2b369fc972'
V25_PATH='checkpoints/portable_v25/portable_v25_cannon_economy_best.pt'
ENV='sim/defense_env.py'
PROTOCOL='data/defense/evaluation_protocol.json'
REGRESSION='tests/test_defense_escort_triggers.py'


def sha(data):return hashlib.sha256(data).hexdigest()


def refuse_credentials(name,data):
    parts=Path(name).parts
    if any(part.lower()in ('.kaggle','.env','access_token','kaggle.json','id_rsa','id_ed25519')
           or part.startswith('KGAT_')for part in parts):
        raise ValueError('Credential-like path is forbidden in evaluation sources')
    if re.search(rb'KGAT_[A-Za-z0-9_-]{16,}',data)or re.search(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',data):
        raise ValueError('Credential-like content is forbidden in evaluation sources')


def verified_package(path,expected_archive_sha256):
    path=Path(path)
    if (path/'manifest.json').is_symlink()or(path/'source.tar.gz').is_symlink():
        raise ValueError('Package input symlinks are forbidden')
    manifest_bytes=(path/'manifest.json').read_bytes();manifest=json.loads(manifest_bytes)
    archive=(path/'source.tar.gz').read_bytes()
    if sha(archive)!=manifest['archive_sha256']or manifest['archive_sha256']!=expected_archive_sha256:
        raise ValueError('Pinned frozen archive SHA mismatch')
    files=manifest['files']
    for name,digest in files.items():
        safe_name(name)
        if not isinstance(digest,str)or re.fullmatch('[0-9a-f]{64}',digest)is None:
            raise ValueError('Invalid manifest file SHA')
    members={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as stream:
        for item in stream:
            name=safe_name(item.name)
            if not item.isfile()or name in members or name not in files:
                raise ValueError('Unsafe, duplicate or unlisted archive member')
            data=stream.extractfile(item).read()
            if sha(data)!=files[name]:raise ValueError('Frozen member SHA mismatch: '+name)
            refuse_credentials(name,data);members[name]=data
    if set(members)!=set(files):raise ValueError('Frozen package is incomplete')
    for name,digest in manifest.get('external_files',{}).items():
        safe_name(name);file=path/name
        if not file.resolve().is_relative_to(path.resolve())or file.is_symlink():
            raise ValueError('Unsafe external package path')
        data=file.read_bytes();refuse_credentials(name,data)
        if sha(data)!=digest:raise ValueError('External package file SHA mismatch')
    return members,manifest,manifest_bytes


def validate_baselines_and_protocol(members,manifest,expected_protocol):
    raw=members[PROTOCOL]
    protocol=json.loads(raw)
    if (sha(raw)!=PROTOCOL_BYTES_SHA256 or manifest['evaluation_protocol_sha256']!=sha(raw)
            or sha(json.dumps(protocol,sort_keys=True).encode())!=expected_protocol
            or manifest['evaluation_protocol_fingerprint']!=expected_protocol):
        raise ValueError('Frozen protocol changed')
    baseline_bytes=members['data/defense/baselines.json']
    if sha(baseline_bytes)!=manifest['baseline_manifest_sha256']:
        raise ValueError('Baseline manifest SHA mismatch')
    catalog=json.loads(baseline_bytes)
    if catalog['schema']!='defense-baselines-v1':raise ValueError('Wrong baseline schema')
    baselines={}
    for row in catalog['checkpoints']:
        name=safe_name(row['path'])
        if not name.startswith('checkpoints/')or row['id']in baselines:
            raise ValueError('Invalid or duplicate explicit baseline path/ID')
        if name not in members or sha(members[name])!=row['sha256']:
            raise ValueError('Baseline weight SHA mismatch')
        baselines[row['id']]=dict(path=name,sha256=row['sha256'])
    if (len(baselines)!=15 or set(baselines)!=set(protocol['baseline_ids'])
            or baselines.get('v25')!=dict(path=V25_PATH,sha256=V25_SHA256)
            or manifest['checkpoint_sha256']!=V25_SHA256):
        raise ValueError('All fifteen explicit baselines and canonical preserved v25 are mandatory')
    return baselines


def build_repaired_evaluation(evaluation_package,recovery_source,output,*,owner='andriashadrin',
        slug='hastycr-defense-eval-repaired-20261022',include_regression=True,
        expected_evaluation_sha256=EVALUATION_ARCHIVE_SHA256,
        expected_recovery_sha256=RECOVERY_ARCHIVE_SHA256,expected_protocol=PROTOCOL_FINGERPRINT):
    output=Path(output)
    if output.exists():raise ValueError('Use a new output directory')
    if re.fullmatch('[A-Za-z0-9_-]+',owner)is None or re.fullmatch('[a-z0-9-]+',slug)is None:
        raise ValueError('Invalid private dataset owner/slug')
    members,old,old_manifest_bytes=verified_package(evaluation_package,expected_evaluation_sha256)
    recovered,recovery,recovery_manifest_bytes=verified_package(recovery_source,expected_recovery_sha256)
    baselines=validate_baselines_and_protocol(members,old,expected_protocol)
    if sha(members[ENV])!=OLD_ENV_SHA256 or sha(recovered[ENV])!=NEW_ENV_SHA256:
        raise ValueError('Repair does not match the exact old/new frozen environment SHA pair')
    migration=recovery['recovery_migration']
    expected_patch=dict(from_sha256=OLD_ENV_SHA256,to_sha256=NEW_ENV_SHA256)
    if (migration['changed_environment_files']!={ENV:expected_patch}
            or recovery['recovery_changed_files'].get(ENV)!=expected_patch):
        raise ValueError('Recovery manifest does not attest the exact environment-only repair')
    original=dict(members)
    members[ENV]=recovered[ENV]
    if include_regression:
        if REGRESSION in members and members[REGRESSION]!=recovered[REGRESSION]:
            raise ValueError('Conflicting regression test member')
        members[REGRESSION]=recovered[REGRESSION]
    changed={name for name,data in original.items()if members[name]!=data}
    if changed!={ENV}:raise ValueError('Evaluation changed outside the environment-only allowlist')
    if members[PROTOCOL]!=original[PROTOCOL]or members['data/defense/scenarios.json']!=original['data/defense/scenarios.json']:
        raise ValueError('Scenario/protocol bytes must remain unchanged')
    if any(members[row['path']]!=original[row['path']]for row in baselines.values()):
        raise ValueError('Baseline weight bytes changed')
    output.mkdir(parents=True,exist_ok=False)
    with(output/'source.tar.gz').open('xb')as destination:
        with gzip.GzipFile(fileobj=destination,mode='wb',mtime=0,filename='')as compressed:
            with tarfile.open(fileobj=compressed,mode='w')as stream:
                for name,data in sorted(members.items()):
                    item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644;item.mtime=0
                    stream.addfile(item,io.BytesIO(data))
    manifest=deepcopy(old)
    manifest.update(archive_sha256=sha((output/'source.tar.gz').read_bytes()),
        files={name:sha(data)for name,data in sorted(members.items())},
        original_evaluation_archive_sha256=old['archive_sha256'],original_evaluation_manifest_sha256=sha(old_manifest_bytes),
        recovery_source_archive_sha256=recovery['archive_sha256'],recovery_source_manifest_sha256=sha(recovery_manifest_bytes),
        evaluation_environment_patch=expected_patch,evaluation_repaired_files=[ENV],
        evaluation_added_files=sorted(set(members)-set(original)),unchanged_original_files=len(original)-1,
        baseline_sha256={name:row['sha256']for name,row in baselines.items()},
        dataset_id=f'{owner}/{slug}',private_dataset=True,quality_passed=None,not_quality_accepted=True,
        external_files={'evaluation_protocol.json':sha(members[PROTOCOL])},
        purpose='Common repaired evaluation environment for fixed candidates; no test-based model selection or acceptance claim')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (output/'evaluation_protocol.json').write_bytes(members[PROTOCOL])
    (output/'dataset-metadata.json').write_text(json.dumps(dict(id=f'{owner}/{slug}',
        title='HastyCR Frozen Repaired Defense Evaluation',licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-package',type=Path,required=True)
    parser.add_argument('--recovery-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--owner',default='andriashadrin')
    parser.add_argument('--slug',default='hastycr-defense-eval-repaired-20261022')
    parser.add_argument('--no-regression-test',action='store_true')
    args=parser.parse_args(argv)
    options=vars(args);options['include_regression']=not options.pop('no_regression_test')
    result=build_repaired_evaluation(**options)
    print(json.dumps({key:result[key]for key in ('archive_sha256','original_evaluation_archive_sha256',
        'recovery_source_archive_sha256','evaluation_environment_patch','unchanged_original_files',
        'evaluation_protocol_fingerprint','private_dataset','quality_passed')},indent=2))


if __name__=='__main__':main()
