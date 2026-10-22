"""Build a private evaluation dataset from verified frozen training inputs.

No archive extraction, imported frozen code, checkpoint deserialization, uploads
or training. Only manifest-listed archive members and explicitly allowlisted
checkpoint files enter the new archive. Existing output paths are rejected.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import re
import shutil
import sys
import tarfile

PROJECT_ROOT=Path(__file__).resolve().parents[2]


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def safe_name(name):
    if not isinstance(name,str) or not name or '\\' in name:
        raise ValueError('Unsafe archive/file path')
    path=PurePosixPath(name)
    if not path.parts or path.is_absolute() or '..' in path.parts or '.' in path.parts or str(path)!=name:
        raise ValueError(f'Unsafe or noncanonical archive/file path: {name}')
    return name


def valid_hash(value):
    return isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None


def add_member(members,name,contents):
    safe_name(name)
    if name in members and digest(members[name])!=digest(contents):
        raise ValueError(f'Conflicting duplicate archive path: {name}')
    members[name]=contents


def verified_frozen_package(source):
    source=Path(source)
    manifest_bytes=(source/'manifest.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    archive=(source/'source.tar.gz').read_bytes()
    if digest(archive)!=manifest['archive_sha256']:
        raise ValueError('Frozen archive SHA256 mismatch')
    expected=manifest['files']
    if not isinstance(expected,dict)or not expected:
        raise ValueError('Frozen file manifest must be nonempty')
    for name,sha in expected.items():
        safe_name(name)
        if not valid_hash(sha):raise ValueError('Invalid frozen file SHA256')
    members={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as stream:
        for member in stream:
            name=safe_name(member.name)
            if not member.isfile()or member.issym()or member.islnk():
                raise ValueError('Only regular frozen archive members are permitted')
            if name in members:raise ValueError('Duplicate frozen archive member')
            if name not in expected:raise ValueError(f'Unlisted frozen member: {name}')
            contents=stream.extractfile(member).read()
            if digest(contents)!=expected[name]:raise ValueError(f'Frozen file SHA256 mismatch: {name}')
            members[name]=contents
    if set(members)!=set(expected):raise ValueError('Frozen manifest files are missing from archive')
    # Verify explicitly described external weights without scanning directories.
    external={}
    if 'checkpoint_sha256'in manifest:
        external['v25.pt']=manifest['checkpoint_sha256']
    for row in manifest.get('warmstarts',{}).values():
        name=safe_name(row['file'])
        if '/'in name:raise ValueError('Warmstart must be a plain explicit external filename')
        if name in external and external[name]!=row['sha256']:
            raise ValueError('Conflicting external weight path')
        external[name]=row['sha256']
    for name,sha in external.items():
        if not valid_hash(sha):raise ValueError('Invalid external weight SHA256')
        path=source/name
        if path.is_symlink()or not path.is_file()or digest(path.read_bytes())!=sha:
            raise ValueError(f'External frozen weight SHA256 mismatch: {name}')
    return members,manifest,manifest_bytes,external


def build_package(source_package,baseline_manifest,output,*,project_root=PROJECT_ROOT,
                  owner,slug,title='HastyCR Frozen Defense Evaluation',copy_warmstarts=False):
    output=Path(output);source_package=Path(source_package)
    if output.exists():raise ValueError('Output already exists; choose a new directory')
    if not re.fullmatch('[A-Za-z0-9_-]+',owner)or not re.fullmatch('[a-z0-9-]+',slug):
        raise ValueError('Invalid private dataset owner or slug')
    members,frozen,frozen_bytes,external=verified_frozen_package(source_package)
    baseline_manifest=Path(baseline_manifest)
    baseline_bytes=baseline_manifest.read_bytes()
    baseline_data=json.loads(baseline_bytes)
    if baseline_data.get('schema')!='defense-baselines-v1':raise ValueError('Wrong baseline allowlist schema')
    root=Path(project_root).resolve()
    # Validate allowlist paths before importing any project core modules.
    for row in baseline_data['checkpoints']:
        name=safe_name(row['path'])
        target=(root/name).resolve()
        if not target.is_relative_to(root/'checkpoints')or not valid_hash(row['sha256']):
            raise ValueError('Baseline must be an explicit hashed checkpoint under checkpoints/')
        if (root/name).is_symlink():raise ValueError('Baseline symlinks are not allowed')
    if str(PROJECT_ROOT)not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
    from sim.defense_protocol import EvaluationProtocol,load_baseline_allowlist,V25_SHA256
    allowed=load_baseline_allowlist(baseline_manifest,project_root=root)
    if frozen.get('checkpoint_sha256')!=V25_SHA256:
        raise ValueError('Frozen source does not preserve canonical v25 SHA256')
    add_member(members,'data/defense/baselines.json',baseline_bytes)
    for identifier,row in allowed.items():
        name=Path(row['path']).relative_to(root).as_posix()
        contents=Path(row['path']).read_bytes()
        if digest(contents)!=row['sha256']:raise ValueError('Baseline changed during packaging')
        add_member(members,name,contents)
    scenarios=json.loads(members['data/defense/scenarios.json'])['scenarios']
    identifiers=[row['scenario_id']for row in scenarios if row['split']=='test']
    if len(identifiers)!=len(set(identifiers)):raise ValueError('Duplicate frozen test scenario IDs')
    protocol=EvaluationProtocol(test_scenario_ids=tuple(sorted(identifiers)),
        baseline_ids=tuple(sorted(allowed)),paired_seeds=tuple(range(710001,710011)),
        defense_seeds=(810001,810002,810003))
    protocol_bytes=(json.dumps(protocol.to_dict(),indent=2,sort_keys=True)+'\n').encode()
    add_member(members,'data/defense/evaluation_protocol.json',protocol_bytes)
    # All verification precedes output creation; no preexisting files are replaced.
    output.mkdir(parents=True,exist_ok=False)
    archive_path=output/'source.tar.gz'
    with archive_path.open('xb')as destination:
        with gzip.GzipFile(fileobj=destination,mode='wb',mtime=0,filename='')as compressed:
            with tarfile.open(fileobj=compressed,mode='w')as stream:
                for name,contents in sorted(members.items()):
                    info=tarfile.TarInfo(name);info.size=len(contents);info.mode=0o644;info.mtime=0
                    stream.addfile(info,io.BytesIO(contents))
    manifest=deepcopy(frozen)
    manifest.update(files={name:digest(contents)for name,contents in sorted(members.items())},
        archive_sha256=digest(archive_path.read_bytes()),
        source_archive_sha256=frozen['archive_sha256'],source_manifest_sha256=digest(frozen_bytes),
        evaluation_protocol_sha256=digest(protocol_bytes),evaluation_protocol_fingerprint=protocol.fingerprint,
        baseline_manifest_sha256=digest(baseline_bytes),baseline_ids=sorted(allowed),
        test_scenario_count=len(identifiers),head_to_head_games_per_baseline=20,
        medium_threshold_confirmed=False,quality_passed=None,private_dataset=True,
        dataset_id=f'{owner}/{slug}',warmstart_files_included=bool(copy_warmstarts),
        external_files={},purpose='Frozen evaluation preparation; no quality result or training claim')
    if copy_warmstarts:
        for name,sha in external.items():
            shutil.copyfile(source_package/name,output/name)
            if digest((output/name).read_bytes())!=sha:raise ValueError('Copied external weight hash mismatch')
            manifest['external_files'][name]=sha
    (output/'evaluation_protocol.json').write_bytes(protocol_bytes)
    manifest['external_files']['evaluation_protocol.json']=digest(protocol_bytes)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    metadata=dict(id=f'{owner}/{slug}',title=title,licenses=[dict(name='other')],isPrivate=True)
    (output/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-package',type=Path,required=True)
    p.add_argument('--baseline-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project-root',type=Path,default=PROJECT_ROOT)
    p.add_argument('--owner',required=True)
    p.add_argument('--slug',required=True)
    p.add_argument('--title',default='HastyCR Frozen Defense Evaluation')
    p.add_argument('--copy-warmstarts',action='store_true')
    args=p.parse_args(argv)
    result=build_package(**vars(args))
    print(json.dumps({key:result[key]for key in ('archive_sha256','baseline_ids','test_scenario_count',
        'evaluation_protocol_sha256','private_dataset','quality_passed')},indent=2))


if __name__=='__main__':main()
