"""Package bound full candidates and frozen evaluation source for Kaggle.

Creates a private, GPU-disabled job pending exact-package preflight. The flat
base64 transport avoids the SDK's implicit nested-folder archive conversion.
"""
import argparse
import base64
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

from scripts.attack.evaluate_full_candidates import validate_protocol, schedule, sha
from scripts.kaggle.run_five_round_robin import prepare_source


def unpack(config, mount, working):
    import base64, io
    mount, working = Path(mount), Path(working)
    encoded = mount / 'evaluation_inputs.b64'
    if sha(encoded) != config['transport_sha256']:
        raise ValueError('Evaluation transport differs')
    raw = base64.b64decode(encoded.read_bytes(), validate=True)
    if hashlib.sha256(raw).hexdigest() != config['archive_sha256']:
        raise ValueError('Evaluation archive differs')
    inputs = working / 'full_eval_inputs'
    inputs.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        members = archive.getmembers()
        if (len(members) != len(config['files']) or {m.name for m in members} != set(config['files'])
                or any(not m.isfile() or PurePosixPath(m.name).is_absolute() or '..' in PurePosixPath(m.name).parts or '\\' in m.name for m in members)):
            raise ValueError('Unsafe evaluation transport members')
        archive.extractall(inputs, filter='data')
    for name, digest in config['files'].items():
        if sha(inputs / name) != digest:
            raise ValueError('Evaluation payload differs')
    project = working / 'full_eval_project'
    prepare_source(inputs / 'source', project, config['source_manifest_sha256'])
    return inputs, project


def bootstrap(config):
    import subprocess, sys
    root = Path('/kaggle/input')
    choices = [p for p in (root / config['slug'], root / 'datasets' / config['owner'] / config['slug']) if p.is_dir()]
    if len(choices) != 1:
        raise ValueError('Ambiguous evaluation dataset mount')
    inputs, project = unpack(config, choices[0], Path('/kaggle/working'))
    print(json.dumps(dict(event='INPUTS_VERIFIED', protocol_sha256=config['files']['protocol.json'],
                         source_manifest_sha256=config['source_manifest_sha256'])), flush=True)
    subprocess.run([sys.executable, '-m', 'scripts.attack.evaluate_full_candidates',
                    '--protocol', str(inputs / 'protocol.json'), '--weights', str(inputs / 'weights'),
                    '--source-manifest', str(inputs / 'source/manifest.json'),
                    '--output', '/kaggle/working/full_candidate_evaluation'], cwd=project, check=True)


def prepare(binding, source, output, owner, slug):
    binding, source, output = Path(binding), Path(source), Path(output)
    if owner not in ('andriashadrin', 'vertel0', 'andreiperelaiko'):
        raise ValueError('A currently authorized working account required')
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', slug):
        raise ValueError('Safe Kaggle slug required')
    protocol = json.loads((binding / 'protocol.json').read_text())
    blueprint = validate_protocol(protocol, binding / 'weights')
    jobs = schedule(protocol, blueprint)
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('schema') != 'frozen-full-evaluation-source-v1' or sha(source / 'source.tar.gz') != manifest['archive_sha256']:
        raise ValueError('Matching frozen evaluation source required')
    candidates = [m for m in protocol['models'] if m['role'] == 'candidate']
    if not {m['origin_kernel'] for m in candidates} <= set(manifest['training_kernels']):
        raise ValueError('Evaluation source was not checked against candidate training')
    # Explicit file list only: never recursively package workspace or tokens.
    names = {'evaluation_blueprint.json'}
    names.update(m['file'] for m in protocol['models'])
    names.update(m['full_proof_file'] for m in candidates)
    payload = {'protocol.json': binding / 'protocol.json',
               'source/manifest.json': source / 'manifest.json', 'source/source.tar.gz': source / 'source.tar.gz'}
    payload.update({'weights/' + n: binding / 'weights' / n for n in names})
    if any(p.is_symlink() for p in payload.values()):
        raise ValueError('Symlink evaluation payload is not allowed')
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / 'upload_dataset'; dataset.mkdir()
    job = output / 'job'; job.mkdir()
    archive_path = output / 'evaluation_inputs.tar.gz'
    pins = {name: sha(path) for name, path in sorted(payload.items())}
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name, path in sorted(payload.items()): archive.add(path, arcname=name, recursive=False)
    transport = dataset / 'evaluation_inputs.b64'
    transport.write_bytes(base64.b64encode(archive_path.read_bytes()))
    config = dict(owner=owner, slug=slug + '-inputs', files=pins,
                  archive_sha256=sha(archive_path), transport_sha256=sha(transport),
                  source_manifest_sha256=sha(source / 'manifest.json'))
    code = ('import hashlib,json,shutil,tarfile\nfrom pathlib import Path,PurePosixPath\n'
            + inspect.getsource(sha) + '\n' + inspect.getsource(prepare_source) + '\n'
            + inspect.getsource(unpack) + '\n' + inspect.getsource(bootstrap)
            + '\nbootstrap(' + repr(config) + ')\n')
    compile(code, 'train.py', 'exec'); (job / 'train.py').write_text(code)
    kernel = owner + '/' + slug
    metadata = dict(id=kernel, title=slug, code_file='train.py', language='python', kernel_type='script',
                    is_private=True, enable_gpu=False, enable_internet=False,
                    dataset_sources=[owner + '/' + config['slug']], competition_sources=[], kernel_sources=[])
    (job / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (dataset / 'dataset-metadata.json').write_text(json.dumps(dict(id=owner + '/' + config['slug'], title=config['slug'],
        licenses=[dict(name='other')], isPrivate=True), indent=2) + '\n')
    result = dict(kernel=kernel, config=config, bootstrap_sha256=sha(job / 'train.py'),
                  total_games=sum(j['games'] for j in jobs), submitted=False, ready_for_submission=False,
                  required_next='Exact decoded package preflight before enabling GPU metadata')
    (output / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('binding', 'source', 'output', 'owner', 'slug'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = prepare(args.binding, args.source, args.output, args.owner, args.slug)
    print(json.dumps({k: result[k] for k in ('kernel', 'total_games', 'submitted', 'ready_for_submission')}))
