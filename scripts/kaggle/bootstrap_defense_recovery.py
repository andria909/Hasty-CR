"""Pinned recovery notebook entry point; preparing a job never submits it."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

EMBEDDED_CONFIG = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_parts(reference):
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', reference):
        raise ValueError('Expected explicit owner/dataset reference')
    return reference.split('/')


def mounted_dataset(reference, input_root):
    owner, slug = dataset_parts(reference)
    root = Path(input_root)
    candidates = [root/slug, root/'datasets'/owner/slug]
    found = [p for p in candidates if p.is_dir()]
    if len(found) != 1:
        raise ValueError(f'Expected exactly one explicit dataset mount: {reference}')
    return found[0]


def validate_config(config):
    if config.get('schema') != 'defense-recovery-launch-v1':
        raise ValueError('Unknown launch schema')
    for key in ('source_dataset', 'recovered_dataset'):
        dataset_parts(config[key])
    for key in ('controller_manifest_sha256', 'recovery_receipt_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', config[key]):
            raise ValueError('Expected pinned SHA256')
    if config['source_dataset'] == config['recovered_dataset']:
        raise ValueError('Source and recovered weights require separate datasets')


def run(config, *, input_root='/kaggle/input', output='/kaggle/working/recovery'):
    validate_config(config)
    source = mounted_dataset(config['source_dataset'], input_root)
    recovered = mounted_dataset(config['recovered_dataset'], input_root)
    manifest_path = source/'controller_manifest.json'
    if digest(manifest_path) != config['controller_manifest_sha256']:
        raise ValueError('Controller manifest SHA mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'defense-recovery-controller-v1' or manifest.get('not_exact_resume') is not True:
        raise ValueError('Unexpected controller semantics')
    controller = source/'controller.py'
    if digest(controller) != manifest['controller_sha256']:
        raise ValueError('Controller SHA mismatch')
    if digest(source/'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Source manifest SHA mismatch')
    source_manifest = json.loads((source/'manifest.json').read_text())
    if source_manifest['archive_sha256'] != manifest['source_archive_sha256']:
        raise ValueError('Source archive pin mismatch')
    if digest(recovered/'recovery_receipt.json') != config['recovery_receipt_sha256']:
        raise ValueError('Recovery receipt SHA mismatch')
    # Import only after all bootstrap pins have been verified. The controller
    # verifies every extracted source file and weight before deserializing.
    spec = importlib.util.spec_from_file_location('pinned_recovery_controller', controller)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.execute(SimpleNamespace(source_input=source,
        source_extracted_root=source/'source', recovered_input=recovered,
        recovery_receipt_sha256=config['recovery_receipt_sha256'], output=Path(output)))


def prepare(config, destination, kernel):
    """Write reviewable private job files. No API call, upload or GPU action."""
    validate_config(config)
    dataset_parts(kernel)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).read_text()
    marker = 'EMBEDDED_CONFIG'+' = None'
    if template.count(marker) != 1:
        raise ValueError('Job must be prepared from the unbound bootstrap template')
    # Kaggle script kernels upload code_file, not adjacent arbitrary JSON files.
    (destination/'bootstrap.py').write_text(template.replace(marker,
        'EMBEDDED_CONFIG = '+repr(config), 1))
    (destination/'recovery_launch.json').write_text(json.dumps(config, indent=2)+'\n')
    metadata = dict(id=kernel, title='HastyCR Defense Semantic Recovery',
        code_file='bootstrap.py', language='python', kernel_type='script',
        is_private=True, enable_gpu=True, enable_internet=False,
        dataset_sources=[config['source_dataset'],config['recovered_dataset']],
        competition_sources=[], kernel_sources=[])
    (destination/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--kernel')
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else EMBEDDED_CONFIG
    if config is None:parser.error('--config required for unbound template')
    if args.prepare is not None:
        if not args.kernel:parser.error('--prepare requires --kernel')
        prepare(config, args.prepare, args.kernel)
    else:
        if args.kernel:parser.error('--kernel is only valid with --prepare')
        run(config)


if __name__ == '__main__':main()
