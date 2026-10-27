"""Freeze evaluation code on the exact physics of the selected full training.

No candidate weights, test inference, uploads or GPU submission. The source can
be prepared while full training finishes, then used by the evaluation packer.
"""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from scripts.attack.evaluate_full_candidates import sha
from scripts.kaggle.run_five_round_robin import prepare_source

EVALUATORS = (
    'scripts/attack/evaluate_continuous_pool.py',
    'scripts/attack/evaluate_full_candidates.py',
    'scripts/attack/verify_full_candidate_evaluation.py',
    'scripts/attack/full_evaluation_recovery.py',
)


def freeze(prepared_roots, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Frozen evaluation source already exists')
    sources = []
    for prepared in map(Path, prepared_roots):
        prep = json.loads((prepared / 'preparation.json').read_text())
        for child in prep['children']:
            root = prepared / 'dataset' / f"bundle_{child['index']}"
            config = child['config']
            for name in ('manifest.json', 'extensions.tar.gz'):
                if sha(root / name) != config['pins'][name]:
                    raise ValueError('Frozen training source changed')
            manifest = json.loads((root / 'manifest.json').read_text())
            if sha(root / 'source.tar.gz') != manifest['archive_sha256']:
                raise ValueError('Frozen source archive changed')
            files = dict(manifest['files']); files.update(config['extensions'])
            # Training-only scripts may legitimately differ; physics, public
            # observations and assets may not silently vary between contestants.
            contract = {n: digest for n, digest in files.items() if not n.startswith('scripts/') and n not in ('sim/train_attack_ppo.py',)}
            sources.append(dict(root=root, config=config, contract=contract, kernel=prep['kernel']))
    if not sources or any(s['contract'] != sources[0]['contract'] for s in sources):
        raise ValueError('Candidates do not share one frozen simulation contract')
    output.mkdir(parents=True, exist_ok=False)
    project = output / 'project'; source = sources[0]; root = source['root']; config = source['config']
    prepare_source(root, project, config['source_manifest_sha256'])
    with tarfile.open(root / 'extensions.tar.gz') as archive:
        members = archive.getmembers()
        if (len(members) != len(config['extensions']) or {m.name for m in members} != set(config['extensions'])
                or any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members)):
            raise ValueError('Invalid frozen extension archive')
        archive.extractall(project, filter='data')
    for name, digest in config['extensions'].items():
        if sha(project / name) != digest:
            raise ValueError('Frozen extension hash differs')
    for name in EVALUATORS:
        target = project / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    files = {str(p.relative_to(project)): sha(p) for p in sorted(project.rglob('*')) if p.is_file()}
    # This archive contains only the original allowlisted training source/assets
    # plus the explicit three evaluation modules, never workspace secrets/logs.
    with tarfile.open(output / 'source.tar.gz', 'w:gz') as archive:
        for name in sorted(files):
            archive.add(project / name, arcname=name, recursive=False)
    manifest = dict(schema='frozen-full-evaluation-source-v1', files=files,
                    archive_sha256=sha(output / 'source.tar.gz'),
                    training_kernels=sorted({s['kernel'] for s in sources}),
                    evaluator_files={n: files[n] for n in EVALUATORS},
                    shared_training_contract=source['contract'], candidate_inference_run=False)
    with (output / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n')
    return dict(manifest_sha256=sha(output / 'manifest.json'), files=len(files), output=str(output), submitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.prepared, args.output), indent=2))
