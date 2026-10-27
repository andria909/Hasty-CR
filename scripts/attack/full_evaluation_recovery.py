"""Commit and recover complete evaluation parts under identical source/weights."""
import json
from pathlib import Path, PurePosixPath
import shutil

from scripts.attack.evaluate_continuous_pool import sha


def source_identity(manifest_path, project):
    manifest_path, project = Path(manifest_path), Path(project).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'frozen-full-evaluation-source-v1' or not manifest.get('files'):
        raise ValueError('Frozen evaluation source manifest required')
    for name, digest in manifest['files'].items():
        path = PurePosixPath(name)
        target = project / name
        if (path.is_absolute() or '..' in path.parts or str(path) != name or '\\' in name
                or target.is_symlink() or not target.resolve().is_relative_to(project) or sha(target) != digest):
            raise ValueError('Evaluation source changed or unsafe')
    return sha(manifest_path)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def identity(protocol_path, source_digest, jobs):
    return dict(schema='full-evaluation-run-identity-v1', protocol_sha256=sha(protocol_path),
                source_manifest_sha256=source_digest, jobs=jobs)


def commit_part(output, job, report, run_identity):
    output = Path(output)
    path = output / job['file']
    write_json(path, report)
    # Written last. A crash during the report write leaves no reusable receipt.
    write_json(output / (job['file'] + '.receipt.json'),
               dict(schema='full-evaluation-part-receipt-v1', file=job['file'],
                    sha256=sha(path), games=len(report['rows']),
                    protocol_sha256=run_identity['protocol_sha256'],
                    source_manifest_sha256=run_identity['source_manifest_sha256']))


def restore_parts(source, output, run_identity, protocol, blueprint):
    from scripts.attack.verify_full_candidate_evaluation import validate_part
    source, output = Path(source), Path(output)
    if json.loads((source / 'run_identity.json').read_text()) != run_identity:
        raise ValueError('Recovery source, protocol or schedule differs')
    jobs = {j['file']: j for j in run_identity['jobs']}
    accepted = {}
    for receipt_path in sorted(source.glob('*.receipt.json')):
        receipt = json.loads(receipt_path.read_text())
        name = receipt['file']
        if (receipt.get('schema') != 'full-evaluation-part-receipt-v1' or name not in jobs
                or receipt_path.name != name + '.receipt.json'
                or receipt['protocol_sha256'] != run_identity['protocol_sha256']
                or receipt['source_manifest_sha256'] != run_identity['source_manifest_sha256']):
            raise ValueError('Foreign evaluation recovery receipt')
        path = source / name
        if path.is_symlink() or receipt_path.is_symlink() or sha(path) != receipt['sha256']:
            raise ValueError('Recovery part changed')
        report = json.loads(path.read_text())
        rows = validate_part(report, jobs[name], protocol, blueprint)
        if receipt['games'] != len(rows):
            raise ValueError('Recovery receipt count differs')
        accepted[name] = report
    # Validate the whole recovery set before copying any completed result.
    for name in accepted:
        for filename in (name, name + '.receipt.json'):
            target = output / filename
            with target.open('xb') as stream, (source / filename).open('rb') as original:
                shutil.copyfileobj(original, stream)
    return accepted
