"""Relocate an unlaunched pilot to another authorized account, preserving inputs."""
import argparse
import ast
import json
from pathlib import Path
import shutil

from scripts.kaggle.prepare_motionfix_pilot import sha


def relocate(source, output, owner):
    source, output = Path(source), Path(output)
    receipt = json.loads((source/'launch_receipt.json').read_text())
    if receipt.get('ref') or receipt != {'error': 'Maximum weekly GPU quota of 30.00 hours reached.'}:
        raise ValueError('Only an explicitly quota-rejected, unlaunched pilot may move')
    if owner not in ('andriashadrin', 'vertel0', 'andreiperelaiko'):
        raise ValueError('Unknown authorized owner')
    prep = json.loads((source/'preparation.json').read_text())
    preflight = json.loads((source/'preflight.json').read_text())
    if not preflight.get('frozen_sources_verified') or not preflight.get('exact_resume_16_to_32'):
        raise ValueError('Actual source preflight required')
    if sha(source/'job/train.py') != prep['bootstrap_sha256']:
        raise ValueError('Source controller changed')
    for name, digest in prep['config']['pins'].items():
        if sha(source/'dataset'/name) != digest:
            raise ValueError('Source input changed')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source/'dataset', output/'dataset')
    shutil.copytree(source/'job', output/'job')
    config = prep['config']; config['owner'] = owner
    tree = ast.parse((source/'job/train.py').read_text())
    prior = ast.dump(ast.Module(body=tree.body[:-1], type_ignores=[]))
    tree.body[-1].value.args = [ast.parse(repr(config), mode='eval').body]
    ast.fix_missing_locations(tree)
    path = output/'job/train.py'; path.write_text(ast.unparse(tree)+'\n')
    if ast.dump(ast.Module(body=ast.parse(path.read_text()).body[:-1], type_ignores=[])) != prior:
        raise ValueError('Account relocation changed executable training logic')
    kernel = owner+'/'+prep['kernel'].split('/')[1]
    for name, identity in [('dataset/dataset-metadata.json', owner+'/'+config['slug']),
                           ('job/kernel-metadata.json', kernel)]:
        path = output/name; row = json.loads(path.read_text()); row['id'] = identity
        if name.startswith('job/'):
            row['dataset_sources'] = [owner+'/'+config['slug']]
        path.write_text(json.dumps(row, indent=2)+'\n')
    prep.update(kernel=kernel, bootstrap_sha256=sha(output/'job/train.py'),
                relocated_from=str(source), relocation_reason=receipt['error'])
    preflight['account_relocation'] = dict(source=str(source),
        source_preflight_sha256=sha(source/'preflight.json'), identical_training_ast=True,
        identical_frozen_inputs=True, owner=owner)
    (output/'preparation.json').write_text(json.dumps(prep, indent=2)+'\n')
    (output/'preflight.json').write_text(json.dumps(preflight, indent=2)+'\n')
    return dict(kernel=kernel, root=str(output), submitted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', required=True)
    p.add_argument('--output', required=True); p.add_argument('--owner', required=True)
    a = p.parse_args(); print(json.dumps(relocate(a.source, a.output, a.owner)))
