"""Freeze two cross-cohort full runs as isolated children of one GPU job.

Preparation only. Exact-package preflight and dedicated output collection /
verification must be completed before submission. No network calls here.
"""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import shutil

from scripts.attack.select_pilot_pool import select
from scripts.attack.verify_continuous_focus import verify
from scripts.kaggle.prepare_selected_continuous_full import full_only_bootstrap
from scripts.kaggle.run_five_round_robin import sha


def isolated_child(source, index):
    """Change orchestration/layout only; keep the full trainer function exact."""
    if index not in (0, 1):
        raise ValueError('One child per CUDA device')
    tree = ast.parse(full_only_bootstrap(source))
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    run = next(n for n in execute.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    original_run = ast.dump(run)
    mount = next(n for n in execute.body if isinstance(n, ast.FunctionDef) and n.name == 'mount')
    assert isinstance(mount.body[-1], ast.Return)
    mount.body[-1].value = ast.parse("paths[0]/config['bundle']", mode='eval').body
    for node in execute.body:
        if isinstance(node, ast.Assign):
            names = [n.id for n in node.targets if isinstance(n, ast.Name)]
            if names == ['project']:
                node.value = ast.parse(f"Path('/kaggle/working/global_full_child_{index}/project')", mode='eval').body
            elif names == ['output']:
                node.value = ast.parse(f"Path('/kaggle/working/global_full_child_{index}/continuous_focus')", mode='eval').body
    class Orchestration(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'run':
                return node
            return self.generic_visit(node)
        def visit_Assert(self, node):
            old = ast.parse("selection['gate_met'] and len(selection['selected']) == 2", mode='eval').body
            if ast.dump(node.test) == ast.dump(old):
                node.test = ast.parse("selection['gate_met'] and len(selection['selected']) == 1", mode='eval').body
            return node
        def visit_JoinedStr(self, node):
            # Only the outer full-run device assignment changes. The nested
            # run() is deliberately skipped by visit_FunctionDef above.
            old = ast.parse("f'cuda:{i}'", mode='eval').body
            if ast.dump(node) == ast.dump(old):
                return ast.Constant(f'cuda:{index}')
            return self.generic_visit(node)
    execute = Orchestration().visit(execute)
    if original_run != ast.dump(next(n for n in execute.body if isinstance(n, ast.FunctionDef) and n.name == 'run')):
        raise ValueError('Full training logic changed')
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + '\n'


def driver(config):
    import json, subprocess, sys
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    roots = [p for p in (Path('/kaggle/input')/config['slug'],
              Path('/kaggle/input/datasets')/config['owner']/config['slug']) if p.is_dir()]
    if len(roots) != 1:
        raise ValueError('Ambiguous paired input mount')
    inputs = roots[0]
    for name, digest in config['pins'].items():
        if sha(inputs/name) != digest:
            raise ValueError('Paired input changed: '+name)
    def child(index):
        process = subprocess.Popen([sys.executable, str(inputs/f'bundle_{index}/train.py')],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with Path(f'/kaggle/working/global_full_child_{index}_driver.log').open('x') as log:
            for line in process.stdout:
                log.write(line); log.flush()
                try:
                    row = json.loads(line)
                except ValueError:
                    row = dict(event='GLOBAL_CHILD_LOG', message=line.rstrip())
                if isinstance(row, dict):
                    row['global_child'] = index
                    if row.get('event') == 'TRIAL_PROGRESS':
                        row['trial_id'] = f'global{index}_'+row['trial_id']
                print(json.dumps(row), flush=True)
        if process.wait():
            raise RuntimeError(f'Global full child{index} failed')
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(child, i) for i in (0, 1)]
        for future in futures:
            future.result()
    Path('/kaggle/working/global_full_result.json').write_text(json.dumps(dict(
        status='complete', children=2, full_steps=8192000, pilot_steps_repeated=0,
        quality_accepted=False, test_used_for_selection=False))+'\n')


def prepare(prepared_roots, output, owner, pair_index=0):
    ranking = select(prepared_roots, limit=4)
    if not ranking['gate_met']:
        raise ValueError('Two independently eligible profiles required before full preparation')
    chosen = ranking['selected'][2*pair_index:2*pair_index+2]
    if pair_index not in (0, 1) or len(chosen) != 2:
        raise ValueError('A full pair requires two selected candidates')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous full preparation')
    # Reverify every selected origin and controller before writing a bundle.
    verified = []
    for row in chosen:
        root = Path(row['prepared'])
        proof = verify(root/'collected', root)
        original = json.loads((root/'preparation.json').read_text())
        if sha(root/'job/train.py') != original['bootstrap_sha256']:
            raise ValueError('Selected controller changed')
        eligible = [r['trial'] for r in proof['pilot_selection']['ranking'] if r['eligible']]
        if row['trial'] not in eligible:
            raise ValueError('Selected trial not independently eligible')
        verified.append((root, proof, original))
    output.mkdir(parents=True)
    dataset = output/'dataset'; dataset.mkdir()
    job = output/'job'; job.mkdir()
    identity = sha(Path(chosen[0]['prepared'])/'job/train.py')[:8] + '-' + str(pair_index)
    slug = 'hastycr-global-full-inputs-'+identity
    kernel = owner+'/hastycr-global-full-'+identity
    (dataset/'global_selection.json').write_text(json.dumps(ranking, indent=2)+'\n')
    children = []
    for index, (row, (root, proof, original)) in enumerate(zip(chosen, verified)):
        bundle = dataset/f'bundle_{index}'
        bundle.mkdir()
        config = copy.deepcopy(original['config'])
        for name, digest in config['pins'].items():
            if sha(root/'dataset'/name) != digest:
                raise ValueError('Original dataset changed')
            destination = bundle/name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/'dataset'/name, destination)
        # Source archives are verified by prepare_source() inside each child.
        shutil.copyfile(root/'dataset/source.tar.gz', bundle/'source.tar.gz')
        for name, value in [('initial_selection.json', proof['initial_selection']),
                            ('selection.json', dict(gate_met=True, selected=[row])),
                            ('pilot_verified.json', proof)]:
            (bundle/name).write_text(json.dumps(value, indent=2)+'\n')
            config['pins'][name] = sha(bundle/name)
        config.update(owner=owner, slug=slug, source_slug=slug, bundle=f'bundle_{index}', allow_full_training=True)
        tree = ast.parse((root/'job/train.py').read_text())
        if not isinstance(tree.body[-1], ast.Expr) or not isinstance(tree.body[-1].value, ast.Call):
            raise ValueError('Unexpected frozen launch')
        tree.body.pop()
        source = isolated_child(ast.unparse(tree), index)+'\nexecute('+repr(config)+')\n'
        compile(source, 'train.py', 'exec')
        (bundle/'train.py').write_text(source)
        children.append(dict(index=index, selected=row, config=config,
            bootstrap_sha256=sha(bundle/'train.py'), full_steps=4096000,
            remote_output=f'global_full_child_{index}/continuous_focus'))
    config = dict(owner=owner, slug=slug,
        pins={str(path.relative_to(dataset)):sha(path) for path in dataset.rglob('*') if path.is_file()})
    code='import hashlib\nfrom pathlib import Path\n'+inspect.getsource(sha)+'\n'+inspect.getsource(driver)+'\ndriver('+repr(config)+')\n'
    compile(code, 'train.py', 'exec'); (job/'train.py').write_text(code)
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug, title=slug,
        licenses=[dict(name='other')], isPrivate=True), indent=2)+'\n')
    result=dict(kernel=kernel, children=children, config=config, submitted=False, pair_index=pair_index,
        bootstrap_sha256=sha(job/'train.py'), full_steps=8192000, pilot_steps_repeated=0,
        requires_exact_package_preflight=True, requires_dedicated_collector_and_verifier=True,
        quality_accepted=False)
    # Deliberately no kernel-metadata.json until preflight/collection support
    # is implemented and verified; generic submitters cannot launch this yet.
    (output/'preparation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--pair-index', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prepared, args.output, args.owner, args.pair_index)))
