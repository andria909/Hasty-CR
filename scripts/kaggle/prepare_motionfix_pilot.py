"""Create a separate pilot on current physics; preserve submitted bundles."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
from scripts.attack.sequence_physics import physics_pins


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, output, slug, data=None):
    source, root = Path(source), Path(output)
    if root.exists():
        raise ValueError('New destination required')
    prep = json.loads((source/'preparation.json').read_text())
    config = prep['config']
    if sha(source/'job/train.py') != prep['bootstrap_sha256']:
        raise ValueError('Source bootstrap changed')
    for name, digest in config['pins'].items():
        if sha(source/'dataset'/name) != digest:
            raise ValueError('Source input changed: '+name)
    root.mkdir(parents=True)
    for folder in ('dataset', 'job'):
        shutil.copytree(source/folder, root/folder)
    dataset = root/'dataset'
    base = json.loads((dataset/'manifest.json').read_text())
    archive = dataset/'extensions.tar.gz'
    with tarfile.open(archive) as stream:
        payload = {m.name: stream.extractfile(m).read() for m in stream.getmembers() if m.isfile()}
    pins = physics_pins()
    for name, digest in pins.items():
        payload[name] = Path(name).read_bytes()
        config['extensions'][name] = digest
        if name in base['files']:
            config['replaced'][name] = base['files'][name]
    if set(payload) != set(config['extensions']):
        raise ValueError('Extension coverage differs')
    with tarfile.open(archive, 'w:gz') as stream:
        for name, content in sorted(payload.items()):
            info = tarfile.TarInfo(name); info.size = len(content)
            stream.addfile(info, io.BytesIO(content))
    config['pins']['extensions.tar.gz'] = sha(archive)
    if data:
        data = Path(data)
        labels = json.loads((data/'manifest.json').read_text())
        if labels['physics_pins'] != pins or sha(data/'examples.npz') != labels['data_sha256']:
            raise ValueError('Data physics/hash mismatch')
        for src, dst in [('examples.npz','air_examples.npz'), ('manifest.json','air_manifest.json')]:
            shutil.copyfile(data/src, dataset/dst)
            config['pins'][dst] = sha(dataset/dst)
    protocol_path = dataset/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol.update(physics_pins=pins, refreshed_from=str(source),
                    comparison_scope='Independent corrected-physics cohort; do not pool with old physics results.')
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    config['pins']['protocol.json'] = sha(protocol_path)
    owner = config['owner']; kernel = owner+'/'+slug
    config.update(slug=slug+'-inputs', source_slug=slug+'-inputs')
    tree = ast.parse((root/'job/train.py').read_text())
    tree.body[-1].value.args = [ast.parse(repr(config), mode='eval').body]
    ast.fix_missing_locations(tree)
    (root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    for file, identity in [('dataset/dataset-metadata.json',owner+'/'+config['slug']), ('job/kernel-metadata.json',kernel)]:
        path = root/file; row = json.loads(path.read_text())
        row.update(id=identity,title=identity.split('/')[-1])
        if file.startswith('job/'):
            row.update(dataset_sources=[owner+'/'+config['slug']],enable_gpu=True)
        path.write_text(json.dumps(row,indent=2)+'\n')
    prep.update(kernel=kernel,config=config,submitted=False,refreshed_from=str(source),
                bootstrap_sha256=sha(root/'job/train.py'),physics_pins=pins)
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return {'root':str(root),'kernel':kernel,'submitted':False}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('source','output','slug'):p.add_argument('--'+arg,required=True)
    p.add_argument('--data');a=p.parse_args()
    print(json.dumps(prepare(a.source,a.output,a.slug,a.data)))
