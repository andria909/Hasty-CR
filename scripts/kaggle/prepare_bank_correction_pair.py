"""Freeze both sides of the curriculum comparison into one private GPU package."""
import argparse
import ast
import io
import json
from pathlib import Path
import shutil
import tarfile

from scripts.attack.train_bank_correction_pair import protocol,depth_protocol
from scripts.kaggle.local_artifact_transfer import file_sha
from scripts.kaggle.prepare_bank_initialization import prepare as prepare_base


def prepare(source,base,corrections,initial,output,slug,technical_identical=False,owner=None,fit_depth=False):
    if fit_depth and technical_identical:raise ValueError('Fitting diagnosis requires distinct curricula')
    p=depth_protocol(base,corrections,initial)if fit_depth else protocol(base,corrections,initial,technical_identical)
    prep=prepare_base(source,base,output,slug,owner,initial)
    root=Path(output);c=prep['config'];archive=root/'dataset/extensions.tar.gz'
    with tarfile.open(archive)as stream:
        payload={x.name:stream.extractfile(x).read()for x in stream.getmembers()if x.isfile()}
    name='scripts/attack/train_bank_correction_pair.py'
    payload[name]=Path(name).read_bytes();c['extensions'][name]=file_sha(name)
    with tarfile.open(archive,'w:gz')as stream:
        for name,raw in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(raw);stream.addfile(info,io.BytesIO(raw))
    c['pins']['extensions.tar.gz']=file_sha(archive)
    for src,dst in (('examples.npz','correction_examples.npz'),('manifest.json','correction_manifest.json')):
        shutil.copyfile(Path(corrections)/src,root/'dataset'/dst);c['pins'][dst]=file_sha(root/'dataset'/dst)
    (root/'dataset/initialization_protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    c['pins']['initialization_protocol.json']=file_sha(root/'dataset/initialization_protocol.json')
    tree=ast.parse((root/'job/train.py').read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    start=next(i for i,n in enumerate(execute.body)if isinstance(n,ast.Assign)and
        isinstance(n.targets[0],ast.Name)and n.targets[0].id=='data')
    execute.body=execute.body[:start]+ast.parse("""
base=project/'control_data'
corrections=project/'correction_data'
for target,prefix in ((base,'skill'),(corrections,'correction')):
    target.mkdir()
    shutil.copyfile(inputs/(prefix+'_examples.npz'),target/'examples.npz')
    shutil.copyfile(inputs/(prefix+'_manifest.json'),target/'manifest.json')
subprocess.run([sys.executable,'-u','-m','scripts.attack.train_bank_correction_pair',
    '--base',str(base),'--corrections',str(corrections),'--initial',str(inputs/'parent.pt'),
    '--protocol',str(inputs/'initialization_protocol.json'),'--output','/kaggle/working/bank_correction_pair'],
    cwd=project,check=True)
""").body
    tree.body[-1].value.args=[ast.parse(repr(c),mode='eval').body];ast.fix_missing_locations(tree)
    if fit_depth:
        for node in ast.walk(execute):
            if isinstance(node,ast.Constant)and node.value=='/kaggle/working/bank_correction_pair':node.value='/kaggle/working/bank_fit_depth'
    (root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    prep.update(study_variant='bank_fit_depth_v1'if fit_depth else'bank_correction_pair_v1',config=c,
        bootstrap_sha256=file_sha(root/'job/train.py'),technical_preflight_only=technical_identical)
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n');return prep


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','base','corrections','initial','output','slug'):p.add_argument('--'+key,required=True)
    p.add_argument('--owner');p.add_argument('--technical-identical',action='store_true');p.add_argument('--fit-depth',action='store_true');a=p.parse_args()
    r=prepare(a.source,a.base,a.corrections,a.initial,a.output,a.slug,a.technical_identical,a.owner,a.fit_depth)
    print(json.dumps(dict(kernel=r['kernel'],submitted=False,technical_preflight_only=r['technical_preflight_only'])))
