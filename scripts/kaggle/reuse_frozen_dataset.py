"""Reuse owned, verified Kaggle inputs; inline only small changed input files."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import shutil

from scripts.kaggle.queue_prepared_pilot import rebind
from scripts.kaggle.local_artifact_transfer import file_sha


def materialize_reused_inputs(source,output,config):
    source=Path(source);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    for name,digest in config['pins'].items():
        if Path(name).name!=name:raise ValueError('Flat input names required')
        target=output/name
        if name in config['inline_inputs']:
            content=base64.b64decode(config['inline_inputs'][name],validate=True)
            if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Inline input digest differs')
            target.write_bytes(content)
        else:
            original=source/name
            if original.is_symlink()or hashlib.sha256(original.read_bytes()).hexdigest()!=digest:
                raise ValueError('Reused input digest differs: '+name)
            shutil.copyfile(original,target)
    return output


def prepare(source,output,reuse):
    source,root,reuse=Path(source),Path(output),Path(reuse)
    p=json.loads((source/'preparation.json').read_text());old=json.loads((reuse/'preparation.json').read_text())
    receipt=json.loads((reuse/'collected/collection_receipt.json').read_text())
    if (not receipt['complete']or receipt['remote_status']!='COMPLETE'or receipt['kernel']!=old['kernel']or
            p['config']['owner']!=old['config']['owner']):
        raise ValueError('Completed verified predecessor on the same owned account required')
    if (source/'kernel_push_requested.json').exists():raise ValueError('Do not repackage a submitted kernel')
    differences={k for k,v in p['config']['pins'].items()if old['config']['pins'].get(k)!=v}
    if differences!={'protocol.json','extensions.tar.gz'}:raise ValueError('Only protocol and extensions may be inlined')
    for name,digest in old['config']['pins'].items():
        if file_sha(reuse/'dataset'/name)!=digest:raise ValueError('Predecessor local input changed')
    if file_sha(source/'dataset/source.tar.gz')!=file_sha(reuse/'dataset/source.tar.gz'):
        raise ValueError('Base source archive differs')
    prep=rebind(source,root,p['config']['owner']);c=prep['config']
    c['source_slug']=old['config']['slug']
    if c['source_slug']==c['slug']:raise ValueError('Distinct staging and predecessor dataset required')
    c['inline_inputs']={name:base64.b64encode((root/'dataset'/name).read_bytes()).decode()for name in sorted(differences)}
    if sum(len(v)for v in c['inline_inputs'].values())>2_000_000:raise ValueError('Inline transport too large')
    tree=ast.parse((root/'job/train.py').read_text());execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run');before=ast.dump(run)
    mount=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='mount')
    mount.body.insert(0,ast.parse("if slug == config['slug']:\n    return materialize_reused_inputs(mount(config['source_slug']), Path('/kaggle/working/focus_inline_inputs'), config)").body[0])
    helper=next(n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='materialize_reused_inputs')
    tree.body.insert(0,ast.Import(names=[ast.alias(name='base64')]))
    tree.body.insert(1,helper)
    tree.body[-1].value.args=[ast.parse(repr(c),mode='eval').body];ast.fix_missing_locations(tree)
    (root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    if ast.dump(run)!=before:raise ValueError('Training algorithm changed')
    metadata=json.loads((root/'job/kernel-metadata.json').read_text())
    reuse_id=c['owner']+'/'+c['source_slug'];metadata['dataset_sources']=[reuse_id]
    (root/'job/kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    prep.update(reused_dataset=reuse_id,reused_prepared=str(reuse),reused_preparation_sha256=file_sha(reuse/'preparation.json'),
        bootstrap_sha256=file_sha(root/'job/train.py'),training_run_ast_unchanged=True,
        inline_input_names=sorted(differences),new_dataset_upload_required=False)
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    staged=materialize_reused_inputs(reuse/'dataset',root/'actual_transport_inputs',c)
    if any(file_sha(staged/name)!=digest for name,digest in c['pins'].items()):raise ValueError('Transport did not preserve all inputs')
    (root/'transport_preflight.json').write_text(json.dumps(dict(all_input_pins_verified=True,training_run_ast_unchanged=True,
        reused_dataset=reuse_id,bootstrap_sha256=prep['bootstrap_sha256'],preparation_sha256=file_sha(root/'preparation.json')),
        indent=2)+'\n')
    return dict(kernel=prep['kernel'],reused_dataset=reuse_id,inline_bytes=sum(len(v)for v in c['inline_inputs'].values()),submitted=False)


def validate_reuse(root,prep):
    root=Path(root);c=prep['config'];proof=json.loads((root/'transport_preflight.json').read_text())
    if (not proof.get('all_input_pins_verified')or not proof.get('training_run_ast_unchanged')or
            proof['bootstrap_sha256']!=file_sha(root/'job/train.py')or
            proof['preparation_sha256']!=file_sha(root/'preparation.json')or
            prep['reused_dataset']!=c['owner']+'/'+c['source_slug']or
            set(c['inline_inputs'])!={'protocol.json','extensions.tar.gz'}):
        raise ValueError('Exact owned-dataset reuse proof required')
    metadata=json.loads((root/'job/kernel-metadata.json').read_text())
    if metadata['dataset_sources']!=[prep['reused_dataset']]:raise ValueError('Different reused mount')
    for name,data in c['inline_inputs'].items():
        if hashlib.sha256(base64.b64decode(data,validate=True)).hexdigest()!=c['pins'][name]:
            raise ValueError('Embedded input changed')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','output','reuse'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.reuse)))
