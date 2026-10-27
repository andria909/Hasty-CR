"""Copy the frozen latency pilot, changing only Fireball demonstration labels."""
import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from scripts.kaggle.local_artifact_transfer import file_sha as sha


def prepare(original,data,output):
    original,data,output=map(Path,(original,data,output))
    prep=json.loads((original/'preparation.json').read_text());config=deepcopy(prep['config'])
    code=(original/'job/train.py').read_text()
    if sha(original/'job/train.py')!=prep['bootstrap_sha256']:raise ValueError('Original frozen bootstrap changed')
    for name,digest in config['pins'].items():
        if sha(original/'dataset'/name)!=digest:raise ValueError('Original frozen input changed')
    labels=json.loads((data/'manifest.json').read_text())
    if (labels.get('schema')!='joint-fireball-latency-filtered-labels-v1' or labels['split']!='train'
            or labels['data_sha256']!=sha(data/'examples.npz')
            or labels['source_data_sha256']!=config['pins']['air_examples.npz']
            or labels['label_timing_by_skill']['fireball_group']!=1350):
        raise ValueError('Data-only1350ms relabel of original pilot required')
    if config['allow_full_training'] or config['evaluation_action_delay_ms']!=1350 or len(config['models'])!=1:
        raise ValueError('Original fixed-initializer latency-only pilot required')
    old_protocol=json.loads((original/'dataset/protocol.json').read_text())
    if old_protocol['pilot_seeds']!=[192,193] or len(old_protocol['cases'])!=4:raise ValueError('Original paired profiles/seeds required')
    output.mkdir(parents=True,exist_ok=False);dataset=output/'dataset';dataset.mkdir();job=output/'job';job.mkdir()
    for name in set(config['pins'])|{'source.tar.gz','dataset-metadata.json'}:
        shutil.copyfile(original/'dataset'/name,dataset/name)
    for old,new in [('examples.npz','air_examples.npz'),('manifest.json','air_manifest.json')]:
        shutil.copyfile(data/old,dataset/new);config['pins'][new]=sha(dataset/new)
    slug='hastycr-fireball-latency-pilot-inputs-20261027';kernel=config['owner']+'/hastycr-fireball-latency-pilot-20261027'
    config.update(slug=slug,source_slug=slug)
    protocol=deepcopy(old_protocol)
    protocol.update(study_variant='fireball1350_labels_paired_v1',rationale_source='reports/defense_plan/fireball_label_latency_audit_20261027.json',
        paired_comparison='Same frozen code/initializer/profiles/seeds/validation as native_latency; only42Fireball success masks filtered810to603. Other168labels unchanged.',
        limitations='TRAIN labels use mixed timing; Fireball1350ms, other225/effective250ms. No validation or test labels reused.',
        paired_original_protocol_sha256=prep['config']['pins']['protocol.json'])
    (dataset/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');config['pins']['protocol.json']=sha(dataset/'protocol.json')
    tree=ast.parse(code);last=tree.body[-1]
    if ast.literal_eval(last.value.args[0])!=prep['config']:raise ValueError('Original controller config differs')
    last.value.args=[ast.parse(repr(config),mode='eval').body];ast.fix_missing_locations(tree)
    changed=ast.unparse(tree)+'\n'
    if [ast.dump(n)for n in ast.parse(code).body[:-1]]!=[ast.dump(n)for n in ast.parse(changed).body[:-1]]:
        raise ValueError('Controller algorithm changed')
    (job/'train.py').write_text(changed)
    dm=json.loads((dataset/'dataset-metadata.json').read_text());dm.update(id=config['owner']+'/'+slug,title=slug)
    (dataset/'dataset-metadata.json').write_text(json.dumps(dm,indent=2)+'\n')
    km=json.loads((original/'job/kernel-metadata.json').read_text());km.update(id=kernel,title=kernel.split('/')[-1],dataset_sources=[config['owner']+'/'+slug])
    (job/'kernel-metadata.json').write_text(json.dumps(km,indent=2)+'\n')
    result=dict(prep,kernel=kernel,config=config,bootstrap_sha256=sha(job/'train.py'),submitted=False,
        study_variant=protocol['study_variant'],paired_previous_study=str(original),controller_ast_unchanged=True,
        changed_training_inputs=['air_examples.npz','air_manifest.json'],original_preparation_sha256=sha(original/'preparation.json'))
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');return result


def preflight(prepared):
    from scripts.kaggle.preflight_global_full_pair import unpack
    prepared=Path(prepared).absolute();prep=json.loads((prepared/'preparation.json').read_text());config=prep['config']
    if sha(prepared/'job/train.py')!=prep['bootstrap_sha256']:raise ValueError('Prepared bootstrap differs')
    protocol=json.loads((prepared/'dataset/protocol.json').read_text());root=prepared/'actual_preflight';root.mkdir()
    results=[]
    for index,trial in enumerate(protocol['cases']):
        child=root/f'child_{index}';child.mkdir();project=child/'project';unpack(prepared/'dataset',config,project)
        spec=dict(root=str(child),inputs=str(prepared/'dataset'),trial=trial,config=config,
            initial_selection=dict(selected=config['models'][0]['id']),bootstrap=str(prepared/'job/train.py'))
        path=child/'spec.json';path.write_text(json.dumps(spec,indent=2)+'\n')
        with(child/'run.log').open('x')as log:
            subprocess.run([sys.executable,str(Path('scripts/kaggle/preflight_global_full_pair.py').absolute()),'--child',str(path)],
                cwd=project,env=dict(os.environ,PYTHONPATH=str(project)),stdout=log,stderr=subprocess.STDOUT,check=True)
        results.append(json.loads((child/'child_preflight.json').read_text()))
        print(json.dumps(dict(profile=trial['id'],cpu_resume_verified=True)),flush=True)
    report=dict(frozen_sources_verified=True,exact_resume_16_to_32=True,children=results,validation_games=48,
        actual_cuda=False,quality_accepted=False,bootstrap_sha256=prep['bootstrap_sha256'],preparation_sha256=sha(prepared/'preparation.json'))
    with(prepared/'preflight.json').open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('original','data','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--preflight',action='store_true');a=p.parse_args()
    result=prepare(a.original,a.data,a.output)
    if a.preflight:preflight(a.output)
    print(json.dumps(dict(kernel=result['kernel'],submitted=False,pilot_steps=819200)))
