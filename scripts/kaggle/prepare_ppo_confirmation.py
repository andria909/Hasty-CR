"""Prepare, without submitting, six second-seed trials from verified PPO screening."""
import argparse
import ast
import json
from pathlib import Path

from scripts.defense.select_ppo_coefficients import select,sha

OWNERS=('andriashadrin','vertel0','andreiperelaiko','demjanklimov')


def prepare(study,output):
    study=Path(study);shared=study/'shared';protocol_path=Path('data/ppo_tuning/protocol.json')
    protocol=json.loads(protocol_path.read_text())
    selected=select([study/owner/'collected' for owner in OWNERS],protocol_path,
                    Path('data/ppo_tuning/validation_contexts.json'),shared/'manifest.json',Path('data/ppo_tuning/profiles'),'screen')
    if protocol['confirm_top_profiles_per_mode']!=3:raise ValueError('Expected three finalists per mode')
    output=Path(output)
    if output.exists():raise ValueError('Use a new confirmation package path')
    # Read the submitted standalone driver rather than changing the running source bundle.
    original=(study/OWNERS[0]/'job/study.py').read_text()
    assignments=[n for n in ast.parse(original).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LAUNCH_CONFIG' for t in n.targets)]
    if len(assignments)!=1:raise ValueError('Ambiguous standalone configuration')
    assignment=assignments[0];lines=original.splitlines(keepends=True)
    jobs=[]
    for index,owner in enumerate(OWNERS[:3]):
        config=json.loads((study/owner/'launch_config.json').read_text())
        for name,digest in config['pins'].items():
            if sha(shared/name)!=digest:raise ValueError('Input pins differ from the screening study')
        config['workers']=[[dict(mode=mode,profile=selected['modes'][mode]['confirmation_profiles'][index],seed=protocol['confirmation_training_seed'])] for mode in protocol['modes']]
        code=''.join(lines[:assignment.lineno-1])+('LAUNCH_CONFIG='+repr(config)+'\n')+''.join(lines[assignment.end_lineno:])
        compile(code,'study.py','exec')
        kernel=owner+'/hastycr-ppo-coef-confirm-20261023'
        metadata=dict(id=kernel,title=kernel.split('/')[1].replace('-',' '),code_file='study.py',language='python',kernel_type='script',is_private=True,
                      enable_gpu=True,enable_internet=False,dataset_sources=[config['dataset']],competition_sources=[],kernel_sources=[])
        jobs.append((owner,config,code,metadata))
    output.mkdir(parents=True,exist_ok=False)
    (output/'screen_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    for owner,config,code,metadata in jobs:
        folder=output/owner;job=folder/'job';job.mkdir(parents=True)
        (folder/'launch_config.json').write_text(json.dumps(config,indent=2)+'\n')
        (job/'study.py').write_text(code);(job/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    report=dict(schema='ppo-confirmation-preparation-v1',selection_sha256=sha(output/'screen_selection.json'),seed=protocol['confirmation_training_seed'],
                training_steps_per_trial=102400,trials=6,source_sha256=selected['source_sha256'],
                jobs=[dict(owner=owner,kernel=metadata['id'],code_sha256=sha(output/owner/'job/study.py'),workers=config['workers']) for owner,config,code,metadata in jobs],
                submitted=False,test_used_for_selection=False)
    (output/'preparation_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(prepare(args.study,args.output),indent=2))


if __name__=='__main__':main()
