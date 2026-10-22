"""Freeze the trainer's validation-best joint weights after target audit.

Never scores a candidate, reads test results or changes the collected inputs.
The provisional validation metric is retained, not called quality acceptance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from scripts.kaggle.audit_strategy_training import audit, exact_file


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(collected,launch_config,source_manifest,output):
    import torch
    root=Path(collected);output=Path(output)
    if output.exists():raise ValueError('Use a new selected-policy directory')
    training=audit(root,launch_config,source_manifest)
    algorithm=training['algorithm']
    terminal=torch.load(exact_file(root,training['checkpoint']),map_location='cpu',weights_only=False)
    receipt=json.loads(exact_file(root,'collection_receipt.json').read_text())
    pattern=(r'ppo/best_validation_(\d{6})\.pt'if algorithm=='ppo'else
             r'grpo/best_validation_u(\d{6})_s(\d{6})\.pt')
    names=[(tuple(map(int,m.groups())),name)for name in receipt['files']
           if (m:=re.fullmatch(pattern,name))]
    if not names:raise ValueError('No trainer validation-best checkpoint; do not substitute test selection')
    name=max(names)[1];path=exact_file(root,name)
    best=torch.load(path,map_location='cpu',weights_only=False)
    score=best.get('best_validation')
    if (isinstance(score,bool)or not isinstance(score,(int,float))or not math.isfinite(score)
            or score!=terminal.get('best_validation')):
        raise ValueError('Selected checkpoint does not match terminal validation-best score')
    if best.get('config')!=terminal.get('config'):
        raise ValueError('Validation-best and terminal training contracts differ')
    count=best.get('step'if algorithm=='ppo'else 'environment_steps')
    if isinstance(count,bool)or not isinstance(count,int)or not 0<count<=training['steps']:
        raise ValueError('Selected validation checkpoint has invalid training counters')
    state=best.get('state_dict',{})
    if (not state or state['scalar.0.weight'].shape[1]!=309 or
            not all(bool(torch.isfinite(value).all())for value in state.values())):
        raise ValueError('Selected joint policy shape/nonfinite weights')
    if algorithm=='grpo'and (best.get('checkpoint_phase')!='boundary'or not best.get('resume_supported')):
        raise ValueError('Validation checkpoint contains an incomplete GRPO group')
    report=dict(schema='joint-strategy-selection-v1',algorithm=algorithm,
        selected_checkpoint=name,selected_sha256=sha(path),selected_training_steps=count,
        selected_validation_score=score,selection_split='validation',
        selection_metric='equal task mean episode return; provisional reward scale',
        selection_source='trainer monotonic validation-best history, confirmed by terminal best score',
        test_used_for_selection=False,quality_accepted=False,
        terminal_training_audit=training,launch_config_sha256=sha(launch_config),
        source_manifest_sha256=sha(source_manifest))
    # All inputs and selection validate before any output is created.
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(path,output/'policy.pt')
    if sha(output/'policy.pt')!=report['selected_sha256']:raise ValueError('Copied policy SHA mismatch')
    with (output/'selection.json').open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collected',type=Path,required=True)
    parser.add_argument('--launch-config',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(select(args.collected,args.launch_config,args.source_manifest,args.output)))


if __name__=='__main__':main()
