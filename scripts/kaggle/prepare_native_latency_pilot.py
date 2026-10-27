"""Freeze four paired latency profiles; common measured-delay validation.

Old full GPU bundles are immutable. This prepares a separate pilot only.
"""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path

from scripts.kaggle.prepare_air_reserve_pilot import prepare as prepare_reserve
from scripts.kaggle.run_five_round_robin import sha


def latency_cases(base):
    result=[]
    for name,delay,replay in [('delay0225',225,0.),('delay1350',1350,0.),
                              ('delay2025',2025,0.),('delay1350_replay100',1350,1.)]:
        trial=deepcopy(base)
        trial.update(id=name,warmstart_updates=800,supervised_replay_weight=replay,
                     action_delay_ms=delay,validation_action_delay_ms=1350)
        result.append(trial)
    return result


def prepare(output,owner='vertel0'):
    root=Path(output);prepare_reserve(root,owner)
    prep=json.loads((root/'preparation.json').read_text());config=prep['config']
    slug='hastycr-native-latency-pilot-inputs-20261027'
    kernel=owner+'/hastycr-native-latency-pilot-20261027'
    config.update(slug=slug,source_slug=slug,evaluation_action_delay_ms=1350)
    path=root/'dataset/protocol.json';protocol=json.loads(path.read_text())
    base=next(t for t in protocol['cases'] if t['id']=='joint_bc800')
    protocol.update(study_variant='native_latency_calibration_v1',pilot_seeds=[192,193],
        cases=latency_cases(base),rationale_split='live_development_feedback',
        rationale_source='reports/defense_plan/reserve_bc800_body_timing_20261027.json',
        paired_comparison='Fixed balanced initializer and reserve data. TRAIN delays225/1350/2025; fourth1350 adds replay1. Every candidate and parent validation uses1350.',
        comparison_scope='Separate measured-latency cohort; scores cannot be pooled with prior225ms validation rankings',
        limitations='Body/deployment timing is sampled and deployment field remains a candidate; fixed delays are an ablation, not a validated complete latency distribution')
    path.write_text(json.dumps(protocol,indent=2)+'\n');config['pins']['protocol.json']=sha(path)
    tree=ast.parse((root/'job/train.py').read_text())
    tree.body[-1].value.args=[ast.parse(repr(config),mode='eval').body]
    ast.fix_missing_locations(tree);(root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    for name,identity in [('dataset/dataset-metadata.json',owner+'/'+slug),('job/kernel-metadata.json',kernel)]:
        path=root/name;row=json.loads(path.read_text());row.update(id=identity,title=identity.split('/')[-1])
        if name.startswith('job/'):row['dataset_sources']=[owner+'/'+slug]
        path.write_text(json.dumps(row,indent=2)+'\n')
    prep.update(kernel=kernel,config=config,bootstrap_sha256=sha(root/'job/train.py'),
                study_variant=protocol['study_variant'],paired_previous_study=None,
                comparison_scope=protocol['comparison_scope'])
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=kernel,pilot_steps=819200,submitted=False,evaluation_action_delay_ms=1350)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--owner',default='vertel0')
    a=p.parse_args();print(json.dumps(prepare(a.output,a.owner)))
