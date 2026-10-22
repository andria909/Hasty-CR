"""Full-catalog heldout evaluation of an explicitly selected shared policy.

Never trains, selects weights, adjusts reward/window coefficients or claims
full-match/live-game acceptance. Protocol and checkpoint must be SHA-pinned.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from sim.eval_defense import evaluate_episode, aggregate
from sim.strategy_env import StrategyEnv, OBSERVATION_CONTRACT
from sim.train_attack_ppo import (build_attack_network, load_attack_state,
                                   summarize_attack, code_fingerprint)
from sim.train_ppo import checkpoint_architecture


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context_schedule(protocol,split,scenarios):
    if protocol.get('schema')!='joint-strategy-evaluation-v1'or split not in ('validation','test'):
        raise ValueError('Explicit heldout strategy protocol required')
    configured=protocol['scenario_ids'][split]
    actual=sorted(s.scenario_id for s in scenarios)
    if sorted(configured)!=actual or len(set(configured))!=len(configured):
        raise ValueError('Protocol must cover the exact heldout catalog once')
    seeds=protocol['seeds'][split]
    if not seeds or len(set(seeds))!=len(seeds)or any(isinstance(n,bool)or not isinstance(n,int)for n in seeds):
        raise ValueError('Distinct integer shuffle seeds required')
    return [dict(scenario_id=name,seed=seed)for name in sorted(configured)for seed in seeds]


def evaluate(network,device,protocol,split,*,env_factory=StrategyEnv):
    env=env_factory(split=split,**protocol['environment'])
    schedule=context_schedule(protocol,split,env.scenarios)
    records=[]
    network.eval()
    for context in schedule:
        record=evaluate_episode(env,network,device,**context)
        if record.get('strategy_scenario_id')!=context['scenario_id']or record.get('split')!=split:
            raise ValueError('Environment returned a different task or split')
        records.append(record)
    attack=[r for r in records if r.get('task')=='attack']
    defense=[r for r in records if r.get('task')=='defense']
    if len(attack)+len(defense)!=len(schedule)or not attack or not defense:
        raise ValueError('Both evaluated tasks required')
    groups={}
    for field in ('difficulty','lane','shuffle_seed','family_id','setup_kind','goal_kind','timer_visible'):
        groups[field]={str(value):summarize_attack([r for r in attack if r.get(field)==value])
                       for value in sorted({r.get(field)for r in attack},key=str)}
    ids=[s.scenario.scenario_id for s in env.scenarios if s.task=='defense']
    return dict(schema='joint-strategy-evaluation-result-v1',split=split,
        expected_episodes=len(schedule),episodes=len(records),records=records,
        attack=dict(overall=summarize_attack(attack),by=groups),
        defense=aggregate(defense,expected_scenarios=ids,expected_lanes=('left','right')),
        quality_accepted=False,full_match_evaluated=False,
        checkpoint_selection_performed=False,mechanics_parity_verified=False)


def load_selected(path,device,environment):
    blob=torch.load(path,map_location='cpu',weights_only=False)
    config=blob.get('config',{})
    if config.get('task_mode')!='joint'or config.get('num_scalars')!=309:
        raise ValueError('Selected checkpoint is not one shared joint policy')
    for key,value in environment.items():
        if config.get(key)!=value:raise ValueError('Evaluation environment differs from training: '+key)
    from sim.eval_defense import environment_contract
    environment_contract(config)
    if blob.get('schema')=='attack-ppo-boundary-v1':
        expected=code_fingerprint('joint');actual=config.get('source_code_sha256')
    elif blob.get('resume_schema')=='attack-grpo-boundary-v1':
        from sim.train_attack_grpo import attack_source_hash
        expected=attack_source_hash('joint');actual=config.get('environment_code_sha256')
    else:raise ValueError('Unknown selected joint checkpoint schema')
    if expected!=actual:raise ValueError('Evaluation/training source semantics differ')
    network=build_attack_network(checkpoint_architecture(blob),device,num_scalars=309)
    load_attack_state(network,blob,exact=True,observation_contract=OBSERVATION_CONTRACT)
    if not all(bool(torch.isfinite(t).all())for t in network.state_dict().values()):
        raise ValueError('Nonfinite selected policy')
    return network.eval().requires_grad_(False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--split',choices=('validation','test'),required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():raise ValueError('Use a new evaluation output')
    if sha(args.checkpoint)!=args.checkpoint_sha256 or sha(args.protocol)!=args.protocol_sha256:
        raise ValueError('Pinned evaluation input SHA mismatch')
    if args.device.startswith('cuda')and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    protocol=json.loads(args.protocol.read_text())
    network=load_selected(args.checkpoint,args.device,protocol['environment'])
    report=evaluate(network,args.device,protocol,args.split)
    report.update(checkpoint_sha256=args.checkpoint_sha256,protocol_sha256=args.protocol_sha256)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x')as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(split=args.split,episodes=report['episodes'],quality_accepted=False)))


if __name__=='__main__':main()
