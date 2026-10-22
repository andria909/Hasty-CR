"""Read-only, paired defense evaluation; never selects training weights on test."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sim.train_ppo import (build_network, checkpoint_architecture,
                           load_policy_state, masked_distribution)

TOWERS = ('left','right','king')


def describe(values):
    values = np.asarray([v for v in values if v is not None],dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0,mean=None,median=None,p90=None,p95=None,max=None)
    return dict(count=len(values),mean=float(values.mean()),median=float(np.median(values)),
                p90=float(np.quantile(values,.9)),p95=float(np.quantile(values,.95)),max=float(values.max()))


def summarize(records):
    def rate(key):
        known=[r[key] for r in records if isinstance(r.get(key),bool)]
        return dict(count=len(known),rate=sum(known)/len(known) if known else None)
    result = dict(episodes=len(records),no_damage=rate('no_tower_damage'),
        timeout_rate=sum(r.get('terminal_reason')=='timeout' for r in records)/len(records) if records else None,
        invalid_script_rate=sum(str(r.get('terminal_reason','')).startswith('invalid_script')
                                for r in records)/len(records) if records else None,
        truncated_rate=sum(bool(r.get('truncated',False)) for r in records)/len(records) if records else None,
        terminal_reasons=dict(Counter(r.get('terminal_reason','unknown') for r in records)))
    for field in ('own_tower_hp_loss','own_tower_fraction_loss','enemy_tower_hp_damage'):
        result[field]={tower:describe([r.get(field,{}).get(tower) for r in records]) for tower in TOWERS}
        result[field]['total']=describe([sum(r[field].values()) for r in records
                                        if all(t in r.get(field,{}) for t in TOWERS)])
    for field in ('elixir_spent','spell_quality','spell_misses','illegal_actions','elapsed_ms'):
        result[field]=describe([r.get(field) for r in records])
    result['tower_survival']={tower:dict(
        count=sum(isinstance(r.get('own_towers_alive',{}).get(tower),bool) for r in records),
        rate=np.mean([r['own_towers_alive'][tower] for r in records
                      if isinstance(r.get('own_towers_alive',{}).get(tower),bool)]).item()
             if any(isinstance(r.get('own_towers_alive',{}).get(tower),bool) for r in records) else None)
        for tower in TOWERS}
    return result


def aggregate(records, expected_scenarios=None, expected_lanes=None):
    grouped={}
    for field in ('difficulty','family_id','lane','shuffle_seed','scenario_id'):
        buckets=defaultdict(list)
        for record in records:
            buckets[str(record.get(field,'unknown'))].append(record)
        grouped[field]={key:summarize(values) for key,values in sorted(buckets.items())}
    seen={r.get('scenario_id') for r in records if r.get('scenario_id') is not None}
    lanes={r.get('lane') for r in records if r.get('lane') is not None}
    expected=set(expected_scenarios) if expected_scenarios is not None else None
    expected_lane_set=set(expected_lanes) if expected_lanes is not None else None
    required=('terminal_reason','own_tower_hp_loss','own_tower_fraction_loss','own_towers_alive',
              'enemy_tower_hp_damage','elixir_spent','spell_quality','spell_misses',
              'illegal_actions','scenario_id','family_id','difficulty','split','lane','no_tower_damage')
    coverage=dict(
        metrics_complete=bool(records) and all(all(k in r for k in required) for r in records),
        tower_metrics_complete=bool(records) and all(
            all(all(t in r.get(k,{}) for t in TOWERS) for k in
                ('own_tower_hp_loss','own_tower_fraction_loss','own_towers_alive','enemy_tower_hp_damage'))
            for r in records),
        scenario_catalog_known=expected is not None,
        all_expected_scenarios_seen=expected <= seen if expected is not None else None,
        missing_scenarios=sorted(expected-seen) if expected is not None else None,
        expected_lanes_known=expected_lane_set is not None,
        all_expected_lanes_seen=expected_lane_set <= lanes if expected_lane_set is not None else None,
        distinct_scenarios=len(seen),distinct_families=len({r.get('family_id') for r in records}),
        distinct_shuffle_seeds=len({r.get('shuffle_seed') for r in records}),
        splits=sorted({str(r.get('split','unknown')) for r in records}),
        environment_flags=sorted({flag for r in records for flag in r.get('coverage_flags',[])}),
        missing_metric_counts={key:sum(key not in r for r in records) for key in required},
        snapshot_parity_verified=False,mechanics_parity_verified=False,
        live_game_quality_verified=False,statistical_independence_guaranteed=False)
    # A direct spell may make zero damage impossible. Missing labels must never pass.
    simple=[r for r in records if r.get('difficulty')=='simple']
    eligible=[r for r in simple if r.get('unavoidable_tower_damage') is False]
    labeled=len(eligible)==len(simple)
    simple_pass=(all(r.get('no_tower_damage') is True for r in eligible)
                 if simple and labeled else None)
    gates=dict(simple_no_damage=dict(passed=simple_pass,eligible_episodes=len(eligible),
        observed_simple_episodes=len(simple),scope='observed labeled avoidable-damage scenarios only',
        complete_catalog_required=True),
        medium_hp_tolerance=dict(passed=None,status='TBD: user tolerance not specified'),
        hard_and_stress=dict(passed=None,status='report losses and survival; no approved numeric threshold'),
        previous_models_winrate=dict(passed=None,status='requires separate paired 1v1 evaluation'),
        overall=dict(passed=None,status='not an acceptance decision'))
    return dict(overall=summarize(records),by=grouped,coverage=coverage,gates=gates)


def evaluate_episode(env,network,device,seed,scenario_id=None):
    obs,info=env.reset(seed=seed,scenario_id=scenario_id)
    steps=0
    while True:
        planes=torch.as_tensor(obs['planes'],device=device)[None]
        scalars=torch.as_tensor(obs['scalars'],device=device)[None]
        mask=torch.as_tensor(np.asarray(info['action_mask']),device=device,dtype=torch.bool)[None]
        if not mask.any():
            raise ValueError('empty action mask')
        with torch.no_grad():
            logits,_=network(planes,scalars)
            if not torch.isfinite(logits[mask]).all():
                raise ValueError('non-finite legal logits')
            action=int(masked_distribution(logits,mask).probs.argmax(-1).item())
        obs,reward,terminated,truncated,info=env.step(action)
        steps+=1
        if terminated or truncated:
            metrics=dict(info['metrics'])
            metrics.setdefault('shuffle_seed',seed)
            metrics.update(seed=seed,environment_steps=steps,
                           terminated=bool(terminated),truncated=bool(truncated))
            return metrics


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('validation','test'),required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--episodes',type=int,default=32)
    p.add_argument('--seeds',help='comma-separated paired shuffle seeds; overrides episodes')
    p.add_argument('--all-scenarios',action='store_true')
    p.add_argument('--action-delay-ms',type=int,default=150)
    p.add_argument('--max-ms',type=int,default=180000)
    p.add_argument('--initial-elixir',type=float,default=10)
    args=p.parse_args(argv)
    if args.output.exists():
        p.error('output already exists; choose a new path')
    if args.episodes<1:
        p.error('episodes must be positive')
    from sim.defense_env import DefenseEnv
    from sim.env import NUM_PLANES,NUM_SCALARS,ACTIONS
    seeds=[int(s) for s in args.seeds.split(',')] if args.seeds else list(range(args.episodes))
    expected=None
    if args.all_scenarios:
        from sim.defense_scenarios import scenarios_for_split
        scenarios=list(scenarios_for_split(args.split))
        expected=[s['scenario_id'] if isinstance(s,dict) else s.scenario_id for s in scenarios]
        if not expected:
            raise ValueError('empty scenario catalog')
    device=torch.device(args.device)
    blob=torch.load(args.init,map_location=device,weights_only=False)
    network=build_network(NUM_PLANES,NUM_SCALARS,ACTIONS,checkpoint_architecture(blob)).to(device)
    load_policy_state(network,blob['state_dict'])
    network.eval().requires_grad_(False)
    env=DefenseEnv(split=args.split,action_delay_ms=args.action_delay_ms,max_ms=args.max_ms,
                   initial_elixir=args.initial_elixir)
    records=[]
    for scenario in expected if expected is not None else [None]:
        for seed in seeds:
            record=evaluate_episode(env,network,device,seed,scenario)
            if record.get('split')!=args.split:
                raise ValueError('environment returned a different evaluation split')
            records.append(record)
    report=aggregate(records,expected_scenarios=expected)
    report.update(checkpoint_sha256=hashlib.sha256(args.init.read_bytes()).hexdigest(),
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        records=records,policy_mode='argmax public observation with stored legal mask; side1')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create protects against a competing process creating the output.
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(report['overall'],indent=2))


if __name__=='__main__':
    main()
