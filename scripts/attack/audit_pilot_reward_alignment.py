"""Receipt-verified return accounting and same-context reward comparisons.

Undiscounted returns do not reconstruct PPO advantages or discounted rewards.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from scripts.attack.compare_ppo_pilot_behavior import index_records
from scripts.attack.select_pilot_pool import select
from scripts.kaggle.local_artifact_transfer import file_sha

WRAPPER_WEIGHTS=('tower_support_weight','spell_waste_weight','crown_only_weight','dormant_king_weight')
DIRECTIONS={'own_fraction_loss':-1,'enemy_fraction_damage':1,'empty_spell_elixir':-1,'elixir_overflow':-1}


def components(row,config,*,rescore=False):
    if row['split'] not in ('train','validation') or row['outcome'] not in ('win','loss','draw'):
        raise ValueError('Known completed train/validation outcome required')
    p=config['continuous_profile'];king=row['dormant_at_request_king_only_elixir'];crown=row['nonlethal_crown_only_elixir']
    if not 0<=king<=crown:raise ValueError('Disjoint paid spell bins required')
    if not rescore and any(row[k]!=config[k]for k in WRAPPER_WEIGHTS):raise ValueError('Recorded reward coefficients differ')
    terms=dict(enemy_damage=p['enemy_hp_reward']*row['enemy_fraction_damage'],
        own_damage=-p['own_hp_penalty']*row['own_fraction_loss'],
        overflow=-p['overflow_penalty']*row['elixir_overflow'],illegal=-.1*row['illegal_actions'],
        missed_fireball=-p['missed_fireball_penalty']*row['missed_fireball_elixir'],
        board=row['board_shaping_return'],reserve=row['reserve_shaping_return'],
        fireball=p['fireball_value_reward']*row['fireball_exchange_value'],
        terminal=p['win_reward']if row['outcome']=='win'else -p['loss_penalty']if row['outcome']=='loss'else 0.,
        tower=config['tower_support_weight']*row['tower_support_paid_value'],
        empty_spell=-config['spell_waste_weight']*row['empty_spell_elixir'],
        crown_spell=-config['crown_only_weight']*(crown-king)-config['dormant_king_weight']*king)
    if not all(math.isfinite(v)for v in terms.values()):raise ValueError('Nonfinite reward')
    if not rescore and not math.isclose(sum(terms.values()),row['episode_return'],abs_tol=1e-5):
        raise ValueError('Return does not match all paid reward components')
    return terms


def paired(before,after,config):
    before=index_records(before);after=index_records(after)
    if before.keys()!=after.keys():raise ValueError('Different contexts')
    inversions=[];tradeoffs=[];worse=[];changed=0
    score={'loss':0.,'draw':.5,'win':1.}
    for key,a in after.items():
        b=before[key]
        # Require baseline accounting under its recorded coefficients with the
        # same base profile before changing ONLY its wrapper coefficients.
        original=dict(config,**{k:b[k]for k in WRAPPER_WEIGHTS})
        components(b,original);ta=components(a,config);tb=components(b,config,rescore=True)
        delta={k:ta[k]-tb[k]for k in ta};dr=sum(delta.values());ds=score[a['outcome']]-score[b['outcome']]
        if ds:changed+=1
        detail=dict(context=list(key),before_outcome=b['outcome'],after_outcome=a['outcome'],
            before_elapsed_ms=b['elapsed_ms'],after_elapsed_ms=a['elapsed_ms'],return_delta=dr,component_deltas=delta,
            before_metrics={k:b[k]for k in DIRECTIONS},after_metrics={k:a[k]for k in DIRECTIONS})
        if ds*dr < -1e-9:inversions.append(detail)
        if ds==0 and (a['own_fraction_loss']-b['own_fraction_loss'])*dr>1e-9:tradeoffs.append(detail)
        if ds==0 and abs(dr)>1e-9:
            high,low=(a,b)if dr>0 else(b,a)
            differences=[DIRECTIONS[k]*(high[k]-low[k])for k in DIRECTIONS]
            if all(d<=1e-9 for d in differences)and any(d < -1e-9 for d in differences):
                detail['higher_return_policy']='after'if dr>0 else'before'
                detail['same_duration']=a['elapsed_ms']==b['elapsed_ms']
                worse.append(detail)
    return dict(comparisons=len(after),changed_outcomes=changed,outcome_inversions=inversions,
        same_outcome_hp_tradeoff_count=len(tradeoffs),higher_return_worse_on_four_metrics=worse)


def audit(selection_path):
    path=Path(selection_path);saved=json.loads(path.read_text())
    if select([s['prepared']for s in saved['studies']],limit=4)!=saved:raise ValueError('Prior verified selection changed')
    sources={str(path):file_sha(path)};runs=[];comparisons=[];checked=0;max_error=0.
    for study in saved['studies']:
        root=Path(study['prepared']);collected=root/'collected';protocol=json.loads((root/'dataset/protocol.json').read_text())
        receipt_path=collected/'collection_receipt.json';receipt=json.loads(receipt_path.read_text())
        if not receipt['complete']or receipt['remote_status']!='COMPLETE'or receipt['kernel']!=study['kernel']:
            raise ValueError('Matching terminal receipt required')
        sources[str(receipt_path)]=file_sha(receipt_path)
        def read(name,lines=False):
            target=collected/name;digest=file_sha(target)
            if digest!=receipt['files'][name]['sha256']:raise ValueError('Artifact changed: '+name)
            sources[str(target)]=digest
            return [json.loads(x)for x in target.read_text().splitlines()]if lines else json.loads(target.read_text())
        variant=protocol['study_variant']
        if variant not in ('policy_adaptation_v1','calibrated_spell_cost_v1','group_entropy_v1'):
            raise ValueError('Audit scoped to three completed short cohorts')
        for case in protocol['cases']:
            for seed in protocol['pilot_seeds']:
                name=f"pilot_{case['id']}_s{seed}";c=read(name+'/config.json');metrics=read(name+'/metrics.jsonl',True)
                if [m['update']for m in metrics]!=list(range(1,101)):raise ValueError('All100 updates required')
                actor=[]
                for m in metrics:
                    for r in m['episodes']:
                        terms=components(r,c);checked+=1;max_error=max(max_error,abs(sum(terms.values())-r['episode_return']))
                        if not m['optimizer_controls']['value_only_warmup']:actor.append((r,terms))
                means={k:statistics.mean(t[k]for _,t in actor)for k in actor[0][1]}
                runs.append(dict(kernel=study['kernel'],id=name,actor_episode_completions=len(actor),
                    spell_waste_weight=c['spell_waste_weight'],mean_return=statistics.mean(r['episode_return']for r,_ in actor),
                    mean_components=means,overflow_affected=sum(r['elixir_overflow']>0 for r,_ in actor),
                    gradient_norm_last_batch=dict(median=statistics.median(m['gradient_norm']for m in metrics if not m['optimizer_controls']['value_only_warmup']),
                        configured_clip=c['max_grad_norm']),
                    limitations='Episode completions during actor phase may have begun during critic warmup; not a checkpoint win rate'))
                after=read(name+'_summary.json')['validation']['records']
                for r in after:components(r,c)
                if variant=='policy_adaptation_v1':reference=case['initial_id']+'_reference.json'
                elif case['id']=='control':continue
                else:reference=f'pilot_control_s{seed}_summary.json'
                result=paired(read(reference)['validation']['records'],after,c)
                comparisons.append(dict(kernel=study['kernel'],id=name,reference=reference,**result))
    if len(runs)!=24:raise ValueError('All24 pilots required')
    return dict(schema='pilot-reward-alignment-v1',runs=runs,paired_validation=comparisons,
        train_returns_verified=checked,max_return_error=max_error,
        paired_comparisons=sum(r['comparisons']for r in comparisons),
        changed_outcomes=sum(r['changed_outcomes']for r in comparisons),
        outcome_inversions=sum(len(r['outcome_inversions'])for r in comparisons),
        higher_return_worse_on_four_metrics=sum(len(r['higher_return_worse_on_four_metrics'])for r in comparisons),
        sources=sources,source_sha256=file_sha(__file__),remote_queries=False,training_changed=False,live_games=False,
        limitations=['Undiscounted episode sums, not PPO advantages or discounted temporal credit',
            'Each comparison rescores baseline wrapper coefficients under the candidate recipe; base return accounting must match',
            'Same outcome/less HP is not the whole objective; survival duration and useful troop damage can differ',
            'Repeated development contexts; flags are diagnostics, not independent statistical evidence or reward prescriptions'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.selection)
    with Path(a.output).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k]for k in ('train_returns_verified','max_return_error','paired_comparisons','changed_outcomes','outcome_inversions','higher_return_worse_on_four_metrics')}))
