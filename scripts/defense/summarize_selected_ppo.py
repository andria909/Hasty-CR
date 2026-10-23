"""Audit heldout full-PPO artifacts and recompute comparable quality metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.defense.pack_evaluation import verified_frozen_package
from scripts.defense.summarize_five_models import rate
from scripts.kaggle.run_selected_ppo_evaluation import verify_selection
from sim.defense_protocol import EvaluationProtocol,acceptance_gate,wilson_interval


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_split(records):
    defense=[r for r in records if r['task']=='defense'];attack=[r for r in records if r['task']=='attack']
    def defense_group(rows):
        losses=[];spent=[]
        for row in rows:
            if str(row.get('terminal_reason','')).startswith('invalid_script'):
                raise ValueError('Invalid scripted episode cannot enter comparable quality summary')
            values=[row.get('own_tower_hp_loss',{}).get(t)for t in ('left','right','king')]
            cost=row.get('elixir_spent')
            if any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<0 for v in values+[cost]):
                raise ValueError('Missing or invalid defense metrics')
            if not isinstance(row.get('no_tower_damage'),bool)or row['no_tower_damage']!=(sum(values)==0):raise ValueError('No-damage label contradicts tower HP loss')
            alive=row.get('own_towers_alive',{})
            if any(not isinstance(alive.get(t),bool)for t in ('left','right','king')):raise ValueError('Missing tower survival labels')
            losses.append(sum(values));spent.append(cost)
        return dict(episodes=len(rows),no_damage=rate(rows,'no_tower_damage'),
                    mean_tower_hp_loss=sum(losses)/len(rows)if rows else None,
                    mean_elixir_spent=sum(spent)/len(rows)if rows else None,
                    princess_survived=rate([dict(value=r['own_towers_alive'][r['lane']])for r in rows],'value'),
                    king_survived=rate([dict(value=r['own_towers_alive']['king'])for r in rows],'value'),
                    timeouts=sum(r.get('terminal_reason')=='timeout'for r in rows),
                    invalid_scripts=sum(str(r.get('terminal_reason','')).startswith('invalid_script')for r in rows))
    hog=[];spell=[]
    for row in attack:
        if row.get('goal_kind')=='hog_tower_damage':
            damage=row.get('hog_target_tower_damage_v2')
            if isinstance(damage,bool)or not isinstance(damage,(int,float))or not math.isfinite(damage)or damage<0:
                raise ValueError('Invalid attributed Hog damage')
            hog.append(dict(value=True if damage>0 else False if row.get('hog_damage_ledger_complete_v2')is True else None))
        elif row.get('goal_kind')=='spell_finish':
            value=row.get('successful_spell_finish_v2')
            if value is not None and not isinstance(value,bool):raise ValueError('Invalid spell attribution label')
            spell.append(dict(value=value))
    return dict(defense=dict(overall=defense_group(defense),by_difficulty={name:defense_group([r for r in defense if r.get('difficulty')==name])
                    for name in ('simple','medium','complex','impossible')}),
                attack=dict(episodes=len(attack),episode_success=rate(attack,'attack_success'),
                            hog_confirmed_hit=rate(hog,'value'),spell_confirmed_finish=rate(spell,'value'),
                            missed_window=rate(attack,'goal_window_missed')))


def verify(collected,prepared,evaluation_package):
    root=Path(collected);prepared=Path(prepared)
    members,manifest,_,_=verified_frozen_package(evaluation_package)
    preparation=json.loads((prepared/'preparation_receipt.json').read_text());config=preparation['config']
    if sha(Path(evaluation_package)/'manifest.json')!=config['manifest_sha256']:raise ValueError('Common evaluation manifest differs')
    selection_path=prepared/'selected_dataset/selection.json';policy=prepared/'selected_dataset/policy.pt'
    if sha(selection_path)!=config['selection_sha256']or sha(policy)!=config['policy_sha256']:raise ValueError('Prepared policy provenance differs')
    selection=json.loads(selection_path.read_text());mode=verify_selection(selection,config,manifest)
    receipt=json.loads((root/'collection_receipt.json').read_text())
    if receipt.get('remote_status')!='COMPLETE'or receipt.get('complete')is not True or receipt.get('kernel')!=preparation['kernel']:
        raise ValueError('Authoritative complete evaluation collection required')
    def checked(name):
        path=root/name
        if receipt.get('files',{}).get(name,{}).get('sha256')!=sha(path):raise ValueError('Changed collected evaluation artifact')
        return json.loads(path.read_text())
    result=checked('result.json')
    if (result.get('schema')!='selected-ppo-final-evaluation-v1'or result.get('status')!='completed'or
        result.get('selected_sha256')!=config['policy_sha256']or result.get('selection_sha256')!=config['selection_sha256']or
        result.get('source_sha256')!=manifest['archive_sha256']or result.get('training_steps')!=2048000 or
        result.get('training_source_sha256')!=selection['source_sha256']or
        result.get('mode')!=mode or result.get('profile')!=selection['profile']):
        raise ValueError('Completed evaluation identity differs')
    if result.get('evaluation_environment_migration')!=manifest.get('environment_migration'):
        raise ValueError('Evaluation environment migration differs')
    protocol=json.loads(members['data/strategy/evaluation_protocol.json']);splits={};records_by_split={}
    if hashlib.sha256(members['data/strategy/evaluation_protocol.json']).hexdigest()!=config['protocol_sha256']:
        raise ValueError('Heldout protocol differs')
    for split,total in (('validation',1056),('test',804)):
        report=checked(split+'.json');records=report['records']
        if result['parts'][split]['sha256']!=sha(root/(split+'.json')):raise ValueError('Internal report SHA differs')
        if report.get('checkpoint_sha256')!=config['policy_sha256']or report.get('protocol_sha256')!=config['protocol_sha256']:
            raise ValueError('Skill policy/protocol differs')
        input_mode='joint_309_cells_v2'if mode=='joint'else'physical_live_306_timer_task_ignored'
        if report.get('input_mode')!=input_mode or report.get('terminal_attribution_enabled')is not True:
            raise ValueError('Skill input/terminal-attribution contract differs')
        expected={(sid,seed)for sid in protocol['scenario_ids'][split]for seed in protocol['seeds'][split]}
        actual=[(r.get('strategy_scenario_id'),r.get('seed'))for r in records]
        if len(actual)!=total or len(expected)!=total or set(actual)!=expected or any(r.get('split')!=split for r in records):
            raise ValueError('Missing, duplicate or off-split skill episode')
        coordinate='physical_live_cells_v2'if mode=='joint'else'physical_live'
        if any(r.get('coordinate_mode')!=coordinate or r.get('geometry')!='legacy_v1'or
               r.get('task')not in ('attack','defense')or not r['strategy_scenario_id'].startswith(r['task']+'::')for r in records):
            raise ValueError('Skill environment/task contract differs')
        splits[split]=summarize_split(records);records_by_split[split]=records
    duels=checked('duels.json')
    if (result['parts']['duels']['sha256']!=sha(root/'duels.json')or duels.get('candidate_sha256')!=config['policy_sha256']):
        raise ValueError('Paired-game policy provenance differs')
    frozen=EvaluationProtocol.from_dict(json.loads(members['data/defense/evaluation_protocol.json']))
    matches=duels['head_to_head'];actual=[(r.get('baseline_id'),r.get('seed'),r.get('candidate_side'))for r in matches]
    expected={(b,s,seat)for b in frozen.baseline_ids for s in frozen.paired_seeds for seat in (1,-1)}
    if len(actual)!=300 or len(expected)!=300 or set(actual)!=expected:raise ValueError('Incomplete or duplicate opponent schedule')
    baseline_modes={r['id']:r['observation_mode']for r in json.loads(members['data/defense/baselines.json'])['checkpoints']}
    candidate_mode='joint_enemy_presence_v1'if mode=='joint'else'physical_live'
    if any(r.get('baseline_sha256')!=manifest['baseline_sha256'].get(r.get('baseline_id'))or
           r.get('baseline_mode')!=baseline_modes.get(r.get('baseline_id'))or r.get('candidate_mode')!=candidate_mode or
           r.get('result')not in ('win','loss','draw','unknown')for r in matches):raise ValueError('Opponent identity/outcome differs')
    opponents={}
    for baseline in frozen.baseline_ids:
        baseline_rows=[r for r in matches if r['baseline_id']==baseline]
        outcomes=[r['result']for r in baseline_rows];wins=outcomes.count('win')
        observed=[r.get('observed_result',r['result'])for r in baseline_rows]
        if any(value not in ('win','loss','draw','unknown')for value in observed):raise ValueError('Invalid observed simulator outcome')
        opponents[baseline]=dict(games=20,wins=wins,losses=outcomes.count('loss'),draws=outcomes.count('draw'),unknown=outcomes.count('unknown'),
            win_rate_all_games=wins/20,wilson_95=wilson_interval(wins,20),
            observed_80_percent_target_met=None if 'unknown'in outcomes else wins>=16,
            observed_simulation=dict(wins=observed.count('win'),losses=observed.count('loss'),draws=observed.count('draw'),
                unknown=observed.count('unknown'),win_rate_all_games=observed.count('win')/20,
                acceptance_confirmed=False))
    defense=[r for r in records_by_split['test']if r['task']=='defense']
    gate=acceptance_gate(frozen,defense,matches,coverage_complete=False)
    return dict(schema='verified-selected-ppo-final-summary-v1',mode=mode,profile=selection['profile'],checkpoint_sha256=config['policy_sha256'],
        training_steps=2048000,evaluation_source_sha256=manifest['archive_sha256'],training_source_sha256=selection['source_sha256'],
        validation=splits['validation'],test=splits['test'],opponents=opponents,acceptance=gate,
        environment_migration=manifest.get('environment_migration'),quality_accepted=False,simulation_only=True,
        test_used_to_change_weights=False,raw_reward_used_to_rank_profiles=False,
        quality_note='Unknown attribution/outcomes remain unknown; live mechanics parity and provisional thresholds still require acceptance.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','prepared','evaluation-package','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();report=verify(args.collected,args.prepared,args.evaluation_package)
    with args.output.open('x')as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
