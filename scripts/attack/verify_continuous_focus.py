"""Audit validation-only initialization, pilot selection, budgets and weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.attack.select_continuous_focus import select_initial,select_profiles,validation_score


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_boundaries(path,target_steps,policy_temperature=None):
    validations=[];update=0;step=0
    with Path(path).open()as stream:
        for line in stream:
            record=json.loads(line);update+=1;step+=1024
            if(record.get('update')!=update or record.get('step')!=step or
                    record.get('target_environment_steps')!=target_steps or not record.get('actual_cuda')or
                    abs(record.get('percent',-1)-100*step/target_steps)>1e-9):
                raise ValueError('Nonconsecutive or inconsistent actual training counters')
            if policy_temperature is not None and record.get('policy_temperature',1.)!=policy_temperature:
                raise ValueError('Actual rollout policy temperature differs')
            if'validation'in record:validations.append(record)
    if step!=target_steps:raise ValueError('Metrics do not reach the actual target budget')
    return validations


def audit_run(root,c,selected_initial,initial_contexts,run_id,trial,seed,steps):
    row=json.loads((root/(run_id+'_summary.json')).read_text())
    boundary=json.loads((root/run_id/'training_result.json').read_text())
    config=json.loads((root/run_id/'config.json').read_text())
    expected_initial=selected_initial
    if trial.get('warmstart_updates',0):
        warm=root/(run_id+'_warmstart');w=json.loads((warm/'result.json').read_text())
        for name in('air_manifest.json','air_examples.npz'):
            if sha(root.parent/'dataset'/name)!=c['pins'][name]:raise ValueError('Warmstart examples changed')
        manifest=json.loads((root.parent/'dataset/air_manifest.json').read_text())
        groups=set(manifest['seeds']);heldout=max(groups)
        if(w.get('supervised_updates')!=trial['warmstart_updates']or w.get('environment_steps')!=0 or
                not w.get('actual_cuda')or w.get('seed')!=seed or w.get('kl_weight')!=trial['warmstart_kl_weight']or
                w.get('initial_sha256')!=selected_initial['sha256']or w.get('data_sha256')!=c['pins']['air_examples.npz']or
                w.get('source_sha256')!=c['extensions']['scripts/attack/train_air_position_warmstart.py']or
                w.get('heldout_seed')!=heldout or set(w.get('train_seeds',[]))!=groups-{heldout}):
            raise ValueError('Warmstart origin, budget or grouped split differs')
        updates=json.loads((warm/'updates.json').read_text())
        if(len(updates)!=trial['warmstart_updates']or any(r['update']!=i or
                not all(math.isfinite(r[k])for k in('loss','imitation','kl'))for i,r in enumerate(updates,1))):
            raise ValueError('Warmstart optimizer updates incomplete')
        expected_initial=dict(id=run_id+'_warmstart',file='/kaggle/working/continuous_focus/'+run_id+'_warmstart/policy.pt',sha256=sha(warm/'policy.pt'))
        import torch
        wb=torch.load(warm/'policy.pt',map_location='cpu',weights_only=False)
        if wb['config'].get('supervised_warmstart')!=w or not all(torch.isfinite(t).all()for t in wb['state_dict'].values()):
            raise ValueError('Warmstart checkpoint does not match its report')
    history_frames=trial.get('history_frames',1)
    expected_planes=41 if history_frames==2 else 20
    expected_contract=('joint_public_presence_continuous_slots_history2_v1'if history_frames==2
        else'joint_public_presence_continuous_slots_v2')
    if (config.get('num_planes',20)!=expected_planes or config.get('history_frames',1)!=history_frames or
            config.get('observation_contract')!=expected_contract):raise ValueError('History input contract differs')
    if not boundary.get('target_met')or boundary.get('step')!=steps or not row.get('target_met')or row.get('step')!=steps:
        raise ValueError('Incomplete focus training budget')
    if(row['id']!=run_id or row['trial']!=trial or row['seed']!=seed or row['initial_model']!=expected_initial or
        config['init_sha256']!=expected_initial['sha256']or config['continuous_focus']!=trial['focus']or
        config['seed']!=seed or
        config['validation_selection']!='worst_family'or config['validation_seed']!=900100 or
        config['entropy_coef']!=trial['entropy_coef']or config['num_scalars']!=309 or
        config.get('device')not in('cuda:0','cuda:1')or config.get('task_mode')!='joint'or
        config['card_balance_weight']!=trial['card_balance_weight']or config['card_balance_target']!='units'or
        config.get('policy_temperature',1.)!=trial.get('policy_temperature',1.)or
        config['continuous_profile']['overflow_penalty']!=trial['overflow_penalty']or
        config['gamma']!=trial.get('gamma',.999)or config['gae_lambda']!=trial.get('gae_lambda',.95)or
        config['continuous_profile'].get('missed_fireball_penalty',0.)!=trial.get('missed_fireball_penalty',0.)):
        raise ValueError('Focus run configuration differs')
    if config['continuous_profile'].get('board_potential_weight',0.)!=trial.get('board_potential_weight',0.):
        raise ValueError('Potential coefficient differs')
    for key in('fireball_value_reward','reserve_potential_weight','own_hp_penalty','barbarian_train_fraction','lane_reflection_fraction'):
        if key in trial and config['continuous_profile'].get(key)!=trial[key]:
            raise ValueError('Tactical reward coefficient differs: '+key)
    replay_weight=trial.get('supervised_replay_weight',0.)
    if config.get('supervised_replay_weight',0.)!=replay_weight:
        raise ValueError('Supervised replay coefficient differs')
    if replay_weight:
        from sim.supervised_replay import SuccessfulActionReplay
        prepared=root.parent/'dataset'
        import tempfile,shutil
        with tempfile.TemporaryDirectory()as temporary:
            data=Path(temporary)
            shutil.copyfile(prepared/'air_examples.npz',data/'examples.npz')
            shutil.copyfile(prepared/'air_manifest.json',data/'manifest.json')
            expected=SuccessfulActionReplay(data,'cpu').metadata
        if (config.get('supervised_replay_metadata')!=expected or
                expected['source_sha256']!=c['extensions']['sim/supervised_replay.py']):
            raise ValueError('Replay data, split or source differs')
        seen=0
        for line in(root/run_id/'metrics.jsonl').read_text().splitlines():
            metric=json.loads(line);replay=metric.get('supervised_replay',{})
            n=replay.get('optimizer_batches',-1)
            if (replay.get('weight')!=replay_weight or replay.get('train_examples')!=expected['train_examples']or
                    type(n)is not int or not 0<=n<=12 or
                    (n and (not math.isfinite(replay['mean_loss'])or metric['optimizer_controls']['value_only_warmup']))):
                raise ValueError('Actual supervised replay updates differ')
            seen+=n
        if not seen:raise ValueError('No supervised replay optimizer batches observed')
    if trial.get('board_potential_weight',0.)and config['continuous_profile']['shaping_gamma']!=config['gamma']:
        raise ValueError('Potential shaping discount differs from PPO')
    validations=metric_boundaries(root/run_id/'metrics.jsonl',steps,trial.get('policy_temperature',1.))
    for record in validations:
        v=record['validation'];score=validation_score(v)
        if abs(score['selection_score']-v['selection_score'])>1e-12:raise ValueError('Validation score differs')
        contexts=[(r['scenario_id'],r['episode_seed'],r['match_seed'])for r in v['records']]
        if contexts!=initial_contexts:raise ValueError('Validation contexts changed between candidates')
        if trial.get('board_potential_weight',0):
            for episode in v['records']:
                expected=-trial['board_potential_weight']*episode['initial_board_potential']
                if not math.isclose(episode['board_shaping_discounted_return'],expected,abs_tol=1e-7):
                    raise ValueError('Potential return changed completed-game objective')
        if trial.get('reserve_potential_weight',0):
            for episode in v['records']:
                # Regulation starts with five elixir; reserve Phi is capped4.
                expected=-4*trial['reserve_potential_weight']
                if not math.isclose(episode['reserve_shaping_discounted_return'],expected,abs_tol=1e-7):
                    raise ValueError('Reserve potential created a terminal hoarding bonus')
    eligible=[r for r in validations if r['validation']['worst_archetype_points']>=.1]
    chosen=max(eligible or validations,key=lambda r:r['validation']['selection_score'])
    if(row['validation']!=chosen['validation']or row['selected_update']!=chosen['update']or
            row['selected_with_family_guard']!=bool(eligible)):raise ValueError('Checkpoint selection differs')
    expected=f"{run_id}/update_{chosen['update']:06d}.pt"
    if row['checkpoint']!=expected or sha(root/expected)!=row['checkpoint_sha256']:raise ValueError('Selected weight differs')
    return row

def verify(collected,prepared):
    root=Path(collected);prepared=Path(prepared);cfg=json.loads((prepared/'preparation.json').read_text())
    c=cfg['config'];receipt=json.loads((root/'collection_receipt.json').read_text())
    if not receipt.get('complete')or receipt.get('remote_status')!='COMPLETE'or receipt['kernel']!=cfg['kernel']:
        raise ValueError('Complete matching focus receipt required')
    for name,meta in receipt['files'].items():
        p=root/name
        if p.is_symlink()or not p.resolve().is_relative_to(root.resolve())or sha(p)!=meta['sha256']:raise ValueError('Focus artifact changed')
    for name in('scripts/attack/select_continuous_focus.py','sim/continuous_validation.py'):
        if sha(name)!=c['extensions'][name]:raise ValueError('Use the frozen selector version to verify this cohort')
    protocol=json.loads((prepared/'dataset/protocol.json').read_text());result=json.loads((root/'result.json').read_text())
    if any(t.get('history_frames',1)==2 for t in protocol['cases']):
        for name in ('sim/public_frame_history.py','sim/continuous_history.py'):
            if sha(name)!=c['extensions'][name]:raise ValueError('Use frozen history adapter version')
    if any(t.get('board_potential_weight',0)for t in protocol['cases']):
        if sha('sim/board_potential.py')!=c['extensions']['sim/board_potential.py']:
            raise ValueError('Use frozen potential reward version')
    if protocol.get('schema')=='continuous-live-tactical-reward-pilot-v1':
        for name in('sim/tactical_rewards.py','sim/lane_reflection.py'):
            if sha(name)!=c['extensions'][name]:raise ValueError('Use frozen tactical reward/reflection version')
    if sha(prepared/'dataset/protocol.json')!=c['pins']['protocol.json']:raise ValueError('Prepared protocol changed')
    if result.get('schema')!='continuous-focus-cohort-result-v1'or result['protocol_sha256']!=c['pins']['protocol.json']:
        raise ValueError('Focus protocol differs')
    refs=[]
    for model in c['models']:
        row=json.loads((root/(model['id']+'_reference.json')).read_text())
        if row['id']!=model['id']or row['sha256']!=model['sha256']:raise ValueError('Warm-start reference differs')
        validation_score(row['validation']);refs.append(row)
    initial=select_initial(refs)
    if initial!=json.loads((root/'initial_selection.json').read_text())or initial!=result['initial_selection']:
        raise ValueError('Warm-start selection differs')
    selected_initial=next(m for m in c['models']if m['id']==initial['selected'])
    initial_contexts=[(r['scenario_id'],r['episode_seed'],r['match_seed'])for r in refs[0]['validation']['records']]
    from functools import partial
    audited_run=partial(audit_run,root,c,selected_initial,initial_contexts)
    pilots=[]
    seeds=protocol.get('pilot_seeds',(62,63));allow_full=c.get('allow_full_training',True)
    for trial in protocol['cases']:
        for seed in seeds:pilots.append(audited_run(f"pilot_{trial['id']}_s{seed}",trial,seed,102400))
    selection=select_profiles(pilots,seeds=seeds)
    if selection!=json.loads((root/'selection.json').read_text())or result['full_gate_met']!=selection['gate_met']:
        raise ValueError('Pilot profile selection differs')
    expected_status=('pilot_complete'if not allow_full else
        'complete'if selection['gate_met']else'pilot_complete_no_eligible_pair')
    if result.get('status')!=expected_status:raise ValueError('Focus completion status differs from gate')
    full=[]
    if selection['gate_met']and allow_full:
        for item in selection['selected']:
            trial=item['trial'];run_id='full_'+trial['id']+'_s64'
            full.append(audited_run(run_id,trial,64,4096000))
    elif result['full_runs']:raise ValueError('Full training started despite failed gate')
    if full!=result['full_runs']:raise ValueError('Final selected model reports differ')
    import torch
    torch.set_num_threads(1)
    for row in full+pilots:
        blob=torch.load(root/row['checkpoint'],map_location='cpu',weights_only=False)
        if(blob['state_dict']['scalar.0.weight'].shape[1]!=309 or
                blob['state_dict']['conv.0.weight'].shape[1]!=(41 if row['trial'].get('history_frames',1)==2 else 20)or
                not all(torch.isfinite(t).all()for t in blob['state_dict'].values())):raise ValueError('Invalid final weights')
    return dict(schema='verified-continuous-focus-cohort-v1',kernel=cfg['kernel'],pilot_steps_verified=819200,
        pilot_selection=selection,automatic_full_training=allow_full,
        pilot_models=[dict(id=r['id'],checkpoint=str(root/r['checkpoint']),checkpoint_sha256=r['checkpoint_sha256'],
            selection_score=r['validation']['selection_score'],worst_family_points=r['validation']['worst_archetype_points'],
            family_guard=r['selected_with_family_guard'])for r in pilots],
        full_steps_verified=4096000*len(full),full_gate_met=selection['gate_met'],initial_selection=initial,
        models=[dict(id=r['id'],checkpoint=str(root/r['checkpoint']),checkpoint_sha256=r['checkpoint_sha256'],
            selection_score=r['validation']['selection_score'],worst_family_points=r['validation']['worst_archetype_points'],
            family_guard=r['selected_with_family_guard'])for r in full],
        ready_for_independent_evaluation=bool(full)and all(r['selected_with_family_guard']for r in full),
        quality_accepted=False,test_used_for_selection=False,live_checkpoint_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collected',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=verify(a.collected,a.prepared);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
