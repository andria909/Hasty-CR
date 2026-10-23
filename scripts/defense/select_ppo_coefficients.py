"""Select PPO reward profiles from pinned, complete validation trials only."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from sim.ppo_tuning import quality
from sim.reward_profiles import load_reward_profile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranking(rows):
    """Aggregate training seeds equally; never compare reward values."""
    if not rows:raise ValueError('Missing seed measurements')
    keys=('score','king_survival','mean_hp_fraction_loss','mean_elixir_spent')
    for row in rows:
        if row.get('eligible') is not True or row.get('raw_reward_used_for_selection') is not False:
            raise ValueError('Ineligible quality result')
        if any(isinstance(row.get(k),bool) or not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) for k in keys):
            raise ValueError('Nonfinite or missing ranking metric')
    return {k:sum(r[k] for r in rows)/len(rows) for k in keys}


def order(row):
    return (-row['score'],-row['king_survival'],row['mean_hp_fraction_loss'],row['mean_elixir_spent'],row['profile'])


def read_trials(directories,protocol,contexts,manifest,profile_dir):
    found={};evidence=[]
    for directory in directories:
        root=Path(directory);receipt=json.loads((root/'receipt.json').read_text())
        if receipt.get('status')!='COMPLETE' or receipt.get('complete') is not True:raise ValueError('Complete terminal study receipt required')
        def read(name):
            if name not in receipt['files'] or sha(root/name)!=receipt['files'][name]['sha256']:raise ValueError('Missing or changed study artifact')
            return json.loads((root/name).read_text())
        study=read('study_result.json')
        if study.get('status')!='completed' or study.get('source_sha256')!=manifest['archive_sha256'] or study.get('test_used_for_search') is not False:
            raise ValueError('Study provenance or validation-only contract differs')
        if study.get('training_steps_per_trial')!=protocol['short_environment_steps']:raise ValueError('Study budget changed')
        ids=[r['trial_id'] for r in study['trials']]
        if len(ids)!=len(set(ids)) or set(ids)!=set(receipt['trial_ids']):raise ValueError('Collected trial schedule differs')
        for row in study['trials']:
            mode,profile,seed=row['mode'],row['profile'],row['seed'];identity=(mode,profile,seed)
            if mode not in protocol['modes'] or profile not in protocol['profiles'] or seed not in (protocol['screen_training_seed'],protocol['confirmation_training_seed']):
                raise ValueError('Unexpected study trial')
            if identity in found:raise ValueError('Duplicate trial across studies')
            trial=f'{mode}_{profile}_s{seed}'
            if row['trial_id']!=trial or row.get('status')!='completed' or row.get('training_steps')!=protocol['short_environment_steps']:
                raise ValueError('Incomplete trial budget or identity')
            if read(trial+'/trial_result.json')!=row:raise ValueError('Trial report differs from study manifest')
            checkpoint=trial+'/train/update_000100.pt';validation=trial+'/validation.json'
            if row['checkpoint']!=checkpoint or row['validation']!=validation:raise ValueError('Unexpected trial artifact path')
            if receipt['files'].get(checkpoint,{}).get('sha256')!=row['checkpoint_sha256'] or sha(root/checkpoint)!=row['checkpoint_sha256']:
                raise ValueError('Checkpoint changed')
            result=read(validation);config=read(trial+'/train/config.json')
            _,reward_sha=load_reward_profile(Path(profile_dir)/(profile+'.json'))
            if sha(Path(profile_dir)/(profile+'.json'))!=manifest['files']['data/ppo_tuning/profiles/'+profile+'.json']:
                raise ValueError('Reward profile differs from frozen study source')
            fixed={k:v for k,v in protocol['fixed_training'].items() if not k.startswith('joint_')}
            fixed.update(seed=seed,envs=protocol['envs'],rollout=protocol['rollout'],updates=protocol['short_updates'])
            if mode=='joint':fixed.update(deadline_scale=protocol['fixed_training']['joint_deadline_scale'],visible_fraction=protocol['fixed_training']['joint_visible_fraction'])
            if any(config.get(k)!=v for k,v in fixed.items()):raise ValueError('Paired training settings differ')
            if config.get('init_sha256')!=protocol['initial_policies'][mode]['sha256'] or config.get('reward_profile_sha256')!=reward_sha:
                raise ValueError('Teacher or reward coefficients changed')
            if (result.get('mode')!=mode or result.get('profile')!=profile or result.get('training_steps')!=102400 or
                result.get('checkpoint_sha256')!=row['checkpoint_sha256'] or result.get('reward_profile_sha256')!=reward_sha or
                result.get('test_used_for_selection') is not False or row['validation_sha256']!=sha(root/validation)):
                raise ValueError('Validation provenance differs')
            expected={(r['scenario_id'],r['seed'],r['task']) for r in contexts['contexts'][mode]}
            actual=[(r.get('strategy_scenario_id') if mode=='joint' else r.get('scenario_id'),r.get('seed'),r.get('task','defense')) for r in result['records']]
            if len(actual)!=128 or len(set(actual))!=128 or set(actual)!=expected:raise ValueError('Fixed validation contexts differ')
            measured=quality(result['records'],result['duels'],mode)
            if measured!=result['quality'] or measured!=row['quality']:raise ValueError('Quality differs from raw validation')
            if measured.get('eligible'):ranking([measured])
            found[identity]=measured
        evidence.append({'directory':str(root),'receipt_sha256':sha(root/'receipt.json'),'study_sha256':sha(root/'study_result.json')})
    return found,evidence


def select(directories,protocol_file,contexts_file,manifest_file,profile_dir,stage):
    protocol=json.loads(Path(protocol_file).read_text());contexts=json.loads(Path(contexts_file).read_text());manifest=json.loads(Path(manifest_file).read_text())
    if sha(contexts_file)!=protocol['validation']['contexts_sha256'] or contexts.get('split')!='validation':raise ValueError('Validation schedule changed')
    if manifest['files']['data/ppo_tuning/protocol.json']!=sha(protocol_file):raise ValueError('Protocol differs from frozen study source')
    trials,evidence=read_trials(directories,protocol,contexts,manifest,profile_dir)
    screen_seed=protocol['screen_training_seed'];confirm_seed=protocol['confirmation_training_seed'];modes={}
    expected={(m,p,screen_seed) for m in protocol['modes'] for p in protocol['profiles']}
    if not expected.issubset(trials):raise ValueError('All16 screening trials required before selection')
    confirmations=set()
    for mode in protocol['modes']:
        screened=sorted([dict(profile=p,**ranking([trials[mode,p,screen_seed]])) for p in protocol['profiles'] if trials[mode,p,screen_seed].get('eligible')],key=order)
        if len(screened)<protocol['confirm_top_profiles_per_mode']:raise ValueError('Too few eligible screening profiles')
        finalists=[r['profile'] for r in screened[:protocol['confirm_top_profiles_per_mode']]]
        confirmations.update((mode,p,confirm_seed) for p in finalists)
        modes[mode]={'screen_ranking':screened,'confirmation_profiles':finalists,
                     'ineligible_screen_profiles':[p for p in protocol['profiles'] if not trials[mode,p,screen_seed].get('eligible')]}
    if stage=='screen':
        if set(trials)!=expected:raise ValueError('Screen selection accepts screening seed only')
    elif stage=='confirmed':
        if set(trials)!=expected|confirmations:raise ValueError('Exactly the6 predefined confirmation trials required')
        for mode,result in modes.items():
            confirmed=sorted([dict(profile=p,**ranking([trials[mode,p,s] for s in (screen_seed,confirm_seed)])) for p in result['confirmation_profiles'] if all(trials[mode,p,s].get('eligible') for s in (screen_seed,confirm_seed))],key=order)
            if len(confirmed)<protocol['full_winners_per_mode']:raise ValueError('Too few eligible confirmed profiles')
            result.update(confirmed_ranking=confirmed,full_training_profiles=[r['profile'] for r in confirmed[:protocol['full_winners_per_mode']]])
    else:raise ValueError('Unknown selection stage')
    return dict(schema='ppo-coefficient-selection-v1',stage=stage,modes=modes,evidence=evidence,
                protocol_sha256=sha(protocol_file),source_sha256=manifest['archive_sha256'],test_used_for_selection=False,
                raw_reward_used_for_selection=False,quality_accepted=False,full_training_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,action='append',required=True)
    p.add_argument('--stage',choices=('screen','confirmed'),required=True)
    p.add_argument('--protocol',type=Path,default=Path('data/ppo_tuning/protocol.json'))
    p.add_argument('--contexts',type=Path,default=Path('data/ppo_tuning/validation_contexts.json'))
    p.add_argument('--profiles',type=Path,default=Path('data/ppo_tuning/profiles'))
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=select(args.study,args.protocol,args.contexts,args.manifest,args.profiles,args.stage)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(report['modes']))


if __name__=='__main__':main()
