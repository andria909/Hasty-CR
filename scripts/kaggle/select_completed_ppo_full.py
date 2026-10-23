"""Freeze a predetermined final PPO checkpoint for evaluation; never test-select."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

from sim.reward_profiles import load_reward_profile


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def select(collected,launch_config,source_manifest,output,profiles=Path('data/ppo_tuning/profiles')):
    import torch
    root=Path(collected);config=json.loads(Path(launch_config).read_text());source=json.loads(Path(source_manifest).read_text())
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt.get('status')!='COMPLETE' or receipt.get('complete') is not True:raise ValueError('Complete terminal full-training receipt required')
    def checked(name):
        path=root/name
        if receipt.get('files',{}).get(name,{}).get('sha256')!=sha(path):raise ValueError('Missing or changed collected full-training file')
        return path
    result=json.loads(checked('result.json').read_text());validation=json.loads(checked('validation.json').read_text())
    mode,profile=config['mode'],config['profile'];selection=config['selection']
    if mode not in ('defense','joint') or profile not in selection['modes'][mode]['full_training_profiles'] or selection.get('stage')!='confirmed':
        raise ValueError('Full-training profile was not confirmed before training')
    selection_sha=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()
    selection_source=source['archive_sha256']
    if 'migration'in config:
        from scripts.defense.pack_evaluation import verified_frozen_package
        # Re-verify the original pinned source and saved boundary, not just labels
        # in the remote result. No source fingerprint or hash check is disabled.
        source_root=Path(source_manifest).parent
        members,verified,_,_=verified_frozen_package(source_root)
        if verified!=source:raise ValueError('Migrated training manifest differs')
        original=json.loads((source_root/'original_manifest.json').read_text())
        name='sim/defense_env.py';expression='return arena.clamp_to_arena(chosen.pos)'
        text=members[name].decode()
        if text.count(expression)!=1 or hashlib.sha256(text.replace(expression,'return chosen.pos').encode()).hexdigest()!=original['files'][name]:
            raise ValueError('Migrated environment differs from declared expression')
        changed={n for n in set(source['files'])|set(original['files'])if source['files'].get(n)!=original['files'].get(n)}
        migration=config['migration']
        if ((mode,profile)!=('defense','economy_soft')or changed!={name} or migration.get('schema')!='script-target-clamp-migration-v1' or
            sha(source_root/'original_manifest.json')!=migration.get('original_manifest_sha256') or
            sha(source_root/'resume.pt')!=migration.get('resume_checkpoint_sha256') or
            (migration.get('resume_update'),migration.get('resume_step'))!=(796,815104) or
            result.get('migration')!=migration or result.get('source_unchanged')is not False or
            result.get('same_source_exact_resume')is not False or
            original['archive_sha256']!=migration.get('original_source_sha256')):
            raise ValueError('Migrated full-training lineage differs')
        selection_source=original['archive_sha256']
    if (selection.get('test_used_for_selection') is not False or result.get('test_used_for_selection') is not False or
        result.get('selection_sha256')!=selection_sha or result.get('source_sha256')!=source['archive_sha256'] or
        selection.get('source_sha256')!=selection_source):raise ValueError('Full-training source or selection provenance differs')
    if result.get('status')!='completed' or result.get('step')!=2048000 or result.get('update')!=2000:
        raise ValueError('Exact full-training budget not completed')
    if result.get('mode')!=mode or result.get('profile')!=profile or result.get('checkpoint')!='train/update_002000.pt':
        raise ValueError('Predetermined final checkpoint identity differs')
    checkpoint=checked('train/update_002000.pt');checkpoint_sha=sha(checkpoint)
    if checkpoint_sha!=result.get('checkpoint_sha256') or sha(root/'validation.json')!=result.get('validation_sha256'):
        raise ValueError('Full-training output SHA differs')
    if (validation.get('checkpoint_sha256')!=checkpoint_sha or validation.get('training_steps')!=2048000 or
        validation.get('test_used_for_selection') is not False or validation.get('mode')!=mode or validation.get('profile')!=profile):
        raise ValueError('Validation does not describe the final checkpoint')
    profile_file=Path(profiles)/(profile+'.json');_,reward_sha=load_reward_profile(profile_file)
    if sha(profile_file)!=source['files']['data/ppo_tuning/profiles/'+profile+'.json']:raise ValueError('Reward file differs from frozen source')
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False);stored=blob['config']
    if blob.get('step')!=2048000 or blob.get('update')!=2000:raise ValueError('Checkpoint counters differ from completed report')
    if stored.get('reward_profile_sha256')!=reward_sha or stored.get('init_sha256')!=config['pins']['teacher_'+mode+'.pt']:
        raise ValueError('Checkpoint reward or teacher differs')
    dimensions=309 if mode=='joint' else 306
    scalar=blob.get('state_dict',{}).get('scalar.0.weight')
    if scalar is None or scalar.ndim!=2 or scalar.shape[1]!=dimensions:raise ValueError('Checkpoint input contract differs')
    if (stored.get('geometry')!='legacy_v1' or stored.get('coordinate_mode')!=('physical_live_cells_v2'if mode=='joint'else'physical_live') or
        mode=='joint' and (stored.get('task_mode')!='joint' or stored.get('num_scalars')!=309)):
        raise ValueError('Checkpoint environment contract differs')
    if not blob.get('exact_resume_available') or len(blob.get('environment_states',[]))!=8:raise ValueError('Final checkpoint lacks exact boundary snapshots')
    def finite(value):
        if torch.is_tensor(value):
            if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite final policy or optimizer')
        elif isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite final policy or optimizer')
        elif isinstance(value,dict):
            for child in value.values():finite(child)
        elif isinstance(value,(list,tuple)):
            for child in value:finite(child)
    if not blob.get('optimiser') or not blob.get('state_dict'):raise ValueError('Incomplete final PPO state')
    finite(blob['state_dict']);finite(blob['optimiser'])
    report=dict(schema='selected-full-ppo-evaluation-input-v1',mode=mode,profile=profile,selected_sha256=checkpoint_sha,
                training_steps=2048000,training_update=2000,num_scalars=dimensions,source_sha256=source['archive_sha256'],
                coefficient_selection_sha256=selection_sha,teacher_sha256=stored['init_sha256'],reward_profile_sha256=reward_sha,
                collection_receipt_sha256=sha(root/'receipt.json'),training_result_sha256=sha(root/'result.json'),
                validation_sha256=sha(root/'validation.json'),selection_basis='preselected coefficient profile; predetermined final update2000',
                test_used_for_selection=False,quality_accepted=False,policy_optimizer_finite=True,exact_resume_available=True)
    if 'migration'in config:report['environment_migration']=config['migration']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(checkpoint,output/'policy.pt')
    if sha(output/'policy.pt')!=checkpoint_sha:raise ValueError('Copied final policy SHA mismatch')
    (output/'selection.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','launch-config','source-manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(json.dumps(select(args.collected,args.launch_config,args.source_manifest,args.output),indent=2))


if __name__=='__main__':main()
