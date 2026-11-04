"""Audit choice-aware PPO counters against effective optimizer settings."""
import math


def audit_update(row,config,trial):
    d=row.get('policy_decisions',{})
    for key,logged in (('policy_reduction','reduction'),('advantage_normalization','advantage_normalization')):
        if trial[key] not in ('all','choice') or config.get(key)!=trial[key] or d.get(logged)!=trial[key]:
            raise ValueError('Actual policy-choice mode differs: '+key)
    integers=('rollout_steps','choice_steps','forced_steps','optimizer_samples','choice_samples',
              'no_choice_minibatches','optimizer_batches')
    if any(type(d.get(k)) is not int or d[k]<0 for k in integers):
        raise ValueError('Actual integer policy-choice counters required')
    size=config['envs']*config['rollout'];epochs=config['epochs']
    max_batches=epochs*math.ceil(size/config['minibatch_size'])
    if (d['rollout_steps']!=size or d['choice_steps']+d['forced_steps']!=size or
            not 0<d['optimizer_samples']<=epochs*size or
            not 0<d['optimizer_batches']<=max_batches or
            d['choice_samples']>min(d['optimizer_samples'],epochs*d['choice_steps']) or
            d['no_choice_minibatches']>d['optimizer_batches']):
        raise ValueError('Policy-choice sample accounting is inconsistent')
    controls=row['optimizer_controls']
    if type(controls.get('kl_stopped')) is not bool or type(controls.get('value_only_warmup')) is not bool:
        raise ValueError('Actual optimizer controls required')
    if d.get('actor_updates_enabled')is not (not controls['value_only_warmup']):
        raise ValueError('Actor warmup accounting differs')
    if d['optimizer_batches']!=row['entropy_components']['optimizer_batches']:
        raise ValueError('Entropy and policy optimizer batches differ')
    if not controls['kl_stopped']:
        if (d['optimizer_samples']!=epochs*size or d['choice_samples']!=epochs*d['choice_steps'] or
                d['optimizer_batches']!=max_batches):
            raise ValueError('Unstopped PPO must complete every epoch')
    elif d['optimizer_batches']>=max_batches or config['target_kl']<=0 or controls['value_only_warmup']:
        raise ValueError('Invalid early KL stop')
    if d['choice_steps']==0 and (d['choice_samples'] or d['no_choice_minibatches']!=d['optimizer_batches']):
        raise ValueError('No-choice rollout accounting differs')
    if d['choice_steps']==size and (d['choice_samples']!=d['optimizer_samples'] or d['no_choice_minibatches']):
        raise ValueError('All-choice rollout accounting differs')
    entropy=row['entropy_components'];weight=config.get('spatial_entropy_weight',1.)
    full,group,objective=(entropy.get(k)for k in ('full','group','objective'))
    if (entropy.get('spatial_weight')!=weight or any(not isinstance(v,(int,float))or
            not math.isfinite(v)or v< -1e-5 for v in (full,group,objective))):
        raise ValueError('Invalid actual entropy accounting')
    if group>full+1e-5 or abs(objective-weight*full-(1-weight)*group)>1e-5:
        raise ValueError('Entropy decomposition differs')
    if d['choice_steps']==0 and any(abs(v)>1e-5 for v in (full,group,objective)):
        raise ValueError('A no-choice rollout cannot have action entropy')
    return d


def audit_checkpoint(config,blob,trial):
    for key in ('policy_reduction','advantage_normalization'):
        if config.get(key)!=trial[key] or blob['config'].get(key)!=trial[key]:
            raise ValueError('Saved policy-choice mode differs: '+key)
