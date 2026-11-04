"""Audit observed training choices and the masked-policy loss denominator.

Completed episode counters are not rollout minibatches. This report never
labels their fraction as a measured gradient norm or an effective learning rate.
"""
import argparse
import json
from pathlib import Path

from scripts.attack.decision_behavior import economy
from scripts.kaggle.local_artifact_transfer import file_sha


def gradient_probe():
    import torch
    from sim.train_attack_ppo import training_distribution
    # Deterministic algebra check of the actual masking/distribution path.
    logits=torch.linspace(-1,1,40,dtype=torch.float64).reshape(20,2).clone().requires_grad_(True)
    mask=torch.zeros_like(logits,dtype=torch.bool);mask[:,0]=True;mask[:4,1]=True
    choice=mask.sum(-1)>1
    actions=torch.tensor([1,0,1,0]+[0]*16)
    advantages=torch.linspace(-2,3,20,dtype=torch.float64)
    dist=training_distribution(logits,mask,0.)
    old=dist.log_prob(actions).detach()
    ratio=(dist.log_prob(actions)-old).exp()
    terms=-torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages)
    all_gradient=torch.autograd.grad(terms.mean(),logits,retain_graph=True)[0]
    choice_gradient=torch.autograd.grad(terms[choice].mean(),logits)[0]
    fraction=float(choice.double().mean())
    if (torch.count_nonzero(all_gradient[~choice]) or
            not torch.allclose(all_gradient,choice_gradient*fraction,atol=1e-12,rtol=1e-12)):
        raise ValueError('Masked policy gradient identity failed')
    return dict(states=20,choice_states=4,forced_wait_states=16,
                actor_gradient_on_forced_wait=0.,all_mean_to_choice_mean_factor=fraction,
                fixed_advantages=True,synthetic_algebra_check=True,
                scope='Before clipping and shared-network/value gradients; not a measured training gradient')


def audit(root,proof_path):
    root,proof_path=Path(root),Path(proof_path)
    proof=json.loads(proof_path.read_text());prep=json.loads((root/'preparation.json').read_text())
    receipt_path=root/'collected/collection_receipt.json';receipt=json.loads(receipt_path.read_text())
    if (proof['kernel']!=prep['kernel'] or receipt['kernel']!=prep['kernel'] or
            not receipt.get('complete') or receipt.get('remote_status')!='COMPLETE' or
            proof.get('pilot_steps_verified')!=3276800):
        raise ValueError('Completed long-study evidence required')
    protocol=json.loads((root/'dataset/protocol.json').read_text());rows=[]
    def checked(name):
        path=root/'collected'/name
        if file_sha(path)!=receipt['files'][name]['sha256']:
            raise ValueError('Collected training evidence changed: '+name)
        return path
    for trial in protocol['cases']:
        for seed in protocol['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}"
            metrics=[json.loads(line)for line in checked(name+'/metrics.jsonl').read_text().splitlines()]
            if [r['update']for r in metrics]!=list(range(1,401)):
                raise ValueError('All400 consecutive updates required')
            episodes=[e for r in metrics if r['update']>=50 for e in r['episodes']]
            if not episodes or any(e['split']!='train'for e in episodes):
                raise ValueError('Actual completed training episodes required')
            result=economy(episodes)
            if not result['complete']:raise ValueError('Decision economy counters missing')
            selected=json.loads(checked(name+'_summary.json').read_text())
            validated=economy(selected['validation']['records'])
            rows.append(dict(id=name,training_completed_episodes= len(episodes),
                first_included_completion_update=50,last_update=400,training=result,selected_validation=validated))
    return dict(schema='policy-choice-exposure-audit-v1',kernel=prep['kernel'],rows=rows,
                proof_sha256=file_sha(proof_path),receipt_sha256=file_sha(receipt_path),
                trainer_sha256=file_sha('sim/train_attack_ppo.py'),gradient_probe=gradient_probe(),
                scope='Completed TRAIN episodes ending at updates50–400; not exact per-minibatch samples',
                limitations=['Policies change during TRAIN episodes; validation uses selected fixed weights',
                    'Actor gradient identity holds for fixed advantages; normalization and shared value gradients also matter',
                    'Changing the loss denominator is a hypothesis, not an established fix'],
                training_modified=False,quality_accepted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','proof','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();result=audit(a.root,a.proof)
    target=Path(a.output)
    if target.exists():raise ValueError('Preserve previous audit reports')
    target.write_text(json.dumps(result,indent=2)+'\n')
