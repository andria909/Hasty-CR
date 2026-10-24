"""Audit a terminal Fireball refinement before any independent evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.attack.train_fireball_refinement import PARENT_SHA,KINDS,contexts
from sim.fireball_curriculum import CONTRACT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(collected,prepared,parent):
    root=Path(collected);prepared=Path(prepared)
    receipt=json.loads((root/'collection_receipt.json').read_text())
    preparation=json.loads((prepared/'preparation.json').read_text())
    if (receipt.get('remote_status')!='COMPLETE' or receipt.get('complete')is not True or
            receipt.get('kernel')!=preparation['kernel']):
        raise ValueError('Complete collection of the same kernel required')
    def checked(name):
        path=root/name
        if receipt['files'].get(name,{}).get('sha256')!=sha(path):
            raise ValueError('Collected artifact SHA differs: '+name)
        return path
    result=json.loads(checked('result.json').read_text())
    if result.get('status')!='complete' or not result.get('candidate_available'):
        raise ValueError('No selected completed candidate')
    if sha(parent)!=PARENT_SHA or preparation['parent_sha256']!=PARENT_SHA:
        raise ValueError('Original selected joint policy differs')
    if sha(prepared/'job/train_fireball.py')!=preparation['bootstrap_sha256']:
        raise ValueError('Prepared bootstrap differs')
    source=json.loads((prepared/'dataset/manifest.json').read_text())
    if sha(prepared/'dataset/source.tar.gz')!=source['archive_sha256'] or source['archive_sha256']!=preparation['source_sha256']:
        raise ValueError('Prepared source differs')
    baseline=json.loads(checked('baseline_validation.json').read_text())
    rows=[json.loads(line) for line in checked('metrics.jsonl').read_text().splitlines()]
    blob=torch.load(checked('candidate.pt'),map_location='cpu',weights_only=False)
    config=blob['config']
    if (config.get('parent_sha256')!=PARENT_SHA or blob.get('parent_sha256')!=PARENT_SHA or
            config.get('fireball_contract')!=CONTRACT or config.get('num_scalars')!=309 or
            config.get('method')!='impact_verified_BC_with_policy_rehearsal' or
            config.get('test_used_for_selection')is not False):
        raise ValueError('Candidate training contract differs')
    if not blob.get('state_dict') or not all(torch.isfinite(t).all()for t in blob['state_dict'].values()):
        raise ValueError('Candidate has missing/nonfinite weights')
    if blob['state_dict']['scalar.0.weight'].shape[1]!=309:
        raise ValueError('Candidate scalar dimension differs')
    if [r.get('epoch')for r in rows]!=list(range(1,config['epochs']+1)):
        raise ValueError('Incomplete epoch history')
    expected=set(contexts('validation',config['eval_episodes']))
    def check_validation(report):
        actual=[(r['kind'],r['lane'],r['seed'])for r in report['rows']]
        if len(actual)!=len(expected) or set(actual)!=expected:
            raise ValueError('Validation contexts differ')
        if any(type(r.get('success'))is not bool for r in report['rows']):
            raise ValueError('Missing task outcome')
        total=sum(r['success']for r in report['rows'])
        if report['successes']!=total or report['episodes']!=len(actual):
            raise ValueError('Validation aggregate differs')
        for kind in KINDS:
            subset=[r for r in report['rows']if r['kind']==kind]
            if report['by_kind'][kind]!={'successes':sum(r['success']for r in subset),'episodes':len(subset)}:
                raise ValueError('Validation kind aggregate differs')
        return total/len(actual)
    check_validation(baseline)
    best=-1;best_epoch=0;selected=None
    for row in rows:
        if 'validation'not in row:continue
        score=check_validation(row['validation'])
        if score>best and row['rehearsal_action_agreement']>=config['min_agreement']:
            best=score;best_epoch=row['epoch'];selected=row
    if (best_epoch!=blob['epoch'] or best_epoch!=result['best_epoch'] or
            best!=result['best_validation'] or blob['validation']!=selected['validation'] or
            blob['rehearsal_action_agreement']!=selected['rehearsal_action_agreement']):
        raise ValueError('Candidate is not the predetermined validation selection')
    audit=json.loads(checked('demonstration_audit.json').read_text())
    if sum(audit['accepted'].values())+len(audit['rejected'])!=config['demonstrations']:
        raise ValueError('Demonstration accounting differs')
    return dict(schema='verified-fireball-candidate-v1',checkpoint_sha256=sha(root/'candidate.pt'),
        parent_sha256=PARENT_SHA,source_sha256=preparation['source_sha256'],epoch=best_epoch,
        method=config['method'],validation_before=baseline['by_kind'],
        validation_after=selected['validation']['by_kind'],rehearsal_action_agreement=selected['rehearsal_action_agreement'],
        independent_test_pending=True,defense_regression_pending=True,matches_pending=True,
        quality_accepted=False,live_checkpoint_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','prepared','parent','output'):p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args();report=verify(args.collected,args.prepared,args.parent)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
