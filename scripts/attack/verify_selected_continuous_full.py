"""Audit full-only continuations against their original verified pilot study."""
import argparse
import json
from pathlib import Path

from scripts.attack.verify_continuous_focus import verify as verify_pilot,audit_run,sha


def verify(collected,prepared):
    root=Path(collected);prepared=Path(prepared);cfg=json.loads((prepared/'preparation.json').read_text());c=cfg['config']
    original=Path(cfg['pilot_prepared']);pilot=verify_pilot(original/'collected',original)
    pinned=json.loads((prepared/'dataset/pilot_verified.json').read_text())
    if pilot!=pinned:raise ValueError('Original pilot proof changed')
    selection=pilot['pilot_selection']
    if not selection['gate_met']or len(selection['selected'])!=2:raise ValueError('Two eligible pilot profiles required')
    for name,digest in c['pins'].items():
        if sha(prepared/'dataset'/name)!=digest:raise ValueError('Full input changed')
    receipt=json.loads((root/'collection_receipt.json').read_text())
    if not receipt.get('complete')or receipt.get('remote_status')!='COMPLETE'or receipt['kernel']!=cfg['kernel']:
        raise ValueError('Matching terminal collection required')
    for name,meta in receipt['files'].items():
        path=root/name
        if path.is_symlink()or not path.resolve().is_relative_to(root.resolve())or sha(path)!=meta['sha256']:
            raise ValueError('Full artifact changed')
    result=json.loads((root/'result.json').read_text())
    if(result.get('status')!='complete'or result.get('pilot_steps')!=0 or result.get('full_steps')!=8192000
            or result.get('protocol_sha256')!=c['pins']['protocol.json']or result.get('initial_selection')!=pilot['initial_selection']):
        raise ValueError('Full-only result budget or initialization differs')
    initial=next(m for m in c['models']if m['id']==pilot['initial_selection']['selected'])
    reference=json.loads((original/'collected/parent_reference.json').read_text())
    contexts=[(r['scenario_id'],r['episode_seed'],r['match_seed'])for r in reference['validation']['records']]
    rows=[]
    for chosen in selection['selected']:
        trial=chosen['trial'];rows.append(audit_run(root,c,initial,contexts,'full_'+trial['id']+'_s64',trial,64,4096000))
    if rows!=result['full_runs']:raise ValueError('Final full model selection differs')
    import torch
    torch.set_num_threads(1)
    for row in rows:
        blob=torch.load(root/row['checkpoint'],map_location='cpu',weights_only=False)
        if(blob['state_dict']['scalar.0.weight'].shape[1]!=309 or
                blob['state_dict']['conv.0.weight'].shape[1]!=(41 if row['trial'].get('history_frames',1)==2 else 20)or
                not all(torch.isfinite(t).all()for t in blob['state_dict'].values())):
            raise ValueError('Invalid full model weights')
    return dict(schema='verified-selected-continuous-full-v1',kernel=cfg['kernel'],pilot_kernel=pilot['kernel'],
        full_steps_verified=8192000,pilot_steps_repeated=0,
        models=[dict(id=r['id'],checkpoint=str(root/r['checkpoint']),checkpoint_sha256=r['checkpoint_sha256'],
            family_guard=r['selected_with_family_guard'],selection_score=r['validation']['selection_score'])for r in rows],
        ready_for_independent_evaluation=all(r['selected_with_family_guard']for r in rows),
        quality_accepted=False,test_used_for_selection=False,live_checkpoint_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collected',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();Path(a.output).write_text(json.dumps(verify(a.collected,a.prepared),indent=2)+'\n')
