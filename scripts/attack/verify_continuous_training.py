"""Verify budgets, selected checkpoints and full-game validation; no promotion."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from scripts.attack.verify_continuous_focus import metric_boundaries


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(collected,prepared):
    collected=Path(collected);prepared=Path(prepared)
    cfg=json.loads((prepared/'preparation.json').read_text())
    receipt=json.loads((collected/'collection_receipt.json').read_text())
    if not receipt.get('complete')or receipt.get('remote_status')!='COMPLETE' or receipt['kernel']!=cfg['kernel']:
        raise ValueError('Complete matching remote receipt required')
    for name,meta in receipt['files'].items():
        p=collected/name
        if p.is_symlink()or not p.resolve().is_relative_to(collected.resolve())or digest(p)!=meta['sha256']:
            raise ValueError('Collected artifact differs: '+name)
    result=json.loads((collected/'result.json').read_text())
    if result.get('protocol_sha256')!=cfg['config']['protocol_sha256'] or result.get('status')!='complete':
        raise ValueError('Training protocol/status differs')
    trials=cfg['config']['trials'];rows=[]
    for trial in trials:
        for seed in (42,43):
            run=f"pilot_{trial['id']}_s{seed}"
            boundary=json.loads((collected/run/'training_result.json').read_text())
            if boundary.get('step')!=102400 or not boundary.get('target_met'):raise ValueError('Incomplete pilot')
            metric_boundaries(collected/run/'metrics.jsonl',102400)
    full=result['full_runs']
    if len(full)!=2 or len({r['trial']['id']for r in full})!=2:raise ValueError('Two distinct full models required')
    import torch
    torch.set_num_threads(1)
    for item in full:
        run=item['id'];boundary=json.loads((collected/run/'training_result.json').read_text())
        if not boundary.get('target_met')or boundary.get('step')!=4096000:raise ValueError('Incomplete full budget')
        metric_boundaries(collected/run/'metrics.jsonl',4096000)
        p=collected/item['checkpoint']
        if digest(p)!=item['checkpoint_sha256']:raise ValueError('Selected checkpoint pin differs')
        blob=torch.load(p,map_location='cpu',weights_only=False)
        if blob['state_dict']['scalar.0.weight'].shape[1]!=309 or not all(torch.isfinite(t).all()for t in blob['state_dict'].values()):
            raise ValueError('Invalid selected weights')
        if blob['config']['init_sha256']!=cfg['config']['source_pins'][cfg['config'].get('parent_file','parent.pt')]:raise ValueError('Parent changed')
        validation=item['validation'];records=validation['records'];count=Counter(r['outcome']for r in records)
        if count['unknown']or len(records)!=48:raise ValueError('Validation missing terminal outcomes')
        score=(count['win']+.5*count['draw'])/len(records)
        if abs(score-validation['selection_score'])>1e-12:raise ValueError('Validation score mismatch')
        rows.append(dict(id=run,checkpoint=str(p),checkpoint_sha256=digest(p),steps=boundary['step'],
            validation=dict(games=len(records),wins=count['win'],losses=count['loss'],draws=count['draw'],score=score),
            mean_elixir_overflow=validation['mean_elixir_overflow']))
    return dict(schema='verified-continuous-ppo-training-v1',kernel=cfg['kernel'],
        all_training_budgets_verified=True,pilot_steps=819200,full_steps=8192000,models=rows,
        actual_cuda_steps_verified=True,actual_consecutive_updates_verified=8800,
        quality_accepted=False,independent_test_pending=True,live_checkpoint_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collected',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=verify(a.collected,a.prepared);Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))
