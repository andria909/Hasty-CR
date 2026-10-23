"""Controlled v25 input-mode comparison: same four policies,80 paired games.

The original200-game tournament remains unchanged. Only v25's declared input
mode changes to the physical_live mode in the historical baseline allowlist.
"""
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

from scripts.kaggle.run_five_round_robin import sha,prepare_source

LAUNCH_CONFIG=None


def schedule(names):
    if len(names)!=5 or len(set(names))!=5 or 'v25'not in names:raise ValueError('Five participants including v25 required')
    return [dict(pair=[name,'v25'],seed=seed,candidate_side=side)for name in names if name!='v25'
            for seed in range(710001,710011)for side in (1,-1)]


def execute(config):
    output=Path('/kaggle/working/parser_control');output.mkdir(exist_ok=False)
    def mount(ref):
        owner,slug=ref.split('/')
        candidates=[p for p in (Path('/kaggle/input')/slug,Path('/kaggle/input/datasets')/owner/slug)if p.is_dir()]
        if len(candidates)!=1:raise ValueError('Expected one exact input mount')
        return candidates[0]
    source=mount(config['source_dataset']);weights=mount(config['weights_dataset'])
    if sha(weights/'participants.json')!=config['participants_sha256']:raise ValueError('Participant manifest SHA mismatch')
    participants=json.loads((weights/'participants.json').read_text())
    baseline=next(p for p in participants if p['id']=='v25')
    original_mode=baseline['mode']
    if config['v25_mode_override']!='physical_live':raise ValueError('Expected explicit parser input control')
    baseline['mode']=config['v25_mode_override']
    project=output/'project';manifest=prepare_source(source,project,config['manifest_sha256'])
    if manifest['archive_sha256']!=config['source_sha256']:raise ValueError('Wrong common source')
    for p in participants:
        path=project/p['path']if p['id']=='v25'else weights/p['path']
        if sha(path)!=p['sha256']:raise ValueError('Selected checkpoint SHA mismatch')
        p['resolved_path']=str(path)
    sys.path[:0]=[str(project),str(project/'scripts')]
    import torch
    from sim.eval_defense_duel import load_network,run_match
    from sim.defense_assets import load_defense_assets
    from sim.runner import resolve_deck
    from sim.env import DECK_26
    torch.set_num_threads(2)
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    report=dict(schema='v25-parser-control-v1',status='evaluating',source_sha256=manifest['archive_sha256'],
        original_v25_mode=original_mode,v25_mode=baseline['mode'],participants=participants,
        gpu_names=[torch.cuda.get_device_name(i)for i in range(2)],expected_matches=80,
        quality_accepted=False,weights_changed=False,original_tournament_changed=False)
    def save():(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save();print(json.dumps(report),flush=True)
    jobs=schedule([p['id']for p in participants])
    def worker(index):
        device=f'cuda:{index}'
        nets={p['id']:load_network(Path(p['resolved_path']),device,p['mode'])for p in participants}
        modes={p['id']:p['mode']for p in participants}
        cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,DECK_26))
        result=[]
        with (output/f'worker_{index}.jsonl').open('x')as stream:
            for ordinal,job in enumerate(jobs):
                if ordinal%2!=index:continue
                name=job['pair'][0]
                row=run_match(nets[name],nets['v25'],device,modes[name],modes['v25'],job['seed'],job['candidate_side'],cards,spells,150,400000)
                row.update(pair=job['pair'],ordinal=ordinal);result.append(row)
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps(dict(event='CONTROL_MATCH',**row)),flush=True)
        return result
    try:
        with ThreadPoolExecutor(2)as pool:rows=[r for group in pool.map(worker,(0,1))for r in group]
        rows.sort(key=lambda r:r['ordinal'])
        if len(rows)!=80 or [r['ordinal']for r in rows]!=list(range(80)):raise ValueError('Incomplete control schedule')
        pairs={}
        for name in (p['id']for p in participants if p['id']!='v25'):
            sample=[r for r in rows if r['pair']==[name,'v25']]
            if len(sample)!=20:raise ValueError('Incomplete paired opponent comparison')
            pairs[name]={outcome:sum(r['outcome']==outcome for r in sample)for outcome in ('win','loss','draw','unknown')}
            pairs[name]['win_rate']=pairs[name]['win']/20
        report.update(status='completed',matches=rows,pairs=pairs)
        save();print(json.dumps(dict(event='CONTROL_COMPLETE',pairs=pairs,quality_accepted=False)),flush=True)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
    return report


if __name__=='__main__':
    if LAUNCH_CONFIG is None:raise RuntimeError('Prepared pinned configuration required')
    execute(LAUNCH_CONFIG)
