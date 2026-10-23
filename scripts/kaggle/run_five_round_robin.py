"""Pinned five-policy round robin: ten pairs, ten paired seeds each, no training."""
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor

LAUNCH_CONFIG = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule(names):
    if len(names) != 5 or len(set(names)) != 5:
        raise ValueError('Exactly five unique participants required')
    return [dict(pair=[a,b],seed=seed,candidate_side=side)
            for a,b in itertools.combinations(names,2)
            for seed in range(710001,710011) for side in (1,-1)]


def standings(rows, names):
    totals={name:dict(matches=0,wins=0,losses=0,draws=0,unknown=0)for name in names}
    matrix={a:{b:dict(matches=0,wins=0,losses=0,draws=0,unknown=0)for b in names if b!=a}for a in names}
    for row in rows:
        a,b=row['pair'];result=row['outcome']
        if result not in ('win','loss','draw','unknown'):raise ValueError('Unknown outcome label')
        for own,other,outcome in ((a,b,result),(b,a,{'win':'loss','loss':'win','draw':'draw','unknown':'unknown'}[result])):
            for entry in (totals[own],matrix[own][other]):
                entry['matches']+=1
                entry[{'win':'wins','loss':'losses','draw':'draws','unknown':'unknown'}[outcome]]+=1
    for entry in totals.values():
        entry['win_rate_all_matches']=entry['wins']/entry['matches']if entry['matches']else None
        entry['score']=entry['wins']+.5*entry['draws']
        entry['score_rate_all_matches']=entry['score']/entry['matches']if entry['matches']else None
    # Predetermined descriptive ranking: points; tied scores remain tied.
    best=max((e['score']for e in totals.values()),default=0)
    return dict(totals=totals,matrix=matrix,leaders=[n for n in names if totals[n]['score']==best],
                ranking_rule='(wins + 0.5*draws)/80; equal scores remain tied; unknown gives no points',
                ranking_provisional=any(e['unknown']for e in totals.values()))


def skill_policy(network, joint, torch):
    if joint:return network
    class BaseInput(torch.nn.Module):
        def __init__(self):
            super().__init__();self.network=network
        def forward(self,planes,scalars):
            return self.network(planes,scalars[:,:306])
    return BaseInput()


def skill_environment(factory,joint,**options):
    if not joint:options['coordinate_mode']='physical_live'
    return factory(**options)


def prepare_source(source, output, manifest_sha):
    if sha(source/'manifest.json')!=manifest_sha:raise ValueError('Manifest SHA mismatch')
    manifest=json.loads((source/'manifest.json').read_text());files=manifest['files']
    for name in files:
        p=PurePosixPath(name)
        if p.is_absolute()or '..'in p.parts or '\\'in name or str(p)!=name:raise ValueError('Unsafe member')
    output.mkdir(parents=True,exist_ok=False)
    if (source/'source.tar.gz').exists():
        if sha(source/'source.tar.gz')!=manifest['archive_sha256']:raise ValueError('Archive SHA mismatch')
        with tarfile.open(source/'source.tar.gz')as stream:
            members=stream.getmembers()
            if len(members)!=len(files)or{m.name for m in members}!=set(files)or any(not m.isfile()for m in members):raise ValueError('Archive members differ')
            stream.extractall(output,filter='data')
    else:
        original=(source/'source').resolve()
        for name in files:
            p=original/name
            if p.is_symlink()or not p.resolve().is_relative_to(original):raise ValueError('Unsafe source')
            target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for name,digest in files.items():
        if sha(output/name)!=digest:raise ValueError('Source member SHA mismatch')
    return manifest


def execute(config):
    output=Path('/kaggle/working/round_robin');output.mkdir(exist_ok=False)
    def mount(ref):
        owner,slug=ref.split('/');options=[p for p in (Path('/kaggle/input')/slug,Path('/kaggle/input/datasets')/owner/slug)if p.is_dir()]
        if len(options)!=1:raise ValueError('Ambiguous/missing exact mount')
        return options[0]
    weights=mount(config['weights_dataset']);source=mount(config['source_dataset'])
    if sha(weights/'participants.json')!=config['participants_sha256']:raise ValueError('Participant manifest mismatch')
    participants=json.loads((weights/'participants.json').read_text())
    project=output/'project';manifest=prepare_source(source,project,config['manifest_sha256'])
    if manifest['archive_sha256']!='c043a0fa7340c14f29a2f5f27981107e629e9d4741a030c6dab8740ad6d0c798':raise ValueError('Unexpected frozen engine')
    sys.path[:0]=[str(project),str(project/'scripts')]
    import torch
    from sim.eval_defense_duel import run_match,load_network,JOINT_MODE
    from sim.defense_assets import load_defense_assets
    from sim.runner import resolve_deck
    from sim.env import DECK_26
    torch.set_num_threads(2)
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    names=[p['id']for p in participants];jobs=schedule(names)
    for p in participants:
        path=project/p['path']if p['id']=='v25'else weights/p['path']
        if sha(path)!=p['sha256']:raise ValueError('Selected participant SHA mismatch')
        p['resolved_path']=str(path)
    gpu=[torch.cuda.get_device_name(i)for i in range(2)]
    print(json.dumps(dict(event='CUDA_READY',gpu=gpu,total=200,source_sha256=manifest['archive_sha256'],participants=participants)),flush=True)
    completed=[];progress_lock=threading.Lock()
    def worker(index):
        device=torch.device(f'cuda:{index}')
        nets={p['id']:load_network(Path(p['resolved_path']),device,p['mode'])for p in participants}
        modes={p['id']:p['mode']for p in participants}
        cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,DECK_26))
        result=[]
        with (output/f'worker_{index}.jsonl').open('x')as log:
            for ordinal,job in enumerate(jobs):
                if ordinal%2!=index:continue
                a,b=job['pair']
                row=run_match(nets[a],nets[b],device,modes[a],modes[b],job['seed'],job['candidate_side'],cards,spells,150,400000)
                if JOINT_MODE in (modes[a],modes[b])and'joint_task_switch_is_public_heuristic'not in row['quality_flags']:
                    row['quality_flags'].append('joint_task_switch_is_public_heuristic')
                row.update(pair=[a,b],ordinal=ordinal);result.append(row)
                log.write(json.dumps(row)+'\n');log.flush()
                print(json.dumps(dict(event='MATCH_RESULT',worker=index,**row)),flush=True)
                with progress_lock:
                    completed.append(row)
                    pair_rows=[r for r in completed if r['pair']==[a,b]]
                    print(json.dumps(dict(event='PROGRESS',done=len(completed),total=200)),flush=True)
                    if len(pair_rows)==20:
                        print(json.dumps(dict(event='PAIR_COMPLETE',pair=[a,b],done=len(completed),total=200,summary=standings(pair_rows,names)['matrix'][a][b])),flush=True)
        from sim.eval_strategy import evaluate
        from sim.strategy_env import StrategyEnv
        protocol_path=project/'data/strategy/evaluation_protocol.json'
        protocol=json.loads(protocol_path.read_text())
        for ordinal,p in enumerate(participants):
            if ordinal%2!=index:continue
            joint=p['mode']==JOINT_MODE
            def factory(**options):
                return skill_environment(StrategyEnv,joint,**options)
            net=skill_policy(nets[p['id']],joint,torch)
            skill=evaluate(net,device,protocol,'test',env_factory=factory)
            if skill['episodes']!=804:raise ValueError('Incomplete common skill test')
            skill.update(participant=p['id'],checkpoint_sha256=p['sha256'],
                input_mode='joint_309_cells_v2'if joint else'physical_live_306_timer_task_ignored',
                source_sha256=manifest['archive_sha256'],protocol_sha256=sha(protocol_path))
            (output/(p['id']+'_common_skill_test.json')).write_text(json.dumps(skill,indent=2)+'\n')
            print(json.dumps(dict(event='COMMON_SKILL_COMPLETE',participant=p['id'],episodes=skill['episodes'],attack=skill['attack']['overall'],defense=skill['defense'])),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2)as pool:
        futures=[pool.submit(worker,i)for i in range(2)]
        rows=[row for future in futures for row in future.result()]
    rows.sort(key=lambda r:r['ordinal'])
    if len(rows)!=200 or [r['ordinal']for r in rows]!=list(range(200)):raise ValueError('Incomplete tournament')
    report=dict(schema='five-model-round-robin-v1',status='complete',total=200,participants=participants,matches=rows,
        source_sha256=manifest['archive_sha256'],engine_sha256=manifest['files']['sim/match.py'],gpu=gpu,
        delay_ms=150,max_ms=400000,quality_accepted=False,test_used_for_selection=False,
        quality_flags=sorted({f for row in rows for f in row['quality_flags']}),**standings(rows,names))
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    for a,b in itertools.combinations(names,2):
        pair_rows=[r for r in rows if r['pair']==[a,b]]
        print(json.dumps(dict(event='PAIR_COMPLETE',pair=[a,b],done=len(pair_rows),summary=standings(pair_rows,names)['matrix'][a][b])),flush=True)
    print(json.dumps(dict(event='TOURNAMENT_COMPLETE',done=200,total=200,totals=report['totals'],leaders=report['leaders'])),flush=True)


if __name__=='__main__':
    if LAUNCH_CONFIG is None:raise RuntimeError('Embedded launch config required')
    execute(LAUNCH_CONFIG)
