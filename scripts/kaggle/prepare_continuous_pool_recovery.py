"""Recover only missing games, preserving the original models and physics."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import tarfile

from scripts.attack.evaluate_continuous_pool import restore_parts
from sim.pathfind import _cache_field


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources(original):
    with tarfile.open(original/'dataset/source.tar.gz')as archive:
        old_path=archive.extractfile('sim/pathfind.py').read().decode()
    new_path=Path('sim/pathfind.py').read_text()
    reverse=new_path.replace('from threading import Lock\n','')
    reverse=reverse.replace('_FIELD_CACHE_LOCK=Lock()\n\n\n'+inspect.getsource(_cache_field),'')
    reverse=reverse.replace('    with _FIELD_CACHE_LOCK:\n        cached = _FIELD_CACHE.get(key)','    cached = _FIELD_CACHE.get(key)')
    reverse=reverse.replace('        return _cache_field(key,field)\n','        _FIELD_CACHE[key] = field\n        return field\n')
    reverse=reverse.replace('    return _cache_field(key,field)\n',
        '    if len(_FIELD_CACHE) >= 512:\n        _FIELD_CACHE.pop(next(iter(_FIELD_CACHE)))\n    _FIELD_CACHE[key] = field\n    return field\n')
    if reverse!=old_path:raise ValueError('Pathfinding changes exceed the reviewed cache fix')
    with tarfile.open(original/'dataset/extensions.tar.gz')as archive:
        old_eval=ast.parse(archive.extractfile('scripts/attack/evaluate_continuous_pool.py').read().decode())
    new_eval=ast.parse(Path('scripts/attack/evaluate_continuous_pool.py').read_text())
    def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    for name in ('load','choose','duel'):
        if ast.dump(function(old_eval,name))!=ast.dump(function(new_eval,name)):raise ValueError('Game inference/duel mechanics changed')
    workers=[function(function(tree,'evaluate'),'worker')for tree in(old_eval,new_eval)]
    loops=[[n for n in worker.body if isinstance(n,ast.For)]for worker in workers]
    if [ast.dump(ast.Module(body=n.body,type_ignores=[]))for n in loops[0]]!=[ast.dump(ast.Module(body=n.body,type_ignores=[]))for n in loops[1]]:
        raise ValueError('Per-game evaluation loop changed')


def prepare(original,output):
    original=Path(original);check_sources(original);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    p=json.loads((original/'preparation.json').read_text());c=p['config']
    if sha(original/'job/train.py')!=p['bootstrap_sha256']:raise ValueError('Original bootstrap changed')
    dataset=output/'dataset';shutil.copytree(original/'dataset',dataset);job=output/'job';job.mkdir()
    protocol=json.loads((dataset/'protocol.json').read_text())
    parts={}
    for source in(original/'collected').glob('*.json'):
        if source.name.endswith(('_test.json','_duels.json')):
            shutil.copyfile(source,dataset/('resume__'+source.name));parts[source.name]=sha(source)
    resume=dict(schema='pinned-continuous-pool-resume-v1',source_kernel=p['kernel'],source_kernel_version=1,
        source_remote_status='ERROR',protocol_sha256=sha(dataset/'protocol.json'),files=parts)
    (dataset/'resume_manifest.json').write_text(json.dumps(resume,indent=2)+'\n')
    test=output/'reuse_preflight';test.mkdir();reports=restore_parts(dataset/'protocol.json',protocol,dataset,test)
    games=sum(len(r['rows'])for r in reports.values())
    if games!=420:raise ValueError('Expected exactly420 frozen completed games')
    staging=output/'extension_staging';staging.mkdir()
    with tarfile.open(original/'dataset/extensions.tar.gz')as archive:archive.extractall(staging,filter='data')
    names=set(c['extensions'])|{'sim/pathfind.py'}
    for name in('sim/pathfind.py','scripts/attack/evaluate_continuous_pool.py'):
        target=staging/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    with tarfile.open(dataset/'extensions.tar.gz','w:gz')as archive:
        for name in sorted(names):archive.add(staging/name,arcname=name,recursive=False)
    c['extensions']={name:sha(staging/name)for name in sorted(names)}
    base=json.loads((dataset/'manifest.json').read_text());c['replaced']['sim/pathfind.py']=base['files']['sim/pathfind.py']
    c['pins']['extensions.tar.gz']=sha(dataset/'extensions.tar.gz')
    for name in('resume_manifest.json',*['resume__'+n for n in parts]):c['pins'][name]=sha(dataset/name)
    code=(original/'job/train.py').read_text().rsplit('\nbootstrap(',1)[0]+'\nbootstrap('+repr(c)+')\n'
    compile(code,'recovery.py','exec');(job/'train.py').write_text(code)
    shutil.copyfile(original/'job/kernel-metadata.json',job/'kernel-metadata.json')
    result=dict(kernel=p['kernel'],config=c,bootstrap_sha256=sha(job/'train.py'),submitted=False,
        recovery_of=str(original),reused_games=games,computed_games=600-games,
        cache_only_source_change_checked=True,per_game_core_ast_unchanged=True)
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'partial_reuse_receipt.json').write_text(json.dumps(resume,indent=2)+'\n')
    return dict(kernel=p['kernel'],reused_games=games,computed_games=600-games,submitted=False)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--original',required=True);a.add_argument('--output',required=True)
    p=a.parse_args();print(json.dumps(prepare(p.original,p.output)))
