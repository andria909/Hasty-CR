"""Recover completed parts after a strictly bounded report-validator correction."""
import argparse
import ast
import base64
import io
import json
from pathlib import Path
import shutil
import tarfile
from scripts.kaggle.prepare_full_candidate_evaluation import unpack,prepare,sha
from scripts.attack.evaluate_full_candidates import validate_protocol,schedule
from scripts.attack.full_evaluation_recovery import identity,restore_parts,commit_part,write_json

CHECKER='scripts/attack/verify_full_candidate_evaluation.py'
OLD="            require(set(row['rejected_commands']) <= {'1', '-1'} and all(nonnegative(v, True) for v in row['rejected_commands'].values()), 'Invalid rejected commands')"
NEW="""            # In-memory duel counters use integer sides; JSON encodes the same keys as strings.
            rejected=row['rejected_commands']
            require(all((type(k)is int and k in (1,-1)) or (type(k)is str and k in ('1','-1')) for k in rejected)
                    and len({str(k)for k in rejected})==len(rejected)
                    and all(nonnegative(v,True)for v in rejected.values()), 'Invalid rejected commands')"""


def corrected_checker(old,current):
    if old.count(OLD)!=1 or old.replace(OLD,NEW)!=current:
        raise ValueError('Only the audited rejected-side validator correction is allowed')
    return current


def build(original,collected,work,output,owner='vertel0'):
    original,collected,work,output=map(Path,(original,collected,work,output))
    work.mkdir(parents=True,exist_ok=False)
    oldprep=json.loads((original/'preparation.json').read_text())
    if sha(original/'job/train.py')!=oldprep['bootstrap_sha256']:raise ValueError('Old bootstrap changed')
    inputs,project=unpack(oldprep['config'],original/'upload_dataset',work)
    old_manifest=inputs/'source/manifest.json';manifest=json.loads(old_manifest.read_text())
    protocol_path=inputs/'protocol.json';protocol=json.loads(protocol_path.read_text())
    blueprint=validate_protocol(protocol,inputs/'weights');jobs=schedule(protocol,blueprint)
    old_identity=identity(protocol_path,sha(old_manifest),jobs)
    validated=work/'old_validated';validated.mkdir()
    parts=restore_parts(collected,validated,old_identity,protocol,blueprint)
    new_checker=corrected_checker((project/CHECKER).read_text(),Path(CHECKER).read_text())
    source=work/'corrected_source';source.mkdir()
    with tarfile.open(source/'source.tar.gz','w:gz') as archive:
        for name,digest in sorted(manifest['files'].items()):
            if sha(project/name)!=digest:raise ValueError('Old source differs: '+name)
            content=new_checker.encode() if name==CHECKER else (project/name).read_bytes()
            info=tarfile.TarInfo(name);info.size=len(content);archive.addfile(info,io.BytesIO(content))
    updated=json.loads(json.dumps(manifest));updated['files'][CHECKER]=sha(CHECKER)
    updated['archive_sha256']=sha(source/'source.tar.gz')
    write_json(source/'manifest.json',updated)
    new_identity=identity(protocol_path,sha(source/'manifest.json'),jobs)
    resume=work/'resume';resume.mkdir();write_json(resume/'run_identity.json',new_identity)
    for job in jobs:
        if job['file']in parts:commit_part(resume,job,parts[job['file']],new_identity)
    migration=dict(schema='report-validator-only-recovery-v1',changed_source_files=[CHECKER],
        old_source_manifest_sha256=sha(old_manifest),new_source_manifest_sha256=sha(source/'manifest.json'),
        protocol_sha256=sha(protocol_path),reused_games=sum(len(p['rows'])for p in parts.values()),
        parts={name:dict(old_sha256=sha(collected/name),new_sha256=sha(resume/name)) for name in parts},
        simulation_and_weights_unchanged=True,reason='Accept int side keys before JSON serialization; no game logic changes')
    # Re-serialization is checked byte-for-byte; no result modifications permitted.
    if any(v['old_sha256']!=v['new_sha256'] for v in migration['parts'].values()):raise ValueError('Result bytes changed')
    prepare(inputs,source,output,owner,'hastycr-full-eval-recovery-1028')
    prep=json.loads((output/'preparation.json').read_text());config=prep['config']
    archive=output/'evaluation_inputs.tar.gz'
    with tarfile.open(archive) as stream:payload={m.name:stream.extractfile(m).read() for m in stream.getmembers()}
    for path in resume.iterdir():payload['resume/'+path.name]=path.read_bytes()
    payload['recovery_migration.json']=(json.dumps(migration,indent=2)+'\n').encode()
    payload['original_source_manifest.json']=old_manifest.read_bytes()
    for name in parts:payload['original_receipts/'+name+'.receipt.json']=(collected/(name+'.receipt.json')).read_bytes()
    with tarfile.open(archive,'w:gz') as stream:
        for name,content in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(content);stream.addfile(info,io.BytesIO(content))
    import hashlib
    config['files']={name:hashlib.sha256(content).hexdigest()for name,content in sorted(payload.items())}
    transport=output/'upload_dataset/evaluation_inputs.b64';transport.write_bytes(base64.b64encode(archive.read_bytes()))
    config.update(archive_sha256=sha(archive),transport_sha256=sha(transport))
    path=output/'job/train.py';tree=ast.parse(path.read_text());changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='run' and node.args and isinstance(node.args[0],ast.List):
            if any(isinstance(v,ast.Constant) and v.value=='scripts.attack.evaluate_full_candidates'for v in node.args[0].elts):
                node.args[0].elts += [ast.Constant('--resume'),ast.parse("str(inputs / 'resume')",mode='eval').body];changed+=1
    if changed!=1:raise ValueError('One recovery evaluator command required')
    tree.body[-1].value.args=[ast.parse(repr(config),mode='eval').body];ast.fix_missing_locations(tree);path.write_text(ast.unparse(tree)+'\n')
    prep.update(config=config,bootstrap_sha256=sha(path),recovery=migration)
    (output/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=prep['kernel'],reused_games=migration['reused_games'],remaining_games=prep['total_games']-migration['reused_games'],submitted=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('original','collected','work','output'):p.add_argument('--'+arg,required=True)
    p.add_argument('--owner',default='vertel0');a=p.parse_args()
    print(json.dumps(build(a.original,a.collected,a.work,a.output,a.owner)))
