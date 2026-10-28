"""Validate saved test/duel parts without claiming a complete tournament."""
import argparse
import json
from pathlib import Path
from scripts.attack.evaluate_full_candidates import validate_protocol,schedule,sha
from scripts.attack.full_evaluation_recovery import restore_parts,identity,source_identity


def audit(inputs,source,output):
    inputs,source,output=Path(inputs),Path(source),Path(output)
    protocol=json.loads((inputs/'protocol.json').read_text())
    blueprint=validate_protocol(protocol,inputs/'weights');jobs=schedule(protocol,blueprint)
    digest=source_identity(inputs/'source/manifest.json',Path.cwd())
    output.mkdir(parents=True,exist_ok=False)
    parts=restore_parts(source,output,identity(inputs/'protocol.json',digest,jobs),protocol,blueprint)
    result=dict(source_manifest_sha256=digest,verified_parts=len(parts),
        verified_games=sum(len(p['rows'])for p in parts.values()),
        missing_jobs=[j for j in jobs if j['file']not in parts],
        tests={k:v['summary']for k,v in parts.items() if v['kind']=='test'},
        complete=False,quality_accepted=False,scope='Verified saved parts only; no complete tournament or winner')
    (output/'partial_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('inputs','source','output'):p.add_argument('--'+arg,required=True)
    a=p.parse_args();r=audit(a.inputs,a.source,a.output)
    print(json.dumps(dict(verified_parts=r['verified_parts'],verified_games=r['verified_games'],missing_games=sum(j['games']for j in r['missing_jobs']))))
