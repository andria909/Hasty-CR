from collections import Counter
import json
import pytest
from scripts.attack.evaluate_continuous_pool import restore_parts,sha


def setup(tmp_path):
    weights=tmp_path/'weights';weights.mkdir();out=tmp_path/'out';out.mkdir()
    p=dict(models=[dict(id=n)for n in('a','b','c','d')],test_seeds=[101],duel_seeds=[202])
    protocol=weights/'protocol.json';protocol.write_text(json.dumps(p))
    rows=[dict(scenario_id=f'{t}::{f}/heldout_random',seed=101,outcome='win')for t in('attack','defense')for f in('hog','giant','air','golem','swarm','bridge')]
    part=weights/'resume__a_test.json';part.write_text(json.dumps(dict(model=dict(id='a'),rows=rows,counts={'win':12})))
    manifest=dict(protocol_sha256=sha(protocol),files={'a_test.json':sha(part)})
    (weights/'resume_manifest.json').write_text(json.dumps(manifest))
    return protocol,p,weights,out,part


def test_reuses_exact_frozen_complete_part(tmp_path):
    protocol,p,weights,out,part=setup(tmp_path)
    result=restore_parts(protocol,p,weights,out)
    assert len(result['a_test.json']['rows'])==12
    assert (out/'a_test.json').read_bytes()==part.read_bytes()


def test_resume_cannot_replace_missing_context_by_a_duplicate(tmp_path):
    protocol,p,weights,out,part=setup(tmp_path);r=json.loads(part.read_text());r['rows'][-1]=dict(r['rows'][0]);part.write_text(json.dumps(r))
    manifest=weights/'resume_manifest.json';m=json.loads(manifest.read_text());m['files']['a_test.json']=sha(part);manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='contexts'):restore_parts(protocol,p,weights,out)


def test_changed_bytes_and_unscheduled_paths_rejected(tmp_path):
    protocol,p,weights,out,part=setup(tmp_path);part.write_text(part.read_text()+' ')
    with pytest.raises(ValueError,match='changed'):restore_parts(protocol,p,weights,out)
    manifest=weights/'resume_manifest.json';m=json.loads(manifest.read_text());m['files']={'../other.json':'unused'};manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='outside'):restore_parts(protocol,p,weights,out)
