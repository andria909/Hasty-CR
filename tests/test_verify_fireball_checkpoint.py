import hashlib
import json

import pytest
import torch

from scripts.attack import verify_fireball_checkpoint as audit
from scripts.attack.train_fireball_refinement import KINDS,contexts
from sim.fireball_curriculum import CONTRACT


@pytest.fixture
def collected(tmp_path,monkeypatch):
    parent=tmp_path/'parent.pt';parent.write_bytes(b'original')
    digest=hashlib.sha256(parent.read_bytes()).hexdigest()
    monkeypatch.setattr(audit,'PARENT_SHA',digest)
    prepared=tmp_path/'prepared';(prepared/'job').mkdir(parents=True);(prepared/'dataset').mkdir()
    (prepared/'job/train_fireball.py').write_text('pinned bootstrap')
    (prepared/'dataset/source.tar.gz').write_bytes(b'pinned archive')
    source_sha=audit.sha(prepared/'dataset/source.tar.gz')
    (prepared/'dataset/manifest.json').write_text(json.dumps(dict(archive_sha256=source_sha)))
    (prepared/'preparation.json').write_text(json.dumps(dict(kernel='owner/fireball',parent_sha256=digest,
        bootstrap_sha256=audit.sha(prepared/'job/train_fireball.py'),source_sha256=source_sha)))
    root=tmp_path/'collected';root.mkdir()
    rows=[dict(kind=k,lane=l,seed=s,success=True)for k,l,s in contexts('validation',6)]
    report=dict(rows=rows,episodes=6,successes=6,by_kind={k:dict(episodes=2,successes=2)for k in KINDS})
    config=dict(parent_sha256=digest,fireball_contract=CONTRACT,num_scalars=309,
        method='impact_verified_BC_with_policy_rehearsal',test_used_for_selection=False,
        epochs=1,eval_episodes=6,min_agreement=.9,demonstrations=6)
    blob=dict(config=config,parent_sha256=digest,epoch=1,validation=report,rehearsal_action_agreement=.95,
              state_dict={'scalar.0.weight':torch.zeros(2,309)})
    torch.save(blob,root/'candidate.pt')
    for name,value in [('result.json',dict(status='complete',candidate_available=True,best_epoch=1,best_validation=1.)),
                       ('baseline_validation.json',report),('demonstration_audit.json',dict(accepted={k:2 for k in KINDS},rejected=[]))]:
        (root/name).write_text(json.dumps(value))
    (root/'metrics.jsonl').write_text(json.dumps(dict(epoch=1,validation=report,rehearsal_action_agreement=.95))+'\n')
    receipt=dict(remote_status='COMPLETE',complete=True,kernel='owner/fireball',
                 files={p.name:dict(sha256=audit.sha(p))for p in root.iterdir()})
    (root/'collection_receipt.json').write_text(json.dumps(receipt))
    return root,prepared,parent


def test_accepts_matching_completed_candidate(collected):
    result=audit.verify(*collected)
    assert result['validation_after']['tower_unit']==dict(episodes=2,successes=2)
    assert not result['quality_accepted'] and result['defense_regression_pending']


def test_rejects_modified_artifact(collected):
    root,_,_=collected
    (root/'metrics.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='artifact SHA'):audit.verify(*collected)


def test_rejects_nonterminal_collection(collected):
    root,_,_=collected
    path=root/'collection_receipt.json';r=json.loads(path.read_text());r['remote_status']='RUNNING';path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Complete collection'):audit.verify(*collected)


@pytest.mark.parametrize('change',['nan','shape','selection'])
def test_rejects_invalid_weights_or_selection_even_with_matching_download_sha(collected,change):
    root,_,_=collected
    path=root/'candidate.pt';blob=torch.load(path,weights_only=False)
    if change=='nan':blob['state_dict']['scalar.0.weight'][0,0]=float('nan')
    elif change=='shape':blob['state_dict']['scalar.0.weight']=torch.zeros(2,306)
    else:blob['epoch']=2
    torch.save(blob,path)
    receipt=root/'collection_receipt.json';r=json.loads(receipt.read_text());r['files']['candidate.pt']['sha256']=audit.sha(path);receipt.write_text(json.dumps(r))
    with pytest.raises(ValueError):audit.verify(*collected)
