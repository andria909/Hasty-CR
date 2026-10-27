from copy import deepcopy
import pytest
from scripts.attack import report_full_candidate_evaluation as module
from sim.env import DECK_26


def example():
    names=['candidate','balanced','v25']
    pairs=[dict(pair=['candidate','balanced'],counts={'win':2,'loss':0,'draw':0}),
           dict(pair=['candidate','v25'],counts={'win':0,'loss':2,'draw':0}),
           dict(pair=['balanced','v25'],counts={'win':1,'loss':1,'draw':0})]
    for d in pairs:
        points=(d['counts']['win']+.5*d['counts']['draw'])/2
        d['uncertainty']=dict(points=points,interval95=[0.,1.])
    summary=dict(counts=dict(win=1,draw=0,loss=1),points=.5,mean_card_plays={c:1. for c in DECK_26},
                 mean_elixir_overflow=2.,mean_empty_fireballs=0.,mean_cluster_fireballs=1.,mean_rejected_commands=0.)
    row=dict(summary=summary,uncertainty=dict(interval95=[0.,1.]),by_task_family={'defense/air':dict(points=.5)})
    verified=dict(duels=pairs,tests={t:{n:deepcopy(row) for n in names} for t in ('scripted_stress','legal_random_compatibility')})
    return verified,names


def test_standings_reverse_results_for_second_player():
    verified,names=example();result=module.standings(verified,names)
    assert result['totals']['candidate']==dict(win=2,loss=2,draw=0)
    assert result['totals']['balanced']==dict(win=1,loss=3,draw=0)
    assert result['totals']['v25']==dict(win=3,loss=1,draw=0)
    assert result['order']==['v25','candidate','balanced']
    assert result['points']['v25']==.75
    assert result['matrix']['v25']['candidate']['win']==2


def test_missing_or_duplicate_pair_cannot_make_leader_table():
    verified,names=example();verified['duels'][0]=deepcopy(verified['duels'][1])
    with pytest.raises(ValueError,match='round robin'):module.standings(verified,names)


def test_render_keeps_test_sets_separate_and_discloses_failed_guard():
    verified,names=example()
    protocol=dict(models=[dict(id=n,family_guard=n!='candidate') for n in names])
    text=module.render(verified,protocol,{n:n for n in names})
    assert 'Основной тест' in text and 'Дополнительный тест' in text
    assert 'не винрейт в Clash Royale' in text
    assert 'сохранены в сравнении: candidate' in text
    assert all(card in text for card in DECK_26)
    assert 'defense/air' in text and 'Против v25' in text


def test_repeated_report_write_is_idempotent_but_never_overwrites(tmp_path):
    path=tmp_path/'report.md';module.write_consistent(path,'original')
    module.write_consistent(path,'original')
    with pytest.raises(ValueError,match='preserve'):module.write_consistent(path,'changed')
    assert path.read_text()=='original'


def test_finalize_full_synthetic_schedule_reverifies_before_publishing(tmp_path):
    import json
    from scripts.kaggle.prepare_full_candidate_evaluation import prepare
    from scripts.attack.evaluate_full_candidates import schedule,validate_protocol
    from scripts.attack.full_evaluation_recovery import identity,commit_part,write_json
    from tests.test_prepare_full_candidate_evaluation import inputs
    from tests.test_full_candidate_verifier import part
    binding,source=inputs(tmp_path);prepared=tmp_path/'package'
    prep=prepare(binding,source,prepared,'vertel0','hastycr-full-eval-test')
    protocol=json.loads((binding/'protocol.json').read_text());blueprint=validate_protocol(protocol,binding/'weights')
    jobs=schedule(protocol,blueprint);collected=tmp_path/'collected';collected.mkdir()
    source_digest=module.sha(source/'manifest.json');protocol_digest=module.sha(binding/'protocol.json')
    (collected/'source_manifest.json').write_bytes((source/'manifest.json').read_bytes())
    run=identity(binding/'protocol.json',source_digest,jobs);write_json(collected/'run_identity.json',run)
    for job in jobs:commit_part(collected,job,part(job,protocol,blueprint),run)
    write_json(collected/'result.json',dict(schema='full-candidate-independent-evaluation-v1',status='complete',
        protocol_sha256=protocol_digest,blueprint_sha256=protocol['blueprint_sha256'],source_manifest_sha256=source_digest,
        models=protocol['models'],total_games=4200,files={j['file']:module.sha(collected/j['file']) for j in jobs},
        test_used_for_selection=False,quality_accepted=False,live_checkpoint_changed=False))
    receipt=dict(kernel=prep['kernel'],remote_status='COMPLETE',complete=True,transfer_complete=True,
        files={p.name:dict(sha256=module.sha(p)) for p in collected.iterdir()})
    write_json(collected/'collection_receipt.json',receipt)
    report=tmp_path/'reports/result.json'
    result=module.finalize(prepared,binding,collected,report)
    assert result['total_games_verified']==4200 and not result['quality_accepted']
    assert report.with_suffix('.md').exists()
    assert module.finalize(prepared,binding,collected,report)==result
    # Stale/tampered download cannot update or publish another report.
    (collected/jobs[0]['file']).write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):
        module.finalize(prepared,binding,collected,tmp_path/'reports/invalid.json')
    assert not (tmp_path/'reports/invalid.json').exists()
