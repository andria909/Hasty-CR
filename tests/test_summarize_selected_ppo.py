import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts.defense.summarize_selected_ppo import verify,summarize_split,sha


def fixture(tmp_path):
    package=tmp_path/'package';package.mkdir();prepared=tmp_path/'prepared';prepared.mkdir()
    collected=tmp_path/'collected';collected.mkdir();weights=prepared/'selected_dataset';weights.mkdir()
    protocol=json.loads(Path('data/strategy/evaluation_protocol.json').read_text())
    frozen=json.loads(Path('data/defense/evaluation_protocol.json').read_text())
    members={name:Path(name).read_bytes()for name in ('data/strategy/evaluation_protocol.json','data/defense/evaluation_protocol.json','data/defense/baselines.json')}
    with tarfile.open(package/'source.tar.gz','w:gz')as stream:
        for name,raw in members.items():
            entry=tarfile.TarInfo(name);entry.size=len(raw);stream.addfile(entry,io.BytesIO(raw))
    manifest=dict(archive_sha256=sha(package/'source.tar.gz'),files={name:hashlib.sha256(raw).hexdigest()for name,raw in members.items()},
                  training_source_archive_sha256='training',baseline_sha256={name:name+'-hash'for name in frozen['baseline_ids']})
    (package/'manifest.json').write_text(json.dumps(manifest));(weights/'policy.pt').write_bytes(b'policy')
    selection=dict(schema='selected-full-ppo-evaluation-input-v1',test_used_for_selection=False,training_steps=2048000,training_update=2000,
        selected_sha256=sha(weights/'policy.pt'),mode='defense',num_scalars=306,source_sha256='training',policy_optimizer_finite=True,profile='baseline')
    (weights/'selection.json').write_text(json.dumps(selection))
    config=dict(policy_sha256=sha(weights/'policy.pt'),selection_sha256=sha(weights/'selection.json'),manifest_sha256=sha(package/'manifest.json'),
                protocol_sha256=hashlib.sha256(members['data/strategy/evaluation_protocol.json']).hexdigest())
    (prepared/'preparation_receipt.json').write_text(json.dumps(dict(kernel='owner/eval',config=config)))
    reports={}
    for split in ('validation','test'):
        records=[]
        for sid in protocol['scenario_ids'][split]:
            for seed in protocol['seeds'][split]:
                task,scenario=sid.split('::',1)
                records.append(dict(strategy_scenario_id=sid,scenario_id=scenario,seed=seed,split=split,task=task,
                    coordinate_mode='physical_live',geometry='legacy_v1',difficulty='simple',lane='left',terminal_reason='cleared',
                    own_tower_hp_loss=dict(left=0,right=0,king=0),own_tower_fraction_loss=dict(left=0,right=0,king=0),
                    own_towers_alive=dict(left=True,right=True,king=True),no_tower_damage=True,elixir_spent=1,
                    goal_kind='spell_finish',successful_spell_finish_v2=None,coverage_flags=[]))
        reports[split+'.json']=dict(records=records,checkpoint_sha256=config['policy_sha256'],protocol_sha256=config['protocol_sha256'],
                                   input_mode='physical_live_306_timer_task_ignored',terminal_attribution_enabled=True)
    modes={r['id']:r['observation_mode']for r in json.loads(members['data/defense/baselines.json'])['checkpoints']}
    matches=[dict(baseline_id=b,baseline_sha256=b+'-hash',seed=s,candidate_side=seat,result='win',baseline_mode=modes[b],candidate_mode='physical_live')
             for b in frozen['baseline_ids']for s in frozen['paired_seeds']for seat in (1,-1)]
    reports['duels.json']=dict(candidate_sha256=config['policy_sha256'],head_to_head=matches)
    for name,value in reports.items():(collected/name).write_text(json.dumps(value))
    result=dict(schema='selected-ppo-final-evaluation-v1',status='completed',selected_sha256=config['policy_sha256'],selection_sha256=config['selection_sha256'],
        source_sha256=manifest['archive_sha256'],training_source_sha256='training',training_steps=2048000,mode='defense',profile='baseline',
        parts={name:dict(sha256=sha(collected/(name+'.json')))for name in ('validation','test','duels')})
    (collected/'result.json').write_text(json.dumps(result))
    def receipt():
        (collected/'collection_receipt.json').write_text(json.dumps(dict(kernel='owner/eval',remote_status='COMPLETE',complete=True,
            files={name:dict(sha256=sha(collected/name))for name in ('result.json',*reports)})))
    receipt();return collected,prepared,package,receipt


def test_full_summary_recomputes_quality_and_preserves_unknown_spell_attribution(tmp_path):
    collected,prepared,package,_=fixture(tmp_path);result=verify(collected,prepared,package)
    assert result['test']['defense']['overall']['episodes']==708
    assert result['test']['attack']['spell_confirmed_finish']['unknown']==96
    assert all(r['games']==20 and r['wins']==20 for r in result['opponents'].values())
    assert result['quality_accepted']is False


def test_duplicate_context_rejected_even_after_transport_and_internal_hash_updates(tmp_path):
    collected,prepared,package,receipt=fixture(tmp_path)
    p=collected/'test.json';r=json.loads(p.read_text());r['records'][0]=r['records'][1];p.write_text(json.dumps(r))
    p=collected/'result.json';r=json.loads(p.read_text());r['parts']['test']['sha256']=sha(collected/'test.json');p.write_text(json.dumps(r));receipt()
    with pytest.raises(ValueError,match='duplicate'):verify(collected,prepared,package)


def test_tower_metric_contradiction_and_invalid_script_are_rejected():
    row=dict(task='defense',no_tower_damage=True,own_tower_hp_loss=dict(left=1,right=0,king=0),elixir_spent=1)
    with pytest.raises(ValueError,match='contradicts'):summarize_split([row])
    row['terminal_reason']='invalid_script_play_timeout'
    with pytest.raises(ValueError,match='Invalid scripted'):summarize_split([row])


def test_unknown_opponent_outcome_does_not_confirm_80_percent_target(tmp_path):
    collected,prepared,package,receipt=fixture(tmp_path)
    p=collected/'duels.json';r=json.loads(p.read_text());baseline=r['head_to_head'][0]['baseline_id'];r['head_to_head'][0]['result']='unknown';p.write_text(json.dumps(r))
    p=collected/'result.json';r=json.loads(p.read_text());r['parts']['duels']['sha256']=sha(collected/'duels.json');p.write_text(json.dumps(r));receipt()
    result=verify(collected,prepared,package)
    assert result['opponents'][baseline]['observed_80_percent_target_met']is None
    assert result['opponents'][baseline]['unknown']==1


def test_zero_hog_damage_with_incomplete_ledger_remains_unknown():
    row=dict(task='attack',goal_kind='hog_tower_damage',hog_target_tower_damage_v2=0,hog_damage_ledger_complete_v2=False)
    result=summarize_split([row])['attack']['hog_confirmed_hit']
    assert result['unknown']==1 and result['known']==0 and result['rate']is None


def test_observed_simulation_win_is_separate_from_unknown_acceptance_result(tmp_path):
    collected,prepared,package,receipt=fixture(tmp_path)
    p=collected/'duels.json';r=json.loads(p.read_text())
    for row in r['head_to_head']:row.update(result='unknown',observed_result='win')
    p.write_text(json.dumps(r));p=collected/'result.json';r=json.loads(p.read_text());r['parts']['duels']['sha256']=sha(collected/'duels.json');p.write_text(json.dumps(r));receipt()
    result=verify(collected,prepared,package)
    for entry in result['opponents'].values():
        assert entry['unknown']==20 and entry['observed_80_percent_target_met']is None
        assert entry['observed_simulation']['wins']==20 and not entry['observed_simulation']['acceptance_confirmed']
