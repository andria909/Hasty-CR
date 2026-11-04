import json
from pathlib import Path

import pytest

from scripts.kaggle import resume_full_boundary as module


def test_original_budget_and_schedules_survive_command_adaptation():
    original=['python','-c','trainer','--init','parent.pt','--output','old',
              '--updates','4000','--target-environment-steps','4096000','--eval-every','250']
    result=module.resumed_command(original,'boundary.pt','new')
    assert module.flag(result,'--updates')=='4000'
    assert module.flag(result,'--target-environment-steps')=='4096000'
    assert module.flag(result,'--eval-every')=='250'
    assert module.flag(result,'--resume')=='boundary.pt'and'--init'not in result
    assert module.flag(original,'--init')=='parent.pt'
    with pytest.raises(ValueError,match='duplicated'):
        module.resumed_command(original+['--init','second.pt'],'boundary','new')


def test_historical_guard_wins_over_higher_unguarded_new_score():
    old=dict(update=250,validation=dict(selection_score=.4,worst_archetype_points=.1))
    new=dict(update=500,validation=dict(selection_score=.8,worst_archetype_points=0))
    chosen,guarded=module.selected_boundary([old,new])
    assert chosen==old and guarded
    assert module.selected_boundary([old,dict(old,update=500)])[0]['update']==250


def test_metrics_join_requires_next_update_and_preserves_partial_last_line(tmp_path):
    p=tmp_path/'metrics';p.write_text('{"update":251}\n{"update":252')
    assert module.read_metrics(p,start=250)==[dict(update=251)]
    with pytest.raises(ValueError,match='gap'):module.read_metrics(p,start=249)
    p.write_text('{"update":251}\nbroken\n')
    with pytest.raises(ValueError,match='interior'):module.read_metrics(p,start=250)


def test_tampered_or_escaping_source_cannot_be_loaded(tmp_path):
    p=tmp_path/'config.json';p.write_text('{}')
    spec=dict(source=str(tmp_path),checkpoint='update_000001.pt',files={'config.json':module.sha(p)})
    with pytest.raises(ValueError,match='Missing'):module.checked_inputs(spec)
    spec['files']={'../outside':'no'}
    with pytest.raises(ValueError,match='Unsafe'):module.checked_inputs(spec)
    spec['files']={'config.json':'wrong'}
    with pytest.raises(ValueError,match='changed'):module.checked_inputs(spec)


def state_fixture(tmp_path,complete=False):
    source=tmp_path/'source';source.mkdir()
    rows=[dict(update=1,step=16,target_environment_steps=32,
               validation=dict(selection_score=.4,worst_archetype_points=.2))]
    if complete:rows.append(dict(update=2,step=32,target_environment_steps=32))
    (source/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
    (source/'config.json').write_text('{}')
    (source/'training_result.json').write_text(json.dumps(dict(target_met=complete,step=32 if complete else 16)))
    (source/'update_000001.pt').write_bytes(b'historical')
    if complete:(source/'update_000002.pt').write_bytes(b'last')
    paths={p.name:p for p in source.iterdir()}
    state=dict(paths=paths,config=dict(updates=2),rows=rows,update=len(rows),step=rows[-1]['step'],
               target=32,per_update=16,technical=True)
    spec=dict(checkpoint=f'update_{len(rows):06d}.pt',command=['python','--init','parent','--output','old'])
    return state,spec


def test_completed_sibling_never_starts_optimizer(monkeypatch,tmp_path):
    state,spec=state_fixture(tmp_path,complete=True)
    monkeypatch.setattr(module,'audit',lambda s:state)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('Completed sibling retrained'))
    output=tmp_path/'reused';report=module.execute(spec,output)
    assert report['total_budget_met']and report['new_steps']==0
    assert report['completed_sibling_reused']and not report['optimizer_process_started']
    assert report['selected_checkpoint']=='update_000001.pt'
    assert(output/'previous_metrics.jsonl').read_bytes()==state['paths']['metrics.jsonl'].read_bytes()


def test_incomplete_resume_combines_logs_and_keeps_earlier_better_checkpoint(monkeypatch,tmp_path):
    from types import SimpleNamespace
    state,spec=state_fixture(tmp_path)
    monkeypatch.setattr(module,'audit',lambda s:state)
    def run(command,**kwargs):
        output=Path(module.flag(command,'--output'));output.mkdir()
        (output/'metrics.jsonl').write_text(json.dumps(dict(update=2,step=32,target_environment_steps=32,
            validation=dict(selection_score=.3,worst_archetype_points=.2)))+'\n')
        (output/'training_result.json').write_text(json.dumps(dict(target_met=True,step=32)))
        (output/'update_000002.pt').write_bytes(b'new')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',run)
    output=tmp_path/'continued';report=module.execute(spec,output)
    assert report['total_budget_met']and report['new_steps']==16
    assert report['selected_checkpoint']=='update_000001.pt'
    assert [r['update']for r in module.read_metrics(output/'metrics.jsonl')]==[1,2]
    assert module.read_metrics(output/'resume_segment_metrics.jsonl',start=1)[0]['update']==2
    assert len(module.read_metrics(state['paths']['metrics.jsonl']))==1


def test_nonzero_process_exit_cannot_claim_completed_budget(monkeypatch,tmp_path):
    from types import SimpleNamespace
    state,spec=state_fixture(tmp_path)
    monkeypatch.setattr(module,'audit',lambda s:state)
    def run(command,**kwargs):
        Path(module.flag(command,'--output')).mkdir()
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(module.subprocess,'run',run)
    report=module.execute(spec,tmp_path/'failed')
    assert not report['total_budget_met']and report['new_steps']==0
