"""Exercise continuation with real validation and reuse of a finished sibling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.kaggle.local_artifact_transfer import file_sha
from scripts.kaggle.preflight_global_full_pair import smoke_args,set_flag
from scripts.kaggle.resume_full_boundary import read_metrics,selected_boundary
from scripts.kaggle.preflight_full_recovery_collection import COMPARE


def preflight(case,output):
    case=Path(case).absolute();output=Path(output).absolute();output.mkdir(parents=True,exist_ok=False)
    proof=json.loads((case/'child_preflight.json').read_text())
    if not proof.get('technical_fixture')or not proof.get('exact_resume_16_to_32'):
        raise ValueError('Frozen full technical case required')
    project=case/'project';env=dict(os.environ,PYTHONPATH=str(project))
    original=proof['original_full_command']
    whole=set_flag(smoke_args(original,output/'whole',32),'--eval-every',1)
    stopped=set_flag(set_flag(whole,'--output',output/'stopped'),'--wallclock-seconds',1e-9)
    with(output/'initial.log').open('x')as log:
        for command in(whole,stopped):subprocess.run(command,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    incomplete=json.loads((output/'stopped/training_result.json').read_text())
    if incomplete['target_met']or incomplete['step']!=16:raise ValueError('Expected actual interruption at16/32')
    runner=Path(__file__).with_name('resume_full_boundary.py').absolute();reports=[]
    for source_name,target,latest in(('stopped','continued',1),('whole','reused_complete',2)):
        source=output/source_name;rows=read_metrics(source/'metrics.jsonl');selected,_=selected_boundary(rows)
        if selected is None:raise ValueError('Actual validation records required')
        names={'config.json','metrics.jsonl','training_result.json',f'update_{latest:06d}.pt',f"update_{selected['update']:06d}.pt"}
        spec=dict(source=str(source),technical_fixture=True,checkpoint=f'update_{latest:06d}.pt',
            command=whole,files={name:file_sha(source/name)for name in sorted(names)})
        spec_path=output/(target+'_spec.json');spec_path.write_text(json.dumps(spec,indent=2)+'\n')
        with(output/(target+'_adapter.log')).open('x')as log:
            subprocess.run([sys.executable,str(runner),'--spec',str(spec_path),'--output',str(output/target)],
                cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((output/target/'continuation.json').read_text())
        combined=read_metrics(output/target/'metrics.jsonl')
        reference=read_metrics(output/'whole/metrics.jsonl')
        if [r['validation']for r in combined]!=[r['validation']for r in reference]:
            raise ValueError('Combined real validation differs')
        if not report['total_budget_met']or report['total_steps']!=32:raise ValueError('Combined budget incomplete')
        if target=='reused_complete'and (report['optimizer_process_started']or report['new_steps']or report['command']is not None):
            raise ValueError('Finished sibling trained twice')
        if target=='continued'and (report['new_steps']!=16 or not report['optimizer_process_started']):
            raise ValueError('Wrong remaining budget')
        for name,digest in spec['files'].items():
            if file_sha(source/name)!=digest:raise ValueError('Recovery changed source artifacts')
        reports.append(report)
        print(json.dumps(dict(target=target,budget_verified=True,new_steps=report['new_steps'],
            selected_checkpoint=report['selected_checkpoint'])),flush=True)
    helper=Path(__file__).resolve().parents[1]/'attack/boundary_state.py'
    with(output/'compare.log').open('x')as log:
        subprocess.run([sys.executable,'-c',COMPARE,str(helper),str(output/'whole/update_000002.pt'),
            str(output/'continued/update_000002.pt'),str(output/'stopped/update_000001.pt'),str(output/'comparison.json')],
            cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    expected,_=selected_boundary(read_metrics(output/'whole/metrics.jsonl'))
    if any(r['selected_checkpoint']!=f"update_{expected['update']:06d}.pt"for r in reports):
        raise ValueError('Historical selection changed')
    games=sum(len(r.get('validation',{}).get('records',[]))for name,start in
        (('whole/metrics.jsonl',0),('stopped/metrics.jsonl',0),('continued/resume_segment_metrics.jsonl',1))
        for r in read_metrics(output/name,start))
    if games!=48:raise ValueError('Actual validation coverage differs')
    result=dict(schema='full-boundary-adapter-preflight-v1',technical_fixture=True,
        actual_cuda=False,training_submitted=False,submission_allowed=False,quality_accepted=False,
        case=str(case),adapter_sha256=file_sha(runner),validation_games=games,
        historical_selection_equal=True,source_artifacts_unchanged=True,
        state_comparison=json.loads((output/'comparison.json').read_text()),runs=reports)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();preflight(a.case,a.output)
