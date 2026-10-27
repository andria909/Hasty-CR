"""Run the fixed Fireball diagnostic after each fully verified pilot cohort.

Local files only; no API calls, emulator access, optimizer or promotion.
Only one low-priority CPU worker is used at a time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

STUDIES=('air_retention_pilot','live_tactical_reward_pilot','joint_skills_pilot','card_exploration_pilot')


def selected_weights(proof):
    if proof.get('pilot_steps_verified')!=819200:raise ValueError('Verified full pilot budget required')
    models=proof.get('pilot_models',[])
    if len(models)!=8 or len({r['id']for r in models})!=8:raise ValueError('All eight selected pilot weights required')
    for row in models:
        if hashlib.sha256(Path(row['checkpoint']).read_bytes()).hexdigest()!=row['checkpoint_sha256']:
            raise ValueError('Verified selected weight changed')
    return models


def render(result,proof,baseline):
    if result['fixtures']!=baseline['fixtures']:raise ValueError('Fixed validation fixtures changed')
    models={Path(r['checkpoint']).parent.name:r for r in result['models']}
    lines=['# Fireball после выбранных PPO-чекпойнтов','',
        '48фиксированных локальных validation-упражнений. Свободный выбор карты и условный прицел Fireball считаются раздельно. Это не48матчей.','',
        '| Модель | Сама выбрала Fireball | Прицел по≥3юнитам | Пустой прицел |','|---|---:|---:|---:|']
    for model in baseline['models']:
        name=Path(model['checkpoint']).parent.name;s=model['summary']['all'];n=s['fixtures']
        lines.append(f"| {name} · исходная | {s['free_fireballs']}/{n} | {s['conditional_group_hits']}/{n} | {s['conditional_empty']}/{n} |")
    for selected in proof['pilot_models']:
        model=models[selected['id']]
        if model['checkpoint_sha256']!=selected['checkpoint_sha256']:raise ValueError('Diagnostic weight identity differs')
        s=model['summary']['all'];n=s['fixtures']
        lines.append(f"| {selected['id']} | {s['free_fireballs']}/{n} | {s['conditional_group_hits']}/{n} | {s['conditional_empty']}/{n} |")
    lines+=['','Состояния и результаты контрольного поиска совпадают с исходной проверкой balanced/strong. Это development-validation, не независимый финальный test. Модели выбраны полным validation ранее; здесь не меняются веса и не запускаются реальные бои.']
    return '\n'.join(lines)+'\n'


def watch(state_file):
    state_file=Path(state_file);state_file.parent.mkdir(parents=True,exist_ok=True)
    if state_file.exists():raise ValueError('Inspect existing diagnostic worker state first')
    state=dict(status='waiting_verified_pilot_cohorts',finished=[],errors={})
    def save():state_file.write_text(json.dumps(state,indent=2)+'\n')
    save();deadline=time.monotonic()+172800
    try:os.nice(10)
    except OSError:pass
    while time.monotonic()<deadline:
        for stem in STUDIES:
            if stem in state['finished']or stem in state['errors']:continue
            proof_path=Path(f'reports/defense_plan/{stem}_verified_20261027.json')
            if not proof_path.exists():continue
            try:proof=json.loads(proof_path.read_text())
            except(ValueError,OSError):continue
            try:
                models=selected_weights(proof)
                out=Path(f'reports/defense_plan/{stem}_fireball_clusters_20261027.json')
                if out.exists():raise ValueError('Existing diagnostic preserved; inspect before reusing')
                state.update(status='evaluating_verified_weights',current=stem);save()
                command=[str(Path('.venv-live/bin/python').absolute()),'-m','scripts.attack.evaluate_fireball_clusters',
                    '--weights',*[m['checkpoint']for m in models],'--output',str(out)]
                with(state_file.parent/(stem+'.log')).open('a')as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                result=json.loads(out.read_text());baseline=json.loads(Path('reports/defense_plan/fireball_cluster_baselines_20261027.json').read_text())
                markdown=Path(f'reports/{stem}_fireball_clusters_20261027.md')
                if markdown.exists():raise ValueError('Existing Markdown diagnostic preserved')
                markdown.write_text(render(result,proof,baseline))
                state['finished'].append(stem)
            except Exception as exc:
                state['errors'][stem]=dict(type=type(exc).__name__,message=str(exc))
            state['status']='waiting_verified_pilot_cohorts';save()
        if len(state['finished'])+len(state['errors'])==len(STUDIES):
            state['status']='complete'if not state['errors']else'complete_with_errors_for_inspection';save();return
        time.sleep(30)
    state['status']='local_wait_expired_remote_jobs_unchanged';save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state-file',required=True);a=p.parse_args();watch(a.state_file)
