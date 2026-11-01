"""Report optimization diagnosis without promoting isolated-skill policies."""
import json
from pathlib import Path
from scripts.kaggle.local_artifact_transfer import file_sha


def summarize(verified,output,markdown):
    verified=Path(verified);proof=json.loads(verified.read_text())
    if (proof.get('schema')!='verified-bank-fit-depth-v1'or proof['supervised_updates_verified']!=35200
            or proof['actual_cuda_verified']is not True):raise ValueError('Complete CUDA fitting study required')
    expected={f'{mode}{steps}_s{seed}'for mode in ('mixed','isolated')for steps in (800,8000)for seed in (402,403)}
    if {r['id']for r in proof['runs']}!=expected or len(proof['runs'])!=8:raise ValueError('All eight runs required')
    rows=[]
    for r in proof['runs']:
        by_skill=r['common_data_score']['by_skill'];new=by_skill['onpolicy_defense_response']
        rows.append(dict(id=r['id'],case=r['case'],seed=r['seed'],
            train_correct=new['train']['successful_choices'],train_examples=new['train']['examples'],
            heldout_correct=new['local_heldout']['successful_choices'],heldout_examples=new['local_heldout']['examples'],
            other_skills_macro_heldout=sum(v['local_heldout']['fraction']for k,v in by_skill.items()if k!='onpolicy_defense_response')/4))
    result=dict(schema='bank-fit-depth-label-summary-v1',kernel=proof['kernel'],verified_sha256=file_sha(verified),
        rows=rows,supervised_updates_verified=35200,full_training_accepted=False,
        scope='Optimization diagnostic; exact local labels, not full-game outcomes; isolated skills cannot be promoted')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Обучаемость защиты: длительность и смешивание навыков','',
        'Все веса проверены на одном полном наборе. Оптимизатор видит только группы352/353;354отложена. '
        'Изолированная защита проверяет обучаемость, а не готовность играть.','',
        '| Набор | Обновления | Seed | Защита TRAIN | Защита отложенная | Остальные навыки, среднее |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        label='Смесь'if r['case']['dataset']=='control'else'Только защита'
        lines.append(f"| {label} | {r['case']['updates']} | {r['seed']} | {r['train_correct']}/{r['train_examples']} | {r['heldout_correct']}/{r['heldout_examples']} | {r['other_skills_macro_heldout']:.1%} |")
    lines+=['','Последний столбец — равное среднее четырёх прежних навыков, не игровой винрейт. '
        'Большое PPO и live этим результатом не разрешаются; для смешанных кандидатов нужны полные матчи.','']
    Path(markdown).write_text('\n'.join(lines));return result
