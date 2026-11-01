"""Verify a complete local behavior cohort and compare paired outcomes.

Local diagnostics never authorize full training or replace full-match validation.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from scripts.kaggle.local_artifact_transfer import file_sha


def summarize(root, output, markdown):
    root=Path(root);receipt=json.loads((root/'result.json').read_text())
    if receipt.get('schema')!='verified-bank-behavior-development-v1' or receipt.get('episodes')!=384 or receipt.get('models')!=8:
        raise ValueError('Complete eight-model behavior cohort required')
    expected={(lane,bank,hand,delay)for lane in ('left','right')for bank in (3000,6000,10000)
              for hand in ('musketeer','fireball')for delay in (1350,2025)}
    models={};physics=None
    for wave in ('none','opposite'):
        entry=receipt['reports'][wave];path=Path(entry['path'])
        if file_sha(path)!=entry['sha256']:raise ValueError('Behavior artifact changed')
        report=json.loads(path.read_text())
        if report['wave_mode']!=wave or len(report['models'])!=8:raise ValueError('Wrong wave/model coverage')
        if file_sha(report['source_snapshot'])!=report['source_sha256']:raise ValueError('Driver snapshot changed')
        if physics is not None and physics!=report['physics_pins']:raise ValueError('Different physics across conditions')
        physics=report['physics_pins']
        for model in report['models']:
            name=Path(model['checkpoint']).parent.name
            key=lambda r:(r['lane'],r['bank'],r['hand'],r['command_delay_ms'])
            if len(model['rows'])!=24 or {key(r)for r in model['rows']}!=expected:
                raise ValueError('Incomplete or duplicate contexts')
            if any(r['seed']!=354 for r in model['rows']):raise ValueError('Local seed changed')
            item=models.setdefault(name,dict(id=name,checkpoint=model['checkpoint'],sha256=model['checkpoint_sha256'],conditions=[]))
            if item['sha256']!=model['checkpoint_sha256']or file_sha(model['checkpoint'])!=item['sha256']:
                raise ValueError('Checkpoint changed across conditions')
            for delay in (1350,2025):
                rows=[r for r in model['rows']if r['command_delay_ms']==delay]
                item['conditions'].append(dict(wave=wave,delay_ms=delay,episodes=len(rows),
                    safe=sum(r['clear_without_damage']for r in rows),
                    mean_damage=sum(r['total_damage']for r in rows)/len(rows),
                    mean_empty_elixir=sum(r['spell_economy']['empty_spell_elixir']for r in rows)/len(rows),
                    under_threat_low_bank_fraction=sum(r['elixir_time']['under_threat_below4_ms']for r in rows)/
                        max(1,sum(r['elixir_time']['under_threat_ms']for r in rows))))
    names={f'{case}_s{seed}'for case in ('retain01','retain001','retain0','zero01')for seed in (392,393)}
    if set(models)!=names:raise ValueError('Wrong candidate identities')
    groups=defaultdict(list)
    for name,item in models.items():
        if len(item['conditions'])!=4:raise ValueError('Missing paired conditions')
        groups[name.rsplit('_s',1)[0]].extend(item['conditions'])
    ranking=[]
    for name,rows in groups.items():
        ranking.append(dict(case=name,safe=sum(r['safe']for r in rows),episodes=sum(r['episodes']for r in rows),
            worst_condition_safe=min(r['safe']for r in rows),
            mean_damage=sum(r['mean_damage']for r in rows)/len(rows),
            mean_empty_elixir=sum(r['mean_empty_elixir']for r in rows)/len(rows)))
    ranking.sort(key=lambda r:(-r['worst_condition_safe'],-r['safe'],r['mean_damage']))
    result=dict(schema='bank-behavior-local-summary-v1',kernel=receipt['kernel'],episodes_verified=384,
        receipt_sha256=file_sha(root/'result.json'),models=list(models.values()),ranking=ranking,
        full_training_accepted=False,independent_test=False)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Поведение прежних BC-кандидатов','',
        '384 локальных эпизода: 8 весов × 12 повторно используемых сцен × 2 задержки × 2 режима волн. Это не игровой винрейт.','',
        '| Вариант | Защит без урона | Худшая комбинация seed/волны/задержки | Средний урон башням | Пустые спеллы, эликсир |',
        '|---|---:|---:|---:|---:|']
    for r in ranking:
        lines.append(f"| {r['case']} | {r['safe']}/{r['episodes']} | {r['worst_condition_safe']}/12 | {r['mean_damage']:.0f} | {r['mean_empty_elixir']:.2f} |")
    lines+=['','Каждая строка объединяет два seed обучения; игровые сцены между ними повторяются. '
            'Сортировка: худший результат по отдельному условию, затем общая защита без урона и величина потерь. '
            'Для большого обучения необходим полный validation по всем семействам.','']
    Path(markdown).write_text('\n'.join(lines));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','output','markdown'):p.add_argument('--'+name,required=True)
    a=p.parse_args();summarize(a.root,a.output,a.markdown)
