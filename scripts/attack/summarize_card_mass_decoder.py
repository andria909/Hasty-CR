"""Recompute the complete decoder comparison from immutable match records."""
import argparse
import json
from pathlib import Path

from scripts.attack.compare_ppo_pilot_behavior import compare_records
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha


def summarize(evaluation,validation,output,markdown):
    evaluation,validation=Path(evaluation),Path(validation)
    raw=json.loads(evaluation.read_text());p=raw['protocol']
    source=json.loads(validation.read_text())
    if (p['smoke']or p['target_games']!=72 or p['control_replays']!=12
            or p['validation_sha256']!=file_sha(validation)or p['physics_pins']!=physics_pins()
            or p['source_sha256']!=file_sha(evaluation.parent/'driver.py')
            or raw['controls']!=[dict(index=i,all_terminal_metrics_exact=True)for i in range(12)]
            or len(raw['records'])!=72 or len(raw['decoder_counts'])!=72):
        raise ValueError('Complete paired decoder validation with exact control replays required')
    baseline=next(r for r in source['reports']if r['model']['sha256']==p['weights_sha256'])
    if file_sha(baseline['model']['checkpoint'])!=p['weights_sha256']:
        raise ValueError('Compared policy changed')
    comparison=compare_records(baseline['records'],raw['records'])
    if comparison!=raw['comparison']:raise ValueError('Stored paired analysis differs')
    for row in raw['decoder_counts']:
        if not 0<=row['changed_actions']<=row['decisions']:raise ValueError('Invalid decoder counts')
    result=dict(schema='verified-card-mass-decoder-comparison-v1',evaluation_sha256=file_sha(evaluation),
                validation_sha256=file_sha(validation),checkpoint_sha256=p['weights_sha256'],
                games_verified=72,control_replays_exact=12,comparison=comparison,
                changed_actions=sum(r['changed_actions']for r in raw['decoder_counts']),
                decisions=sum(r['decisions']for r in raw['decoder_counts']),
                quality_accepted=False,live_checkpoint_changed=False,training_changed=False,
                scope='Inference ablation on reused development validation, same BC800 weights; not a trained new model')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    all_=comparison['all'];m=all_['metrics']
    lines=['# Выбор карты по суммарной вероятности: полные матчи','',
           'Те же веса BC800 seed402. Меняется только экспериментальный способ выбора действия: '
           'суммировать вероятности легальных клеток каждой карты, сравнить карты и WAIT, '
           'затем взять лучшую клетку выбранной карты. Обучение и реальный бот не изменены.','',
           '72 одинаковых validation-сценария; до опыта 12 контрольных матчей заново воспроизведены '
           'с точным совпадением всех итоговых метрик. Это использованный ранее validation, не независимый test.','',
           '| Показатель | Обычный argmax | Сначала карта |','|---|---:|---:|',
           f"| Победы | {all_['before_wins']}/72 | {all_['after_wins']}/72 |"]
    for key,title in [('empty_spell_elixir','Пустые спеллы, эликсир / матч'),('own_fraction_loss','Потери своих башен, сумма долей HP'),
                      ('fireball_per_game','Fireball / матч'),('unit_hits_per_minute','Цели Fireball / минуту'),
                      ('duration_seconds','Длительность матча, с')]:
        lines.append(f"| {title} | {m[key]['before']:.2f} | {m[key]['after']:.2f} |")
    lines+=['',f"Приобретено {all_['gained_wins']} побед, потеряно {all_['lost_wins']}. "
            f"Изменено {result['changed_actions']}/{result['decisions']} решений на траекториях нового декодера.",'',
            '| Режим | Обычный argmax | Сначала карта |','|---|---:|---:|']
    for task in ('attack','defense'):
        c=comparison['tasks'][task];lines.append(f"| {task} | {c['before_wins']}/{c['games']} | {c['after_wins']}/{c['games']} |")
    lines+=['','Выбор более вероятной группы не гарантирует полезного действия. '
            'Даже положительный результат одного исходника не заменяет обучение и независимую проверку.','']
    Path(markdown).write_text('\n'.join(lines));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('evaluation','validation','output','markdown'):p.add_argument('--'+key,required=True)
    a=p.parse_args();summarize(**vars(a))
