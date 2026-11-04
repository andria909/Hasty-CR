"""Describe every scheduled checkpoint of verified pilots, without reselecting weights."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from scripts.attack.compare_ppo_pilot_behavior import index_records,FAMILIES
from scripts.attack.decision_behavior import card_usage,economy
from scripts.attack.select_pilot_pool import verify_study
from scripts.kaggle.local_artifact_transfer import file_sha


def summarize(records,reference):
    indexed=index_records(records)
    if indexed.keys()!=index_records(reference).keys():
        raise ValueError('Learning curve changed validation contexts')
    families={f:[r for r in records if r['family_id']==f]for f in FAMILIES}
    usage=card_usage(records)
    return dict(games=len(records),wins=sum(r['outcome']=='win'for r in records),
        defense_games=sum(r['task']=='defense'for r in records),
        defense_wins=sum(r['task']=='defense'and r['outcome']=='win'for r in records),
        worst_family_points=min(mean(1. if r['outcome']=='win'else .5 if r['outcome']=='draw'else 0.
                                    for r in rows)for rows in families.values()),
        by_family_wins={f:sum(r['outcome']=='win'for r in rows)for f,rows in families.items()},
        mean_own_fraction_loss=mean(r['own_fraction_loss']for r in records),
        mean_empty_spell_elixir=mean(r['empty_spell_elixir']for r in records),
        mean_elixir_overflow=mean(r['elixir_overflow']for r in records),
        mean_duration_seconds=mean(r['elapsed_ms']for r in records)/1000,
        card_plays_per_game={c:v['per_game']for c,v in usage.items()},
        card_plays_per_minute={c:v['per_minute']for c,v in usage.items()},
        decision_economy=economy(records))


def audit(roots):
    studies=[]
    for path in roots:
        root=Path(path);proof=verify_study(root)
        if proof['pilot_steps_verified']!=3276800:raise ValueError('Complete eight-run long study required')
        protocol=json.loads((root/'dataset/protocol.json').read_text())
        receipt_path=root/'collected/collection_receipt.json';receipt=json.loads(receipt_path.read_text())
        def checked(name):
            file=root/'collected'/name
            if file_sha(file)!=receipt['files'][name]['sha256']:raise ValueError('Curve evidence changed: '+name)
            return json.loads(file.read_text())if file.suffix=='.json'else [json.loads(s)for s in file.read_text().splitlines()]
        reference=checked('parent_reference.json')['validation']['records'];runs=[]
        for trial in protocol['cases']:
            for seed in protocol['pilot_seeds']:
                name=f"pilot_{trial['id']}_s{seed}";rows=checked(name+'/metrics.jsonl')
                if [r['update']for r in rows]!=list(range(1,401)):raise ValueError('All400 updates required')
                selected=checked(name+'_summary.json');curve=[]
                for row in rows:
                    if 'validation'not in row:continue
                    point=summarize(row['validation']['records'],reference)
                    point.update(update=row['update'],step=row['step'])
                    if point['wins']!=row['validation']['wins']:raise ValueError('Logged win count differs')
                    curve.append(point)
                if [p['update']for p in curve]!=list(range(50,401,50)):raise ValueError('All scheduled validations required')
                chosen=next(p for p in curve if p['update']==selected['selected_update'])
                if summarize(selected['validation']['records'],reference)!= {k:v for k,v in chosen.items()if k not in ('update','step')}:
                    raise ValueError('Selected checkpoint validation differs from its curve')
                runs.append(dict(id=name,selected_update=selected['selected_update'],curve=curve,
                    final_minus_selected_wins=curve[-1]['wins']-chosen['wins'],
                    final_minus_selected_defense_wins=curve[-1]['defense_wins']-chosen['defense_wins'],
                    final_minus_first_wins=curve[-1]['wins']-curve[0]['wins'],
                    final_minus_first_defense_wins=curve[-1]['defense_wins']-curve[0]['defense_wins'],
                    kl_stopped_updates=sum(r['optimizer_controls']['kl_stopped']for r in rows),
                    metrics_sha256=receipt['files'][name+'/metrics.jsonl']['sha256']))
        trajectory=[]
        for i,update in enumerate(range(50,401,50)):
            points=[r['curve'][i]for r in runs]
            trajectory.append(dict(update=update,run_count=len(points),
                mean_wins_per72=mean(p['wins']for p in points),
                mean_defense_wins_per36=mean(p['defense_wins']for p in points),
                runs_meeting_family_floor=sum(p['worst_family_points']>=.1 for p in points),
                mean_own_fraction_loss=mean(p['mean_own_fraction_loss']for p in points),
                mean_empty_spell_elixir=mean(p['mean_empty_spell_elixir']for p in points),
                mean_duration_seconds=mean(p['mean_duration_seconds']for p in points),
                card_plays_per_minute={c:mean(p['card_plays_per_minute'].get(c,0)for p in points)
                    for c in sorted({c for p in points for c in p['card_plays_per_minute']})},
                card_plays_per_game={c:mean(p['card_plays_per_game'].get(c,0)for p in points)
                    for c in sorted({c for p in points for c in p['card_plays_per_game']})}))
        studies.append(dict(kernel=proof['kernel'],root=str(root),receipt_sha256=file_sha(receipt_path),
            parent=summarize(reference,reference),runs=runs,trajectory=trajectory,
            selected_updates=dict(Counter(r['selected_update']for r in runs)),
            final_fewer_wins_than_selected=sum(r['final_minus_selected_wins']<0 for r in runs),
            final_fewer_defense_wins_than_selected=sum(r['final_minus_selected_defense_wins']<0 for r in runs)))
    return dict(schema='verified-learning-curves-v1',studies=studies,quality_accepted=False,
                checkpoint_selection_changed=False,independent_test=False,
                scope='Repeated development validation of every scheduled checkpoint; no new games or training',
                limitations=['Runs include repeated controls; averages are descriptive, not independent samples',
                    'Selected checkpoints use the existing family guard/score, not maximal total wins',
                    'Late degradation does not establish its cause or prove early stopping will generalize'])


def render(report):
    runs=[r for study in report['studies']for r in study['runs']]
    first=[r['curve'][0]for r in runs];last=[r['curve'][-1]for r in runs]
    lines=['# Как менялось поведение при обучении','',
        f'Проверены завершённые прогоны ({len(runs)}): все восемь оценок от update 50 до 400, '
        'по 72 одинаковых validation-сценария. Веса, журналы и receipt повторно проверены. '
        'Ниже среднее по прогонам; повторные контроли и сценарии не являются независимыми испытаниями.','',
        '| Карта | Постановок/матч, update 50 | Update 400 | В минуту, update 50 | Update 400 |',
        '|---|---:|---:|---:|---:|']
    names=dict(cannon='Пушка',musketeer='Мушкетёр',hog_rider='Хог',fireball='Fireball',
               knight='Рыцарь',skeletons='Скелеты',electro_spirit='Электродух',zap='Zap')
    for c,label in names.items():
        values=[mean(p[key][c]for p in group)for key in ('card_plays_per_game','card_plays_per_minute')for group in (first,last)]
        lines.append('| '+label+' | '+' | '.join(f'{v:.2f}'for v in values)+' |')
    lines+=['','| Показатель | Update 50 | Update 400 |','|---|---:|---:|']
    for key,label in [('wins','Побед из 72'),('defense_wins','Защитных побед из 36')]:
        lines.append(f"| {label} | {mean(p[key]for p in first):.2f} | {mean(p[key]for p in last):.2f} |")
    for family,label in [('air','Побед против воздуха из 12'),('hog','Против хога из 12')]:
        lines.append(f"| {label} | {mean(p['by_family_wins'][family]for p in first):.2f} | {mean(p['by_family_wins'][family]for p in last):.2f} |")
    def fraction(group,key):
        return sum(p['decision_economy']['totals'][key]for p in group)/sum(p['decision_economy']['totals']['decisions']for p in group)
    lines.append(f"| Доступный эликсир <4, доля шагов | {100*fraction(first,'available_below4'):.2f}% | {100*fraction(last,'available_below4'):.2f}% |")
    def card_fraction(group,numerator,denominator):
        return sum(p['decision_economy']['cards']['musketeer'][numerator]for p in group)/sum(p['decision_economy']['cards']['musketeer'][denominator]for p in group)
    lines.append(f"| Мушкетёр доступен, когда находится в руке | {100*card_fraction(first,'legal','in_hand'):.2f}% | {100*card_fraction(last,'legal','in_hand'):.2f}% |")
    lines.append(f"| Выбран мушкетёр, когда его можно поставить | {100*card_fraction(first,'selected','legal'):.2f}% | {100*card_fraction(last,'selected','legal'):.2f}% |")
    lines+=['',
        '**Вывод:** мушкетёр используется реже при резко меньшем количестве допустимых возможностей. '
        'Условная частота выбора при наличии такой возможности, наоборот, растёт. '
        'Снижение постановок подтверждается и в расчёте на минуту, а не только на матч. '
        'Доступный эликсир учитывает уже зарезервированные расходы; это счётчики среды на шагах решений.','',
        f"В {sum(r['final_minus_selected_wins']<0 for r in runs)}/{len(runs)} прогонах финальные веса дают меньше общих побед, "
        f"а в {sum(r['final_minus_selected_defense_wins']<0 for r in runs)}/{len(runs)} — меньше защитных побед, чем уже сохранённый выбранный checkpoint. "
        'Выбор checkpoint использует прежний порог семейств и score; здесь он не менялся. '
        'На update 400 порог каждого семейства достигается только у '
        f"{sum(p['worst_family_points']>=.1 for p in last)}/{len(runs)} отдельных прогонов. Это ещё не прохождение парного отбора профиля.", '',
        '**Следующий шаг:** завершить текущий опыт GAE × баланс карт и проверить, меняется ли этот расход эликсира '
        'без ухудшения защиты. Подготовленный опыт по шагам выбора проверяет другую гипотезу. '
        'Эта диагностика не устанавливает причину деградации, не оправдывает произвольный запрет дешёвых карт '
        'и не является независимым test или реальным винрейтом.','',
        'Источники и все кривые: [JSON](defense_plan/learning_curves_with_opportunities_20261104.json).']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();target=Path(a.output)
    if target.exists():raise ValueError('Preserve existing audit')
    target.write_text(json.dumps(audit(a.roots),indent=2)+'\n')
