"""Render the short dashboard from audited metrics pinned to the four final weights."""
import argparse
import json
from pathlib import Path

ROSTER=(('defense','spell_precision','Защита · spell_precision'),
        ('defense','economy_soft','Защита · economy_soft'),
        ('joint','economy_soft','Атака + защита · economy_soft'),
        ('joint','tower_survival','Атака + защита · tower_survival'))


def percent(value):return '—'if value is None else f'{100*value:.2f}%'.replace('.',',')
def number(value,places=1):return f'{value:.{places}f}'.replace('.',',')


def rate(value):
    if not value['known']:return 'Неизвестно: '+str(value['unknown'])
    return f"{percent(value['rate'])} · {value['success']}/{value['known']}"+(f"; неизвестно {value['unknown']}"if value.get('unknown')else'')


def winrate(report,baseline=None):
    if report is None:return '—'
    rows=[report['opponents'][baseline]]if baseline else list(report['opponents'].values())
    if any(r['observed_simulation']['unknown']for r in rows):return 'Есть неизвестные исходы'
    wins=sum(r['observed_simulation']['wins']for r in rows);games=sum(r['games']for r in rows)
    return f'{percent(wins/games)} · {wins}/{games}'


def load_models(reports):
    completion=json.loads((reports/'ppo_selected_four_full_completion.json').read_text())
    if completion.get('all_four_full_budgets_verified')is not True:raise ValueError('Four completed full budgets required')
    pinned={(r['mode'],r['profile']):r for r in completion['models']};models=[]
    for mode,profile,label in ROSTER:
        boundary=pinned[(mode,profile)]
        if (boundary['steps'],boundary['update'])!=(2048000,2000):raise ValueError('Incomplete model cannot show100%')
        item=dict(mode=mode,profile=profile,label=label,key=mode+'_'+profile,full=None,train=None)
        for kind,schema in (('full','verified-selected-ppo-final-summary-v1'),('train','verified-selected-ppo-train-summary-v1')):
            path=reports/('ppo_'+item['key']+'_'+kind+'_metrics.json')
            if not path.exists():continue
            report=json.loads(path.read_text())
            if (report.get('schema')!=schema or report.get('checkpoint_sha256')!=boundary['checkpoint_sha256']or
                report.get('mode')!=mode or report.get('profile')!=profile):
                raise ValueError('Dashboard metrics do not describe the pinned final weights')
            item[kind]=report
        models.append(item)
    return models


def render(models,refinement=None):
    ready=sum(m['full']is not None for m in models);trained=sum(m['train']is not None for m in models)
    lines=['# Где мы сейчас и что получилось','',
        f'**Этап:** все 4 модели обучены. Val/test и матчи проверены у {ready}/4; отдельные train-замеры — у {trained}/4.','',
        '| Модель | Обучение | Готовые замеры | Против v25 | Против всех 15 |','|---|---:|---|---:|---:|']
    if refinement:
        state=('обучение завершено; независимая проверка '+refinement['evaluation']['observed_status']
               if refinement.get('evaluation') else refinement['remote_status'])
        note=('**Сейчас:** дообучение сохранённой PPO «атака + защита · economy_soft» на Fireball: '
              'добивание башни/здания и попадание по башне с юнитом одним шаром. '
              'Kaggle: '+state+'. Исходные веса сохранены; результаты нового кандидата ещё не проверены. '
              '[Статус](reports/defense_plan/fireball_refinement_status.json).')
        lines[2:2]=[note,'']
        candidate=refinement.get('candidate_summary')
        if candidate:
            extra=['**Fireball, validation (отбор кандидата; ещё не независимый test):**','',
                   '| Задача | Исходник | После дообучения |','|---|---:|---:|']
            for key,label in (('tower_finish','Добить башню'),('building_finish','Добить здание'),('tower_unit','Башня + юнит одним шаром')):
                before=candidate['validation_before'][key];after=candidate['validation_after'][key]
                extra.append(f"| {label} | {before['successes']}/{before['episodes']} | {after['successes']}/{after['episodes']} |")
            lines[4:4]=extra+['']
    for m in models:
        state=('Val/test и матчи готовы'if m['full']else'Val/test и матчи ожидают отчёта')+('; train готов'if m['train']else'; train ожидает отчёта')
        lines.append(f"| {m['label']} | 100% | {state} | {winrate(m['full'],'v25')} | {winrate(m['full'])} |")
    lines+=['','100% — бюджет обучения, 2 048 000 шагов. Настройки награды указаны после точки. `—` — проверенного результата ещё нет.','',
        '## Защита от заданного пуша','','«Без урона» — ни одна наша башня не потеряла HP. HP и эликсир — среднее за эпизод; эликсир восстанавливается. Таймаут — обрыв по лимиту времени, незавершённая защита.','',
        '| Модель | Набор | Эпизодов | Таймаутов | Без урона | Потеря HP всех башен | Эликсир | Принцесса на линии атаки жива | Наш король жив |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in models:
        for split,label in (('train','Train'),('validation','Val'),('test','Test')):
            data=m['train']['metrics']if split=='train'and m['train']else(m['full'][split]if split!='train'and m['full']else None)
            if data is None:lines.append(f"| {m['label']} | {label} | Ждём отчёт | — | — | — | — | — | — |");continue
            d=data['defense']['overall']
            lines.append(f"| {m['label']} | {label} | {d['episodes']} | {d['timeouts']} | {rate(d['no_damage'])} | {number(d['mean_tower_hp_loss'])} | {number(d['mean_elixir_spent'],2)} | {percent(d['princess_survived']['rate'])} | {percent(d['king_survived']['rate'])} |")
    lines+=['','Train — учебные сценарии; Val — проверочные; Test — отложенные. По Test веса и коэффициенты не подбираются.','',
        '| Модель · test | Простой пуш без урона | Средний | Сложный | Стресс |','|---|---:|---:|---:|---:|']
    for m in models:
        if m['full']:lines.append('| '+m['label']+' | '+' | '.join(rate(m['full']['test']['defense']['by_difficulty'][d]['no_damage'])for d in ('simple','medium','complex','impossible'))+' |')
    lines+=['','## Атака в отдельных упражнениях','','Хог — фактическое попадание в нужную башню; добивание — завершающий Fireball/Zap. У защитных моделей это контроль: отдельной атаке их не обучали.','',
        '| Модель | Набор | Хог попал | Добил заклинанием | Пропущено окно атаки |','|---|---|---:|---:|---:|']
    for m in models:
        for split,label in (('train','Train'),('validation','Val'),('test','Test')):
            data=m['train']['metrics']if split=='train'and m['train']else(m['full'][split]if split!='train'and m['full']else None)
            if data is None:lines.append(f"| {m['label']} | {label} | Ждём отчёт | — | — |");continue
            a=data['attack'];lines.append(f"| {m['label']} | {label} | {rate(a['hog_confirmed_hit'])} | {rate(a['spell_confirmed_finish'])} | {rate(a['missed_window'])} |")
    lines+=['','[Проверка добивания на двух train-примерах](reports/defense_plan/ppo_train_spell_finisher_probe.json): точный Fireball — 2/2, tower_survival — 0/2.','',
        '## Полноценные матчи в симуляторе','','На соперника — 20 матчей: 10 шафлов с обменом сторон. v25 входит в итог 300 матчей. Цель 80% против каждой прошлой модели пока не достигнута.','',
        '| Соперник | '+' | '.join(m['label']for m in models)+' |','|---|'+'---:|'*len(models)]
    baselines=next((list(m['full']['opponents'])for m in models if m['full']),[])
    for b in baselines:lines.append('| '+b+' | '+' | '.join(winrate(m['full'],b)for m in models)+' |')
    lines+=['','## Подробные проверенные отчёты','']
    for m in models:
        links=[]
        for key,label in (('full','Val/test и матчи'),('train','Train')):
            if m[key]:links.append(f"[{label}](reports/defense_plan/ppo_{m['key']}_{key}_metrics.json)")
        if links:lines.append(m['label']+': '+', '.join(links)+'.')
    lines+=['','Train: 1484 сценария × 3 шафла = 4452 эпизода, веса не меняются. Живые бои выключены; соответствие симулятора реальной игре ещё не принято.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,default=Path('reports/defense_plan'))
    parser.add_argument('--output',type=Path,default=Path('done.md'))
    args=parser.parse_args()
    status=args.reports/'fireball_refinement_status.json'
    args.output.write_text(render(load_models(args.reports),json.loads(status.read_text()) if status.exists() else None))
