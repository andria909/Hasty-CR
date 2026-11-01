"""Verify paired full-match validation and report economy without promotion."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.attack.select_continuous_focus import validation_score
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha


def summarize(source, output, markdown):
    source=Path(source); raw=json.loads(source.read_text()); protocol=raw['protocol']
    if (protocol.get('schema')!='bank-full-validation-protocol-v1' or
            protocol.get('split')!='validation' or protocol.get('independent_test') is not False or
            protocol.get('live_games') is not False or protocol['physics_pins']!=physics_pins() or
            file_sha(source.parent/'driver.py')!=protocol['source_sha256']):
        raise ValueError('Pinned simulator validation required')
    reports=raw['reports']
    if len(reports)!=3 or [r['model']for r in reports]!=protocol['models']:
        raise ValueError('Parent and two paired candidates required')
    models=[]; reference=None
    labels=('Исходная balanced','Прежний BC · seed392','Прежний BC · seed393')
    for label,report in zip(labels,reports):
        rows=report['records'];score=validation_score(report)
        if file_sha(report['model']['checkpoint'])!=report['model']['sha256']:
            raise ValueError('Checkpoint changed')
        keys=[(r['episode_seed'],r['match_seed'],r['family_id'],r['task'])for r in rows]
        if {r['episode_seed']for r in rows}!=set(range(protocol['seed_base'],protocol['seed_base']+72)):
            raise ValueError('Unexpected validation seeds')
        if reference is None: reference=dict(zip(keys,rows))
        if set(keys)!=set(reference):raise ValueError('Unpaired match contexts')
        if any(r['outcome']not in ('win','loss','draw')for r in rows):raise ValueError('Unknown outcome')
        for r in rows:
            if (r['normal_range_mode']!='edge_to_edge_v1' or r['hog_sight_mode']!='side_clip_v1' or
                    r['cannon_center_only']!=1 or r['action_cooldown_ms']!=0 or r['cannon_outside_center']!=0):
                raise ValueError('Physical execution contract differs')
        wins=sum(r['outcome']=='win'for r in rows)
        if report['wins']!=wins or report['games']!=72 or report['score']!=score:
            raise ValueError('Stored score differs from actual matches')
        cards=Counter();decisions=Counter()
        for r in rows:
            cards.update(r['accepted_card_counts'])
            decisions.update({k:v for k,v in r['decision_economy'].items()if isinstance(v,(float,int))})
        tasks={task:dict(games=sum(r['task']==task for r in rows),
            wins=sum(r['task']==task and r['outcome']=='win'for r in rows))for task in ('attack','defense')}
        if any(t['games']!=36 for t in tasks.values()):raise ValueError('Unbalanced tasks')
        paired=Counter()
        for key,r in zip(keys,rows):
            before=reference[key]['outcome']=='win';after=r['outcome']=='win'
            paired['gained_win'if after and not before else'lost_win'if before and not after else'unchanged_win_status']+=1
        models.append(dict(label=label,**report['model'],wins=wins,games=72,score=score,tasks=tasks,
            all_family_floor_passed=score['worst_archetype_points']>=.1,
            paired_against_parent=dict(paired),accepted_cards_per_game={k:v/72 for k,v in sorted(cards.items())},
            available_below4_decision_fraction=decisions['available_below4']/decisions['decisions'],
            voluntary_wait_decision_fraction=decisions['voluntary_wait']/decisions['decisions'],
            empty_spell_elixir_per_game=sum(r['empty_spell_elixir']for r in rows)/72,
            overflow_per_game=sum(r['elixir_overflow']for r in rows)/72))
    result=dict(schema='verified-bank-full-validation-v1',source=str(source),source_sha256=file_sha(source),
        episodes_verified=216,unique_contexts=72,models=models,full_training_accepted=False,
        independent_test=False,live_games=False,scope='Paired simulator validation, not head-to-head or live win rate')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Полные матчи: перенос прежнего BC','',
        '72 одинаковых контекста на модель: 36 атак и 36 защит, по 12 матчей каждого из шести семейств. '
        'Текущая физика, задержка 1350 мс, пушка только в центре. Это validation в симуляторе; '
        'модели встречают одинаковых сценарных противников, не играют друг с другом.','',
        '| Модель | Победы / 72 | Атака / 36 | Защита / 36 | Худшее семейство |',
        '|---|---:|---:|---:|---:|']
    for r in models:
        lines.append(f"| {r['label']} | {r['wins']} — {100*r['wins']/72:.1f}% | {r['tasks']['attack']['wins']} | {r['tasks']['defense']['wins']} | {100*r['score']['worst_archetype_points']:.1f}% очков |")
    lines+=['','Очки семейства: победа = 1, ничья = 0,5. Минимум для продолжения — 10% в каждом семействе.','',
        '| Модель | Решения при доступном эликсире < 4 | Добровольное ожидание | Пустые спеллы, эликсир/матч | Перелив, эликсир/матч |',
        '|---|---:|---:|---:|---:|']
    for r in models:
        lines.append(f"| {r['label']} | {100*r['available_below4_decision_fraction']:.1f}% | {100*r['voluntary_wait_decision_fraction']:.1f}% | {r['empty_spell_elixir_per_game']:.2f} | {r['overflow_per_game']:.2f} |")
    cards=sorted(set().union(*(r['accepted_cards_per_game']for r in models)))
    lines+=['','Среднее число фактически принятых постановок за матч:','',
        '| Карта | balanced | BC392 | BC393 |','|---|---:|---:|---:|']
    for card in cards:lines.append('| '+card+' | '+' | '.join(f"{r['accepted_cards_per_game'].get(card,0):.2f}"for r in models)+' |')
    lines+=['','Парные изменения относительно balanced на тех же 72 контекстах:','']
    for r in models[1:]:
        p=r['paired_against_parent'];lines.append(f"- {r['label']}: новых побед {p.get('gained_win',0)}, утраченных побед {p.get('lost_win',0)}.")
    lines+=['','Это прежние веса на старом учителе, не три текущие GPU-серии. '
        'Накопление само по себе не является критерием успеха: нужны победы, сохранение башен и своевременные ответы. '
        'Большое обучение этим отчётом не разрешается.','']
    Path(markdown).write_text('\n'.join(lines));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','output','markdown'):p.add_argument('--'+key,required=True)
    a=p.parse_args();r=summarize(a.source,a.output,a.markdown);print(json.dumps(dict(episodes_verified=r['episodes_verified'])))
