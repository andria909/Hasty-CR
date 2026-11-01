"""Explain full-match economy and card use, including available opportunities.

Reads existing complete simulator validation only. A low empty-spell count
does not imply good play when the policy barely plays any cards.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from scripts.attack.select_continuous_focus import validation_score
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha


def ratio(numerator,denominator):
    return numerator/denominator if denominator else None


def percent(value):
    return '—'if value is None else f'{value:.1%}'


def label(model,index):
    if index==0:return 'Исходные веса'
    name=Path(model['checkpoint']).parent.name
    if name.startswith(('control_kl','corrections_kl')):
        kind,kl,seed=name.split('_')
        return f"{'Контроль'if kind=='control'else'Исправления'} · KL {'0,01'if kl=='kl001'else'0,1'} · {seed}"
    return name


def aggregate(rows):
    cards=Counter();legal=Counter();selected=Counter();decisions=Counter()
    resolved=Counter();empty=Counter();lanes={side:Counter()for side in ('left','right')}
    for r in rows:
        if not math.isfinite(r['elapsed_ms'])or r['elapsed_ms']<=0:raise ValueError('Positive match duration required')
        cards.update(r['accepted_card_counts'])
        economy=r['decision_economy']
        total=economy['decisions']
        if total<=0 or sum(economy[k]for k in ('forced_wait','voluntary_wait','plays','invalid'))!=total:
            raise ValueError('Decision accounting differs')
        for key in ('decisions','forced_wait','voluntary_wait','plays','invalid','available_below4','available_elixir_sum'):
            decisions[key]+=economy[key]
        for card,counts in economy['cards'].items():
            if not 0<=counts['selected']<=counts['legal']<=counts['in_hand']<=total:
                raise ValueError('Impossible card opportunity accounting')
            legal[card]+=counts['legal'];selected[card]+=counts['selected']
        resolved.update(r['resolved_spell_counts']);empty.update(r['empty_spell_counts'])
        for side in lanes:lanes[side].update(r['accepted_card_lane_counts'].get(side,{}))
    minutes=sum(r['elapsed_ms']for r in rows)/60000
    names=sorted(set(cards)|set(legal))
    n=len(rows)
    return dict(games=n,wins=sum(r['outcome']=='win'for r in rows),
        duration_minutes=minutes,mean_duration_s=minutes*60/n,
        cards={card:dict(accepted=cards[card],per_game=cards[card]/n,per_minute=cards[card]/minutes,
            legal_decisions=legal[card],selected_decisions=selected[card],
            selected_given_legal=ratio(selected[card],legal[card]),
            accepted_left=lanes['left'][card],accepted_right=lanes['right'][card])for card in names},
        available_below4_fraction=ratio(decisions['available_below4'],decisions['decisions']),
        mean_available_elixir=ratio(decisions['available_elixir_sum'],decisions['decisions']),
        forced_wait_fraction=ratio(decisions['forced_wait'],decisions['decisions']),
        voluntary_wait_fraction=ratio(decisions['voluntary_wait'],decisions['decisions']),
        plays_given_choice=ratio(decisions['plays'],decisions['plays']+decisions['voluntary_wait']),
        empty_spell_elixir_per_game=sum(r['empty_spell_elixir']for r in rows)/n,
        empty_spell_elixir_per_minute=sum(r['empty_spell_elixir']for r in rows)/minutes,
        overflow_per_game=sum(r['elixir_overflow']for r in rows)/n,
        own_tower_damage_per_game=sum(r['own_tower_damage']for r in rows)/n,
        nonlethal_crown_only_elixir_per_game=sum(r['nonlethal_crown_only_elixir']for r in rows)/n,
        spells={c:dict(resolved=resolved[c],empty=empty[c],empty_fraction=ratio(empty[c],resolved[c]))for c in sorted(resolved|empty)})


def summarize(source,output,markdown):
    source=Path(source);raw=json.loads(source.read_text());p=raw['protocol']
    if (p.get('schema')!='bank-full-validation-protocol-v1'or p['physics_pins']!=physics_pins()or
            p['split']!='validation'or p['independent_test']or p['live_games']or
            file_sha(source.parent/'driver.py')!=p['source_sha256']):raise ValueError('Pinned simulator validation required')
    if [r['model']for r in raw['reports']]!=p['models']:raise ValueError('Model inventory differs')
    models=[];keys=None
    for r in raw['reports']:
        score=validation_score(r);rows=r['records']
        observed={(v['episode_seed'],v['match_seed'],v['family_id'],v['task'])for v in rows}
        if keys is None:keys=observed
        if observed!=keys or score!=r['score']or file_sha(r['model']['checkpoint'])!=r['model']['sha256']:
            raise ValueError('Unpaired models or changed weights')
        row=dict(model=r['model'],all=aggregate(rows),tasks={t:aggregate([v for v in rows if v['task']==t])for t in ('attack','defense')})
        if row['all']['wins']!=r['wins']or row['all']['games']!=r['games']:raise ValueError('Stored match counts differ')
        models.append(row)
    result=dict(schema='full-match-decision-analysis-v1',source=str(source),source_sha256=file_sha(source),models=models,
        full_training_accepted=False,live_games=False,
        scope='Simulator validation; legal decisions are correlated opportunities, not independent trials')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Почему модель ставит или не ставит карты','',
        'Одинаковые полные сценарии симулятора. «При выборе» исключает вынужденное ожидание. '
        'Доступность карты проверяется по маске действий, а не только наличию в руке.','',
        '| Веса | Победы | Эликсир <4 | Разыгрыш при выборе | Пустые спеллы, эликсир/матч | Перелив/матч |','|---|---:|---:|---:|---:|---:|']
    for i,row in enumerate(models):
        r=row['all'];name=label(row['model'],i)
        lines.append(f"| {name} | {r['wins']}/{r['games']} | {percent(r['available_below4_fraction'])} | {percent(r['plays_given_choice'])} | {r['empty_spell_elixir_per_game']:.2f} | {r['overflow_per_game']:.2f} |")
    for i,row in enumerate(models):
        lines+=['',f'## {label(row["model"],i)}','',
            '| Карта | За матч | За минуту | Выбрана / доступна, решений | Доля выбора при доступности |','|---|---:|---:|---:|---:|']
        for card,r in row['all']['cards'].items():
            rate='—'if r['selected_given_legal']is None else f"{r['selected_given_legal']:.1%}"
            lines.append(f"| {card} | {r['per_game']:.2f} | {r['per_minute']:.2f} | {r['selected_decisions']}/{r['legal_decisions']} | {rate} |")
    lines+=['','Много доступных решений не означает много независимых моментов: одна и та же рука повторяется через 250 мс. '
        'Низкая доля выбора сама по себе не ошибка: ожидание может быть правильным. '
        'Сопоставлять с исходами и потерями башен; не награждать одинаковую частоту всех карт. '
        'Количество принятых команд не доказывает успешное попадание.','']
    Path(markdown).write_text('\n'.join(lines));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','output','markdown'):p.add_argument('--'+key,required=True)
    a=p.parse_args();summarize(a.source,a.output,a.markdown)
