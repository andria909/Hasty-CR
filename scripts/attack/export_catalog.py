"""Export reviewable attack preparation artifacts without training or overwrites."""
import argparse
import json
from pathlib import Path

from sim.attack_scenarios import build_catalog


def readable_combinations(rows,coverage):
    lines=['# Комбинации атакующих пушей', '',
        f'Всего {coverage["count"]} сценариев, {coverage["families"]} семейств. Обе линии, скрытый и видимый учебный таймер.', '',
        '## Соло-хог', '',
        '- Hog Rider с моста.',
        '- Hog Rider с глубины своей половины.',
        '- Hog Rider из кармана на половине врага после уничтожения башни лучниц.', '',
        'Если обе башни лучниц живы, цель — башня с меньшим HP; при равном HP допустима любая. После гибели одной башни лучниц цель — король.', '',
        '## Хог под уже идущую поддержку', '']
    supports=sorted({u.card for s in rows if s.goal_kind=='hog_tower_damage'for u in s.initial_units})
    labels={'musketeer':'Musketeer','knight':'Knight','giant':'Giant','skeletons':'Skeletons',
            'electro_spirit':'Electro Spirit','cannon':'Cannon','tesla':'Tesla','bomb_tower':'Bomb Tower'}
    lines+=['- Hog Rider + '+labels.get(card,card)+' (поддержка уже на поле).'for card in supports]
    lines+=['', 'Поддержка стартует на двух расстояниях от моста; политика выбирает момент и место выпуска хога.', '',
            '## Хог против здания', '']
    buildings=sorted({p.card for s in rows if s.goal_kind=='hog_tower_damage'for p in s.enemy_plays})
    lines+=['- Hog Rider против '+labels.get(card,card)+'; в руке доступны Fireball и Zap.'for card in buildings]
    lines+=['', 'Два варианта постановки здания: в заранее выбранную точку или после появления хога с задержкой реакции. Попадание предиктом и реактивным заклинанием ещё предстоит оценить.', '',
            '## Добивания заклинаниями', '',
            '- Fireball по башне лучниц.',
            '- Zap по башне лучниц.',
            '- Fireball по королю после уничтожения башни лучниц.',
            '- Zap по королю после уничтожения башни лучниц.',
            '- Fireball по башне лучниц, пока противник готовит ответный Fireball по нашей слабой башне.',
            '- Zap по башне лучниц, пока противник готовит ответный Fireball по нашей слабой башне.', '',
            'Для каждого заклинания HP цели: на единицу ниже урона, ровно урон и на единицу выше. Последний вариант требует комбинации. Начальный эликсир: 6 или 10.', '',
            'Урон по коронным башням в текущем профиле симулятора: '+
            ', '.join(f'{name} — {damage}'for name,damage in coverage['crown_damage_level11'].items())+'.', '',
            '## Окна выполнения', '']
    from sim.attack_scenarios import DEFAULT_DEADLINES
    lines += [f'- {kind}: {low/1000:g}–{high/1000:g} секунд.'for kind,(low,high)in DEFAULT_DEADLINES.items()]
    lines += ['', '## Разбиение и ограничения', '',
              ', '.join(f'{name}: {count}'for name,count in coverage['by_split'].items())+'. Родственные варианты остаются в одном split.', '',
              'Это подготовленные упражнения, обучение атаки ещё не запускалось. Окна предварительные; стартовые руки каталога фиксированы по типу цели. Метки мост/глубина/карман пока не ограничивают действие модели. Полный аудит предиктов, комбинаций и игровых механик не завершён.', '']
    return '\n'.join(lines)


def export(destination,markdown_output=None):
    rows,coverage=build_catalog()
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'scenarios.json').write_text(json.dumps(
        dict(schema='attack-v1',scenarios=[s.to_dict()for s in rows]),indent=2)+'\n')
    (destination/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    lines=['# Атакующие упражнения: подготовка', '',
        'Это каталог задач симулятора, не обученная модель. Скрытый и видимый таймер — парные варианты одной задачи.',
        'Окна предварительные. Стартовые руки пока фиксированы по типу цели; back/bridge/pocket — метки упражнений, фактическую постановку ещё нужно контролировать метриками.', '',
        '| Задача | Split | Цель и условия | Окно, с |',
        '|---|---|---|---:|']
    for s in rows:
        if s.deadline_visibility!='hidden':continue
        descriptions=[s.goal_kind,s.setup_kind,'линия '+s.lane,'цель '+s.target_lane]
        descriptions += [f'{u.card} уже на поле ({u.x},{u.y})'for u in s.initial_units]
        descriptions += [f'{p.card} в ({p.x},{p.y}) {p.position_kind}'for p in s.enemy_plays]
        descriptions += [f'башня {t.side}/{t.lane}: HP={t.hp}'for t in s.tower_hp]
        if s.finisher_spell:
            descriptions += [f'{s.finisher_spell}: урон башне {s.finisher_tower_damage}, HP {s.finisher_hp_relation}']
        lines.append(f'| {s.paired_case_id} | {s.split} | '+ '; '.join(descriptions)+
                     f' | {s.deadline_ms_range[0]/1000:g}–{s.deadline_ms_range[1]/1000:g} |')
    (destination/'pushes.md').write_text('\n'.join(lines)+'\n')
    markdown_output=Path(markdown_output)if markdown_output is not None else destination/'attack.md'
    with markdown_output.open('x')as stream:stream.write(readable_combinations(rows,coverage))
    return coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--markdown-output',type=Path)
    args=parser.parse_args()
    report=export(args.output,args.markdown_output)
    print(json.dumps({k:report[k]for k in ('count','families','by_split','full_goal_split_coverage','trained')}))


if __name__=='__main__':main()
