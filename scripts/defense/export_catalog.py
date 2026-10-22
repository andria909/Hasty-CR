"""Export the executable catalog and its readable combinations together."""
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.defense_scenarios import SCHEMA, MEDIUM, COMPLEX, BUILDING_PRESSURE, BUILDING_GUESSES, build_catalog, catalog_report


def main():
    scenarios, _ = build_catalog()
    report = catalog_report()
    destination = ROOT / 'data/defense'
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in [('scenarios', {'schema': SCHEMA, 'count': len(scenarios),
                                        'scenarios': [s.to_dict() for s in scenarios]}),
                          ('coverage', report)]:
        (destination / f'{name}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    names = {c['key']: c['name'] for c in json.loads((ROOT/'data/royaleapi/cards.json').read_text())['cards']}
    families = {s.family_id for s in scenarios}
    def supported(combos, prefix):
        return [combo for combo in combos if prefix+':'.join(sorted(combo)) in families]
    def label(combo):
        return ' + '.join(names.get(card, card) for card in combo)
    medium = supported(MEDIUM, 'pair:')
    complex_pushes = supported(COMPLEX, 'push:')
    lines = ['# Комбинации пушей', '',
             f'Всего {len(scenarios)} сценария с вариантами стороны, стартовой позиции и режима эликсира.', '',
             '## Простые', '', 'Одна карта, обе линии, старт с моста или сзади. Направляемые карты используют цель сценария.', '']
    cards = sorted({s.plays[0].card for s in scenarios if s.difficulty=='simple'})
    lines += ['- '+names.get(card, card) for card in cards]
    lines += ['', f'## Средние — {len(medium)} комбинаций', '',
              'Обе линии. Варианты: группа с моста; поддержка сзади или из угла, затем перед ней основной юнит у моста. Карты ставятся последовательно с учётом доступного эликсира.', '']
    for length in (2, 3, 4):
        subset = [combo for combo in medium if len(combo)==length]
        lines += [f'### {length} карты — {len(subset)} комбинаций', '']
        lines += ['- '+label(combo) for combo in subset]
        lines += ['']
    lines += [f'## Сложные — {len(complex_pushes)} комбинаций', '',
              'Построение сзади, обе линии, эликсир x1/x2/x3. Поддерживающие заклинания ждут подхода основной группы к мосту.', '']
    for length in (3, 4):
        subset = [combo for combo in complex_pushes if len(combo)==length]
        lines += [f'### {length} карты — {len(subset)} комбинаций', '']
        lines += ['- '+label(combo) for combo in subset]
        lines += ['']
    lines += [f'## Стрессовые — {len(complex_pushes)} комбинаций', '',
              'Сложный пуш с дополнительным резервом 10 эликсира, заклинанием и подкреплением. Обе линии, x1/x2/x3.', '']
    for combo in complex_pushes:
        spell = 'zap' if 'fireball' in combo else 'fireball'
        troop = 'battle_ram' if 'hog_rider' in combo else 'hog_rider'
        lines.append('- '+label(combo)+f'; подкрепление: {label((spell, troop))}')
    building = supported(BUILDING_PRESSURE, 'building:')
    lines += ['', f'## Юниты + заклинание по предполагаемому зданию — {len(building)} комбинаций', '',
              'Пять фиксированных предположений о центральной постановке: центр, ближе к реке, глубже, наружу и внутрь. Для правой линии X отражён. Координаты реальных зданий не читаются. Режимы x1/x2/x3; заклинание запускается после подхода первого юнита к мосту либо его гибели на подходе.', '']
    lines += ['- '+label(combo) for combo in building]
    lines += ['', 'Точки для левой линии (в тайлах, Y растёт к нашей башне):', '']
    lines += [f'- {name}: ({x/1000:g}, {y/1000:g})' for name,(x,y) in BUILDING_GUESSES.items()]
    lines += ['', 'Earthquake пока исключён: в публичном снимке не хватает точных данных для текущей реализации.', '']
    lines += ['', '## Ограничения', '',
              'Это комбинации поддержанных карт публичного снимка. Неподдержанные карты и ограничения механик перечислены в [coverage.json](defense/coverage.json).', '',
              'Расписания всех постановок: [scenarios.json](defense/scenarios.json).', '']
    (ROOT/'data/pushes.md').write_text('\n'.join(lines))
    print(json.dumps({key: report[key] for key in ('scenarios','families','by_split','by_difficulty')}, ensure_ascii=False))
    print('medium', dict(Counter(map(len, medium))), 'complex', dict(Counter(map(len, complex_pushes))))


if __name__ == '__main__':
    main()
