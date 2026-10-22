"""Export reviewable attack preparation artifacts without training or overwrites."""
import argparse
import json
from pathlib import Path

from sim.attack_scenarios import build_catalog


def export(destination):
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
    return coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=export(args.output)
    print(json.dumps({k:report[k]for k in ('count','families','by_split','full_goal_split_coverage','trained')}))


if __name__=='__main__':main()
