"""Wait locally for all three verified studies, then independently select profiles."""
import argparse
import json
from pathlib import Path
import time

from scripts.attack.select_pilot_pool import select


def render(result):
    lines=['# Отбор PPO для длинного обучения','',
        'Только development validation: 72 одинаковых сценария. Повторённые контрольные профили объединены по весам исходника и параметрам; лучший случайный повтор не выбирается.',
        'Порог по всем семействам сохранён. Дополнительно каждый seed должен сохранить общие победы и победы в защите относительно своего исходника или PPO-контроля.', '',
        '| Серия / профиль | Seed | Оценка отбора | Худшее семейство | Защита и победы сохранены | Допущен |',
        '|---|---:|---:|---:|---|---|']
    for row in result['ranking']:
        lines.append(f"| {row['study'].split('/')[-1]} / {row['trial']['id']} | {row['independent_training_seeds']} | "
                     f"{row['score']:.3f} | {row['minimum_family_points']:.3f} | "
                     f"{'да'if row['quality_diagnostics']['nonregression_gate']else'нет'} | {'да'if row['eligible']else'нет'} |")
    lines+=['','Оценка отбора и очки семейства — не винрейт. Разница по seed является описательной; это не независимая статистическая проверка.', '',
            'Выбраны: '+(', '.join(r['study']+'/'+r['trial']['id']for r in result['selected'])or'нет достаточной пары; требуется следующая гипотеза.'),
            'В JSON сохранены отдельные победы атаки/защиты, потери HP, пустые спеллы и полезные попадания Fireball для каждого seed. Полное обучение ещё не отправлено.','']
    return '\n'.join(lines)


def watch(config_path):
    config=json.loads(Path(config_path).read_text());deadline=time.monotonic()+86400
    while time.monotonic()<deadline:
        if all(Path(row['proof']).exists()for row in config['studies']):
            result=select([row['root']for row in config['studies']],limit=4)
            for key,value in [('output',json.dumps(result,indent=2)+'\n'),('markdown',render(result))]:
                with Path(config[key]).open('x')as f:f.write(value)
            print(json.dumps(dict(event='ECONOMY_SELECTION_COMPLETE',eligible=result['eligible_profiles'],
                                  selected=len(result['selected']),output=config['output'])),flush=True)
            return
        time.sleep(30)
    raise TimeoutError('Local selection deadline; remote jobs unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);watch(p.parse_args().config)
