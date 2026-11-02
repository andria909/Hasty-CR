"""Compare observable real-match reward terms with identical simulator terms.

Does not estimate tower damage attribution or spell hits from geometry.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def common_reward(own, enemy, outcome, profile):
    if outcome not in ('win', 'loss', 'draw'):
        return None
    return (profile['enemy_hp_reward'] * enemy - profile['own_hp_penalty'] * own
            + (profile['win_reward'] if outcome == 'win' else
               -profile['loss_penalty'] if outcome == 'loss' else 0))


def tower_loss(initial, final, side, *, missing_princess_confirmed):
    """Vanished towers require terminal crown evidence; missing king is unknown."""
    old = {u['entity_id']: u for u in initial['units']
           if u['name'] in ('KingTower', 'PrincessTower')}
    if len(old) != 6 or sum(u['side'] == side for u in old.values()) != 3:
        return None
    new = {u['entity_id']: u for u in final['units']}
    total = 0.
    for key, u in old.items():
        if u['side'] != side:
            continue
        maximum = u['max_hp']
        if maximum <= 0 or u['hp'] != maximum:
            return None  # Mid-game attachment is not a full-match return.
        v = new.get(key)
        if v is None:
            if u['name'] != 'PrincessTower' or not missing_princess_confirmed:
                return None
            hp = 0
        else:
            if (v['side'], v['name'], v['max_hp']) != (u['side'], u['name'], maximum):
                return None
            hp = v['hp']
        if not 0 <= hp <= maximum:
            return None
        total += 1 - hp / maximum
    return total


def aggregate(records):
    valid = [r['common_reward'] for r in records if r['common_reward'] is not None]
    return dict(matches=len(records), measured=len(valid),
                outcomes=dict(Counter(r['outcome'] for r in records)),
                mean_common_reward=statistics.mean(valid) if valid else None,
                common_reward_min=min(valid) if valid else None,
                common_reward_max=max(valid) if valid else None)


def summarize(series_path, selected_path):
    series = json.loads(Path(series_path).read_text())
    selected_path = Path(selected_path)
    selected = json.loads(selected_path.read_text())
    checkpoint = Path(series['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint differs from real series')
    config = json.loads((checkpoint.parent/'config.json').read_text())
    profile = config['continuous_profile']
    real = []
    for match in series['matches']:
        run = Path(match['run'])
        events = list(rows(run/'events.jsonl'))
        start = next(e for e in events if e['event']=='MATCH_START' and e['match']==match['local_match'])
        end = next(e for e in events if e['event']=='MATCH_END' and e['match']==match['local_match'])
        accepted = [f for f in rows(run/'frames.jsonl')
                    if start['host_ns']-100_000_000 <= f['received_host_ns'] <= end['host_ns']
                    and f['status']=='ok']
        if not accepted:
            raise ValueError('Completed match lacks accepted frames')
        initial, final = accepted[0], accepted[-1]
        crown_evidence = match['evidence'] in ('frozen_terminal_crown_body_comparison', 'king_hp_zero')
        own = tower_loss(initial, final, start['side'], missing_princess_confirmed=crown_evidence)
        enemy = tower_loss(initial, final, 1-start['side'], missing_princess_confirmed=crown_evidence)
        if match['result'] == 'unknown':
            # The last frame before a disconnect is not the terminal board.
            own = enemy = None
        score = common_reward(own, enemy, match['result'], profile) if own is not None and enemy is not None else None
        acks = Counter(e['card'] for e in events if e['event']=='PLAY_ACK' and e['match']==match['local_match'])
        plays = [e for e in events if e['event']=='PLAY' and e['match']==match['local_match']]
        times = [e['timing']['parser_request_to_input_return_ms'] for e in plays]
        real.append(dict(number=match['number'], outcome=match['result'], common_reward=score,
                         own_fraction_loss=own, enemy_fraction_damage=enemy,
                         confirmed_cards=dict(acks), input_attempts=len(plays),
                         mean_read_to_input_return_ms=statistics.mean(times) if times else None,
                         initial_tick=initial['frame'].get('battle_tick'),
                         final_tick=final['frame'].get('battle_tick')))
    def sim_record(r):
        return dict(outcome=r['outcome'], own_fraction_loss=r['own_fraction_loss'],
                    enemy_fraction_damage=r['enemy_fraction_damage'],
                    common_reward=common_reward(r['own_fraction_loss'], r['enemy_fraction_damage'], r['outcome'], profile))
    validation = [sim_record(r) for r in selected['validation']['records']]
    train = [sim_record(r) for m in rows(checkpoint.parent/'metrics.jsonl')
             if m['update'] <= selected['selected_update'] for r in m.get('episodes', [])]
    return dict(series=str(series_path), series_status=series['status'], requested=series['requested'],
                checkpoint_sha256=selected['checkpoint_sha256'],
                coefficients={k:profile[k] for k in ('own_hp_penalty','enemy_hp_reward','win_reward','loss_penalty')},
                real=real, comparison={'real':aggregate(real), 'selected_validation':aggregate(validation),
                                      'training_up_to_selected_update':aggregate(train)},
                unavailable_live_terms=['tower_damage_attribution_bonus', 'spell_value_and_waste',
                                        'reserve_shaping', 'overflow_full_return', 'illegal_action_penalty'],
                limitations=['Partial undiscounted score only: tower HP fractions plus outcome; not total PPO reward.',
                             'Training episodes use changing stochastic policies, not evaluation of the selected checkpoint.',
                             'Real opponents, levels and durations differ from simulator development validation.',
                             'PLAY_ACK proves hand rotation, not spell impact or uncontaminated manual control.'])


def render(report):
    lines=['# Реальные бои и симулятор: бонус башен 0,15','',
           f"Серия: `{report['series_status']}`; запрошено {report['requested']} боёв. "
           'Неизвестный исход после разрыва связи не считается поражением или победой.','',
           '| Источник | Победы / известные исходы | Неизвестно | Средняя сопоставимая награда |',
           '|---|---:|---:|---:|']
    for key,label in [('real','Реальные бои'),('selected_validation','Validation выбранных весов'),
                      ('training_up_to_selected_update','Train до выбранного обновления')]:
        row=report['comparison'][key];unknown=row['outcomes'].get('unknown',0)
        score=row['mean_common_reward'];shown='—' if score is None else f'{score:.2f}'
        lines.append(f"| {label} | {row['outcomes'].get('win',0)}/{row['matches']-unknown} | {unknown} | {shown} |")
    lines+=['','Сравнивается только `3 × снятые доли HP врага − 6 × потерянные доли своих HP ±20 за исход`. '
            'Доли суммируются по трём башням. Это часть награды, не полный return PPO. '
            'Train — меняющаяся стохастическая политика, а validation — выбранный чекпойнт; '
            'реальные соперники, уровни и длительность отличаются.','',
            'Бонус за урон именно своих башен, попадания/пустые спеллы, полный перелив эликсира '
            'и shaping достоверно не восстановлены. У прерванного боя итоговые HP неизвестны.','',
            'Две проверки в тренировочном лагере не входят в серию. После первого боя заменён '
            'эталон результата, после третьего изменился вид кнопки Battle. В четвёртом были '
            'длительные разрывы наблюдения, затем эмулятор стал недоступен. Счёт серии сохранён.','',
            '[Парсер 160635007](parser_160635007_20261102.json) · '
            '[Числа и ограничения](live_tower015_rewards_20261102.json)','',
            f"Чекпойнт SHA256: `{report['checkpoint_sha256']}`."]
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--series',type=Path,required=True)
    p.add_argument('--selected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--markdown',type=Path)
    a=p.parse_args()
    result=summarize(a.series,a.selected)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    if a.markdown:a.markdown.write_text(render(result))
    print(json.dumps(result['comparison'],indent=2))


if __name__=='__main__':
    main()
