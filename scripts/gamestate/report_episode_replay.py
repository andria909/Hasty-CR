"""Reproduce native/simulator overlays from the small committed episode clips."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from scripts.gamestate.replay_recorded_episode import replay


def report(directory):
    rows = []
    for path in sorted(Path(directory).glob('*.json')):
        clip = json.loads(path.read_text())
        result = replay(clip)
        result['clip'] = str(path)
        result['clip_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        uid = clip['focal_identity'][0]
        motions, hp_drops = [], []
        previous = None
        for frame in clip['frames']:
            bodies = {b['entity_id']: b for b in frame['bodies']}
            if previous:
                old_tick, old_bodies = previous
                before, now = old_bodies[uid], bodies[uid]
                speed = math.hypot(now['x']-before['x'], now['y']-before['y'])*20/(frame['tick']-old_tick)
                motions.append(dict(first_tick=old_tick, last_tick=frame['tick'], speed_mt_s=speed,
                    river_jump_interior=(14700 <= min(before['y'], now['y'])
                                         and max(before['y'], now['y']) <= 16300)))
                for target_uid, body in bodies.items():
                    if body['name'] not in ('KingTower', 'PrincessTower', 'Cannon'):
                        continue
                    old = old_bodies.get(target_uid)
                    if old and old['hp'] > body['hp'] >= 0:
                        hp_drops.append(dict(target_uid=target_uid, name=body['name'],
                            interval_ticks=[old_tick, frame['tick']], amount=old['hp']-body['hp']))
            previous = frame['tick'], bodies
        result['native_motion_intervals'] = motions
        result['native_building_hp_drops_unattributed'] = hp_drops
        fast = [m['speed_mt_s'] for m in motions if m['river_jump_interior'] and m['speed_mt_s'] > 0]
        result['native_river_interior_speed_median_mt_s'] = median(fast) if fast else None
        rows.append(result)
    paths = ['sim/engine.py', 'sim/entities.py', 'sim/arena.py', 'sim/pathfind.py',
        'sim/gamedata.py', 'sim/public_gamedata.py', 'sim/projectile_calibration.py',
        'scripts/gamestate/replay_recorded_episode.py',
        'scripts/gamestate/report_episode_replay.py',
        'data/royaleapi/cards_stats_characters.json',
        'data/royaleapi/cards_stats_building.json', 'data/royaleapi/cards_stats_projectile.json']
    return dict(schema='recorded-episode-replay-report-v1', episodes=rows,
        source_fingerprints={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        physics_fully_validated=False, new_live_games=False, new_training=False)


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, len(result['episodes']), figsize=(13, 9),
                             gridspec_kw=dict(height_ratios=[2, 1]))
    titles = dict(hog_cannon_one_tile_left_miss='Хог: пушка не отвлекла в реальности',
                  hog_cannon_right_pull='Хог: пушка отвлекла в реальности',
                  musketeer_cannon_then_tower='Мушкетёр: пушка → башня')
    for column, episode in enumerate(result['episodes']):
        ax, error_ax = axes[0, column], axes[1, column]
        native = [r['native_xy'] for r in episode['rows']]
        predicted = [r['simulated_xy'] for r in episode['rows']]
        for points, label, color in [(native, 'Реальный лог', '#1677bb'), (predicted, 'Симулятор', '#e48720')]:
            ax.plot([p[0]/1000 for p in points], [p[1]/1000 for p in points],
                    '.-', markersize=3, label=label, color=color)
            ax.scatter(points[0][0]/1000, points[0][1]/1000, color=color, s=60, marker='o')
        clip = json.loads(Path(episode['clip']).read_text())
        for b in clip['frames'][0]['bodies']:
            if b['name'] in ('Cannon', 'PrincessTower', 'KingTower'):
                ax.scatter(b['x']/1000, b['y']/1000, marker='s', s=65,
                           color='#a64d4d' if b['name'] == 'Cannon' else '#88909a')
        ax.add_patch(Rectangle((0, 15.5), 18, 1, color='#d9edf7', alpha=.65))
        ax.set(xlim=(0, 18), ylim=(27, 12), aspect='equal', title=titles[episode['name']], xlabel='x, клетки', ylabel='y, клетки')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9, loc='lower left')
        start = episode['rows'][0]['tick']
        error_ax.plot([(r['tick']-start)*.05 for r in episode['rows']],
                      [r['error_tiles'] for r in episode['rows']], color='#9c3c30')
        error_ax.set(xlabel='Игровые секунды от начала фрагмента', ylabel='Ошибка позиции, клетки')
        error_ax.grid(alpha=.2)
    fig.suptitle('Сопоставление сохранённых эпизодов с движком', fontsize=16)
    fig.text(.5, .015, 'Один юнит движется в симуляторе; окружение следует реальному логу.\n'
             'Начальные цели и таймеры атак неизвестны. Это диагностика пути, не полный повтор матча.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(output)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips', type=Path, default=Path('data/recorded_mechanics_20261028'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    result = report(args.clips)
    if args.plot and args.plot.exists():
        raise FileExistsError(args.plot)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    if args.plot:
        plot(result, args.plot)
    print(json.dumps([dict(name=r['name'], mean_error=r['mean_position_error_tiles'],
        max_error=r['max_position_error_tiles'], river_speed=r['native_river_interior_speed_median_mt_s'])
        for r in result['episodes']]))
