"""Wait for all three verified pilots, rank distinct profiles and start one full pair.

No live gameplay. Network observations remain in the existing20-minute
observers. This process reads local receipts and submits at most one full job.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.attack.select_pilot_pool import select
from scripts.attack.watch_global_pilot_selection import ready_studies


def rank_complete(config):
    ready = ready_studies(config)
    if len(ready) != 3 or len(ready) != len(config['studies']):
        return None
    ranking = select([r['prepared'] for r in ready], limit=4)
    if len(ranking['ranking']) != 10:
        raise ValueError('Expected ten distinct settings after collapsing shared controls')
    return ranking


def render_ranking(config, ranking):
    from scripts.attack.summarize_clock_pilots import summarize
    summaries = {}
    for entry in config['studies']:
        summaries[entry['prepared']] = summarize(entry['prepared'],
            verified=json.loads(Path(entry['proof']).read_text()))
    lines = ['# Накопление эликсира: отбор PPO', '',
        '24 коротких обучения завершены и проверены. Общий контроль повторён в трёх сериях; '
        'он занимает одну строку. Порядок — по сбалансированной оценке всех групп противников, '
        'не по сумме вспомогательных наград.', '',
        '| Серия / настройка | Победы | Худшая группа на одном seed | Доступно <4 | Пустые спеллы, эликсир / матч | Потери HP (0–3) | Порог |',
        '|---|---:|---:|---:|---:|---:|---|']
    for candidate in ranking['ranking']:
        row = next(r for r in summaries[candidate['prepared']]['rows'] if r['id'] == candidate['trial']['id'])
        lines.append(f"| {row['experiment']} / {row['id']} | {row['wins']}/{row['games']} | "
            f"{candidate['minimum_family_points']:.1%} | {row['available_below4_fraction']:.1%} | "
            f"{row['mean_empty_spell_elixir']:.2f} | {row['mean_own_fraction_loss']:.3f} | "
            f"{'да' if candidate['eligible'] else 'нет'} |")
    lines += ['', 'Каждая настройка:72 validation-контекста на двух тренировочных seed. '
        'Это повторно используемая выборка разработки; test и реальные бои здесь не проводились. '
        'Полное обучение получают только два лучших разных профиля, прошедших минимальный порог '
        'по всем шести группам на обоих seed. Частота карт и ожидание — диагностика, не критерий победы.',
        'Потери HP — сумма потерянных долей здоровья трёх своих башен, от0 до3.',
        '', 'Ограничение: прыжок хога через реку ещё не согласован с реальной игрой. '
        'Ни один результат здесь не объявляет модель лучшей для реальных боёв.']
    return '\n'.join(lines)+'\n'


def continue_full(config, ranking):
    if not ranking['gate_met']:
        return dict(status='completed_no_eligible_pair', full_submitted=False)
    from scripts.kaggle.prepare_global_full_pair import prepare
    from scripts.kaggle.preflight_global_full_pair import preflight, enable_submission
    root = Path(config['full_root'])
    prepare([s['prepared'] for s in config['studies']], root, config['full_owner'])
    preflight(root, root/'actual_preflight')
    enable_submission(root)
    sdk = '/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/python'
    with (root/'submit.log').open('x') as log:
        result = subprocess.run([sdk, '-m', 'scripts.kaggle.submit_prepared_pilot',
            '--root', str(root), '--token-index', str(config['full_token_index'])],
            stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        receipt = root/'launch_receipt.json'
        response = json.loads(receipt.read_text()) if receipt.exists() else {}
        return dict(status='full_submission_rejected' if response.get('error') else 'full_submission_requires_inspection',
                    full_submitted=False, response=response, root=str(root))
    from scripts.kaggle.start_global_full_observers import start
    pids = start(root, 'mana_full_pair0_20261029', config['full_token_index'])
    return dict(status='full_pair_submitted_pending_cuda_proof', full_submitted=True,
                root=str(root), observers=pids, target_steps=8192000, models=2)


def watch(config_path, state_file):
    config = json.loads(Path(config_path).read_text()); state_file = Path(state_file)
    if state_file.exists():
        raise FileExistsError('Inspect existing campaign state before restarting')
    state = dict(status='waiting_for_three_verified_cohorts', full_submitted=False,
                 expected_pilot_runs=24, api_poll_seconds=1200, live_games=False)
    def save():
        state_file.write_text(json.dumps(state, indent=2)+'\n')
        subprocess.run([sys.executable, '-m', 'scripts.defense.update_done'], check=True)
    save()
    try:
        os.nice(10)
    except OSError:
        pass
    deadline = time.monotonic()+172800
    try:
        while time.monotonic() < deadline:
            ranking = rank_complete(config)
            if ranking is not None:
                with Path(config['ranking_report']).open('x') as out:
                    json.dump(ranking, out, indent=2); out.write('\n')
                with Path(config['markdown_report']).open('x') as out:
                    out.write(render_ranking(config, ranking))
                state.update(status='ranked_preparing_full' if ranking['gate_met'] else 'completed_no_eligible_pair',
                    ranking_report=config['ranking_report'], eligible_profiles=ranking['eligible_profiles'])
                save()
                state.update(continue_full(config, ranking)); save()
                return
            ready = ready_studies(config)
            state['verified_pilot_runs'] = 8*len(ready)
            save()
            time.sleep(1200)
        state['status'] = 'local_wait_expired'; save()
    except Exception as exc:
        state.update(status='inspection_required', error_type=type(exc).__name__, message=str(exc))
        save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--state-file', required=True)
    a = p.parse_args(); watch(a.config, a.state_file)
