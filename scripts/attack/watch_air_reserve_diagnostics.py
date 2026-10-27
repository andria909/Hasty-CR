"""Compare verified reserve candidates on four fixed TRAIN full-game traces.

This is a mechanism diagnostic, not independent test or candidate selection.
Only local files/CPU are used; remote polling is owned by existing observers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.attack.audit_air_responses import audit
from scripts.attack.watch_pilot_diagnostics import selected_weights


def summary(report):
    games = report['games']
    total = sum(g['unprotected_frames'] for g in games)
    return dict(checkpoint_sha256=report['checkpoint_sha256'],
        seeds=[g['seed'] for g in games], games=len(games),
        wins=sum(g['metrics']['outcome'] == 'win' for g in games),
        unprotected_frames=total,
        unprotected_bank_below_four=sum(g['unprotected_bank_below_four'] for g in games),
        unprotected_musketeer_legal=sum(g['unprotected_musketeer_legal'] for g in games),
        bank_below_four_fraction=(sum(g['unprotected_bank_below_four'] for g in games) / total if total else None),
        mean_own_tower_fraction_loss=sum(g['metrics']['own_fraction_loss'] for g in games) / len(games),
        mean_card_plays={card:sum(g['metrics']['accepted_card_counts'].get(card, 0) for g in games) / len(games)
                         for card in ('musketeer', 'fireball', 'hog_rider', 'cannon', 'electro_spirit', 'zap')})


def watch(root, state_file, stem='air_reserve_pilot'):
    if stem not in ('air_reserve_pilot', 'air_alternatives_pilot'):
        raise ValueError('Unknown verified study')
    report_stem = stem.removesuffix('_pilot')
    root, state_file = Path(root), Path(state_file)
    if state_file.exists():
        raise FileExistsError('Inspect existing diagnostic worker before restarting')
    root.mkdir(parents=True, exist_ok=False)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = dict(status='waiting_for_verified_eight_candidates', remote_api_calls=False,
                 training_performed=False, quality_accepted=False)
    def save():
        state_file.write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        os.nice(10)
    except OSError:
        pass
    deadline = time.monotonic() + 172800
    proof_path = Path(f'reports/defense_plan/{stem}_verified_20261027.json')
    while not proof_path.exists():
        if time.monotonic() >= deadline:
            state['status'] = 'local_wait_expired_remote_unchanged'
            save()
            return
        time.sleep(30)
    try:
        proof = json.loads(proof_path.read_text())
        candidates = selected_weights(proof)
        base = json.loads(Path('reports/defense_plan/balanced_air_response_trace_20261027.json').read_text())
        rows = [dict(id='balanced_reference', **summary(base))]
        for candidate in candidates:
            state.update(status='evaluating_verified_candidate', current=candidate['id'])
            save()
            report = audit(candidate['checkpoint'], root / (candidate['id'] + '.json'), games=4)
            row = summary(report)
            if row['seeds'] != rows[0]['seeds'] or row['checkpoint_sha256'] != candidate['checkpoint_sha256']:
                raise ValueError('Weight or diagnostic context identity differs')
            if report['source_sha256'] != base['source_sha256']:
                raise ValueError('Diagnostic source changed since reference')
            rows.append(dict(id=candidate['id'], **row))
        result = dict(schema='air-reserve-full-game-train-diagnostic-v1', rows=rows,
            verified_proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),
            scope='Four fixed TRAIN games per model; descriptive mechanism audit, not independent quality test',
            limitations=['Frames are correlated', 'Four games do not establish a reliable win-rate',
                         'Lower spending alone does not prove better defense',
                         'Any friendly Musketeer counts, including one in the other lane',
                         'Absence of Musketeer does not imply undefended air; Cannon and helper cards can defend too'],
            quality_accepted=False, live_games_started=False)
        destination = Path(f'reports/defense_plan/{report_stem}_full_game_diagnostics_20261027.json')
        with destination.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        lines = ['# Полные TRAIN-партии: банк и ответ на воздух', '',
                 'Четыре одинаковых диагностических партии на модель. Это не независимый test.', '',
                 '| Модель | Победы /4 | Банк<4 при воздухе без мушкетёра | Мушкетёр /игру | Потеря HP башен |',
                 '|---|---:|---:|---:|---:|']
        for row in rows:
            fraction = row['bank_below_four_fraction']
            rate = '—' if fraction is None else f'{100 * fraction:.1f}%'
            lines.append(f"| {row['id']} | {row['wins']} | {rate} | {row['mean_card_plays']['musketeer']:.2f} | {row['mean_own_tower_fraction_loss']:.3f} |")
        lines += ['', 'Кадры зависимы. Меньшая частота нехватки банка не доказывает полезность ожидания: смотрим также потери башен и полный validation. Сумма потерь HP — по всем своим башням, в долях их начального HP.']
        with Path(f'reports/{report_stem}_full_game_diagnostics_20261027.md').open('x') as stream:
            stream.write('\n'.join(lines) + '\n')
        state.update(status='complete', report=str(destination))
        save()
    except Exception as error:
        state.update(status='error_requires_inspection', error_type=type(error).__name__, message=str(error))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--state-file', required=True)
    parser.add_argument('--stem', choices=('air_reserve_pilot','air_alternatives_pilot'), default='air_reserve_pilot')
    args = parser.parse_args()
    watch(args.root, args.state_file, args.stem)
