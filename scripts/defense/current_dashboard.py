"""Compact current dashboard; historical measurements stay in linked reports."""
import json
from pathlib import Path


STUDIES = (
    ('tower_support_motionfix','Помощь башен'),
    ('spell_waste_motionfix','Штраф пустых спеллов'),
    ('native_fireball_slot_motionfix','Прицел Fireball'),
    ('pending_sequence_pilot','Последовательная защита'),
    ('pending_diversity_pilot','Разнообразие рук и слотов'),
    ('fireball_reward_ablation','Награда Fireball × обучение на примерах'),
)


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return None


def progress(state):
    """Lower bound from matching local observer; never perform a network request."""
    snapshot=state.get('progress_snapshot',{})
    if state.get('root'):
        live=read(Path(state['root'])/'live_stream/progress.json')
        if live and live.get('kernel')==state.get('kernel') and live.get('checked_host_ns',0)>snapshot.get('checked_host_ns',0):
            snapshot=live
    steps=sum(row.get('steps',0)for row in snapshot.get('trials',{}).values())
    target=state.get('pilot_target_steps',819200)
    if steps and 0<steps<=target:return f'≥{100*steps/target:.1f}%', 'шаги видны в логах'
    return '—', 'отправлено; шаги ещё не подтверждены'


def render(reports):
    reports=Path(reports)
    lines=['# Текущий этап','',
        '**Улучшаем PPO после ошибок реальных боёв. Новая модель для live пока не принята.**', '',
        '| Короткое обучение | Прогресс | Результат / состояние |','|---|---:|---|']
    completed=[]
    for stem,label in STUDIES:
        state=read(reports/f'{stem}_training_20261027.json')
        summary=read(reports/f'{stem}_summary_20261027.json')
        proof=read(reports/f'{stem}_verified_20261027.json')
        if proof and proof.get('pilot_steps_verified')==819200:
            accepted=summary and summary.get('eligible_for_full')
            note='порог продолжения пройден'if accepted else 'порог продолжения не пройден'if summary else 'результаты проверяются'
            lines.append(f'| {label} | 100% | {note} |')
            if summary:completed.append((stem,label,summary))
        elif state:
            pct,note=progress(state);lines.append(f'| {label} | {pct} | {note} |')
    lines+=['','Каждая серия:4 варианта ×2 seed,819200 PPO-шагов. Процент — объём обучения, не качество. Проверка Kaggle раз в20 минут.','',
        '**Завершённые короткие опыты — победы на validation:**','',
        '| Опыт | Контроль | Остальные варианты |','|---|---:|---|']
    for stem,label,summary in completed:
        rows=summary['rows'];control=next((r for r in rows if r['id']=='control'),None)
        score=lambda r:f"{r['wins']}/{r['games']}"
        values='; '.join(r['id']+': '+score(r)for r in rows if r is not control)
        lines.append(f'| [{label}](reports/{stem}_results_20261027.md) | {score(control)if control else "—"} | {values} |')
    lines+=['','В каждой строке72 одинаковых контекста на двух seed: это не144 независимых ситуации. Между опытами seed различаются; сравниваем прежде всего с их собственным контролем.','',
        '**Следующая гипотеза:** отделяем пользу примеров Fireball от влияния награды за размен. [Дизайн и измерения](reports/fireball_reward_ablation_design_20261028.md).','']
    recovery=read(reports/'full_candidate_recovered_evaluation_20261028.json')
    stream=read('tmp/kaggle/full_candidate_eval_recovery_20261028/live_stream/progress.json')
    if recovery and recovery.get('total_games_verified')==6300:
        lines+=['**Большой тест и турнир проверены.** [Полный отчёт](reports/defense_plan/full_candidate_recovered_evaluation_20261028.md).','']
    elif stream and stream.get('kernel')=='vertel0/hastycr-full-eval-recovery-1028':
        games=stream.get('evaluation',{}).get('games',{})
        if games.get('total')==6300:
            lines += [f"**Большой тест + турнир:** по логам ≥{games['completed']}/6300 ({100*games['completed']/6300:.1f}%). Проверены5040 тестовых игр; итог турнира ещё не подтверждён.",'']
    partial=read(reports/'full_candidate_partial_verified_20261028.json')
    if partial:
        lines += ['**Большие модели — проверенный test на старой физике,225мс:**','',
            '| Модель | Сценарные пуши | Случайный легальный соперник |','|---|---:|---:|']
        for key,label in partial['labels'].items():
            scores=[]
            for kind in ('scripted_stress','legal_random_compatibility'):
                row=partial['tests'][f'test__{kind}__{key}.json'];wins=row['counts'].get('win',0);games=row['games']
                scores.append(f'{wins}/{games} — {wins/games:.1%}')
            lines.append('| '+label+' | '+' | '.join(scores)+' |')
        lines+=['','Это симуляция; цифры не сопоставимы напрямую с новыми опытами1350мс или реальными боями. [Подробности](reports/full_candidate_partial_tests_20261028.md).','']
    lines+=['**Последние реальные серии:**','',
        '| Модель | Завершено | Победы / поражения | Винрейт |','|---|---:|---:|---:|']
    for slug,label in [('full_bc800','Накопление · BC800'),('full_cannon_spells_replay1','Пушка + спеллы · повторение1')]:
        live=read(Path('tmp/live/full_top2_20261028')/slug/'series/series.json')
        if live:
            n=live['finished'];rate=f"{live['wins']/n:.1%}"if n and not live['unknown']else'—'
            lines.append(f"| {label} | {n}/10 | {live['wins']} / {live['losses']} | {rate} |")
    lines+=['','Бои остановлены: разрыв связи и отказ входа по региону. У второй серии исход6-го неизвестен, ещё4 игры не начаты.','',
        '**Исправлено:** асимметрия движения —0 расхождений в908 зеркальных парах; запись точных входов модели готова. Live-маска пушки ещё не включена. Соответствие всех механик реальной игре не доказано.','',
        '[Все прежние таблицы train/val/test и этапы](reports/training_dashboard_history_20261028.md). Исторический снимок содержит прежние статусы; актуальные — выше.','']
    return '\n'.join(lines)
