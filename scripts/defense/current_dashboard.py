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
    ('combined_reward_pilot','Помощь башен × штраф пустых спеллов'),
    ('central_tower_pilot','Пушка в центре + исправленный слот + урон башен'),
    ('tactical_starts_pilot','Разные защитные старты → полный матч'),
)


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return None


def progress(state):
    """Lower bound from matching local observer; never perform a network request."""
    if state.get('status')=='upload_failed_before_submission':
        return '0%', 'проверено локально; загрузка Kaggle не принята'
    snapshot=state.get('progress_snapshot',{})
    if state.get('root'):
        live=read(Path(state['root'])/'live_stream/progress.json')
        if live and live.get('kernel')==state.get('kernel') and live.get('checked_host_ns',0)>snapshot.get('checked_host_ns',0):
            snapshot=live
    remote=read(Path(state['root'])/'collected.status.json')if state.get('root')else None
    collecting=bool(remote and remote.get('kernel')==state.get('kernel')and remote.get('remote_status')=='COMPLETE')
    steps=sum(row.get('steps',0)for row in snapshot.get('trials',{}).values())
    target=state.get('pilot_target_steps',819200)
    if steps and 0<steps<=target:return f'≥{100*steps/target:.1f}%', 'Kaggle завершён; проверяем файлы'if collecting else 'шаги видны в логах'
    return '—', 'Kaggle завершён; проверяем файлы'if collecting else 'отправлено; шаги ещё не подтверждены'


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
    lines+=['','В каждой строке72 одинаковых контекста на двух seed: это не144 независимых ситуации. Между опытами seed различаются; сравниваем прежде всего с их собственным контролем.','']
    ablation=read(reports/'fireball_reward_ablation_summary_20261027.json')
    if read(reports/'combined_reward_pilot_summary_20261027.json'):
        lines+=['**Сочетание наград проверено:** control72/144, tower63/144, spell72/144, combined64/144. Ни один профиль не прошёл порог по слабым группам; большое продолжение не запущено. Сейчас проверяется отдельное обучение с центральной пушкой и исправленным слотом. [Результаты](reports/combined_reward_pilot_results_20261027.md).','']
    elif ablation:
        lines+=['**Последние опыты завершены:** отключение вспомогательной награды Fireball не улучшило результат; расширенные примеры защиты тоже не прошли отбор. Большое продолжение этих вариантов не запущено. Следующий опыт сочетает помощь башен и штраф пустых спеллов; состояние запуска — в таблице выше. [Дизайн](reports/combined_reward_pilot_design_20261028.md), [Fireball](reports/fireball_reward_ablation_results_20261027.md), [защита](reports/pending_diversity_pilot_results_20261027.md).','']
    else:
        lines+=['**Следующая гипотеза:** отделяем пользу примеров Fireball от влияния награды за размен. [Дизайн и измерения](reports/fireball_reward_ablation_design_20261028.md).','']
    recovery=read(reports/'full_candidate_recovered_evaluation_20261028.json')
    stream=read('tmp/kaggle/full_candidate_eval_recovery_20261028/live_stream/progress.json')
    if recovery and recovery.get('total_games_verified')==6300:
        lines+=['**Большой тест и турнир проверены.** [Полный отчёт](reports/defense_plan/full_candidate_recovered_evaluation_20261028.md).','']
        from scripts.attack.report_full_candidate_evaluation import standings
        table=standings(recovery,list(recovery['model_labels']))
        lines+=['| Модель | Победы / 360 | Очки с половиной ничьих | Победы против v25 / 60 |',
                '|---|---:|---:|---:|']
        for name in table['order']:
            opponent=table['matrix'][name].get('v25')
            versus='—'if opponent is None else str(opponent.get('win',0))+'/60'
            lines.append(f"| {recovery['model_labels'][name]} | {table['totals'][name].get('win',0)}/360 | {table['points'][name]:.1%} | {versus} |")
        lines+=['','Турнир в симуляторе на прежней физике и225мс. Небольшой разрыв strong/balanced не доказывает превосходства; реальные винрейты ниже отдельно.','']
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
    strong=read('tmp/live/strong_20261028_live5/series/series.json')
    if strong:
        n=strong['finished'];rate=f"{strong['wins']/n:.1%}"if n and not strong['unknown']else'—'
        replaced=strong['status']=='replaced_after_current_match'
        lines.append(f"| strong · новая серия | {n}/{n if replaced else 5} | {strong['wins']} / {strong['losses']} | {rate} |")
        tower=read('tmp/live/tower_support_20261028_remaining1/series/series.json')
        if replaced and tower:
            count=tower['finished'];rate=f"{tower['wins']/count:.1%}"if count and not tower['unknown']else'—'
            lines.append(f"| Помощь башен0,4 · замена strong | {count}/1 | {tower['wins']} / {tower['losses']} | {rate} |")
            additional=read('tmp/live/tower_support_20261028_live5/series/series.json')
            if additional:
                count=additional['finished'];rate=f"{additional['wins']/count:.1%}"if count and not additional['unknown']else'—'
                lines.append(f"| Помощь башен0,4 · ещё5 по запросу | {count}/5 | {additional['wins']} / {additional['losses']} | {rate} |")
                lines+=['',f"**Продолжение модели с бонусом башен:** отдельно разрешены ещё5 боёв, статус {additional['status']}. Первый пробный бой и4 боя strong считаются отдельно; решения записываются."]
            else:
                lines+=['',f"**По просьбе пользователя модель заменена после4-го боя:** последний из5 — вариант с бонусом помощи башен, статус {tower['status']}. Счёт моделей раздельный; решения записываются."]
        else:
            lines+=['',f"**Новая серия strong:** {strong['status']}, лимит5; решения записываются. Старые серии не возобновлены."]
    lines+=['','Старая серия прервалась из-за связи и отказа входа. У второй старой серии исход6-го неизвестен, ещё4 игры не начаты.','',
        '**Три приоритета:** лишняя блокировка слота убрана; пушка ограничена центром; бонус за урон своих башен добавлен. Новые веса ещё проверяются. [Перебор8288 сцен](reports/central_cannon_coverage_20261028.md): центральных клеток хватает для раннего хога, позднюю реакцию они не исправляют. [Протокол оценки](reports/central_evaluation_execution_20261028.md) обновлён. [Проверка самостоятельной защиты](reports/central_defense_policy_probe_20261028.md): balanced26/64, strong27/64 без урона; это TRAIN-сцены, не винрейт. Соответствие всей симуляции игре пока не доказано.','',
        '**Следующий опыт подготовлен:** разнообразные оплаченные защитные старты с продолжением полного матча. 36 профильных тестов, точный resume,12 одинаковых пар validation и24 завершённых TRAIN-матча с проверкой награды. GPU-запуск — после результатов текущего опыта. [Проверка](reports/tactical_full_games_20261028.md), [подготовка](reports/tactical_starts_preflight_20261028.md).','',
        '[Все прежние таблицы train/val/test и этапы](reports/training_dashboard_history_20261028.md). Исторический снимок содержит прежние статусы; актуальные — выше.','']
    return '\n'.join(lines)
