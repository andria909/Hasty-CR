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
    ('normal_range_pilot','Исправленная дальность + защитные старты'),
    ('mana_reserve_pilot','Накопление эликсира · 8 прогонов'),
    ('mana_balance_pilot','Ослабление награды за частоту карт · 8 прогонов'),
    ('mana_wait_pilot','Исследование WAIT и доступных карт · 8 прогонов'),
)


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return None


def progress(state):
    """Lower bound from matching local observer; never perform a network request."""
    if state.get('status')=='invalidated_teacher' or state.get('promotion_blocked'):
        if state.get('remote_run_ended'):
            return 'исключён', 'GPU-задача завершилась; модели не допускаются к продолжению'
        return 'исключён', 'учитель пропустил пушку; остановка GPU пока не подтверждена'
    if state.get('status')=='upload_failed_before_submission':
        return '0%', 'проверено локально; загрузка Kaggle не принята'
    if state.get('status')=='preflight':
        return '0%', 'техническая проверка; GPU ещё не отправлен'
    if state.get('status')=='uploading':
        return '0%', 'загружаем закреплённый пакет; GPU ещё не принят'
    if state.get('status')=='queued_capacity':
        queue=read(Path(state['root'])/'capacity_queue.json')
        if queue and queue.get('status')=='inspection_required':return '0%', 'ошибка очереди; требуется разбор'
        return '0%', 'очередь: ждёт свободного GPU-слота'
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


def render_crown(reports,state):
    snapshot=state.get('progress_snapshot',{})
    trials=dict(snapshot.get('trials',{}))
    fresh=read(Path(state['root'])/'live_stream/progress.json') or {}
    if fresh.get('kernel')==state['kernel']:
        for key,row in fresh.get('trials',{}).items():
            if row.get('steps',0)>=trials.get(key,{}).get('steps',0):trials[key]=row
    proof=read(reports/'crown_economy_pilot_verified_20261030.json') or {}
    expected={f'pilot_{name}_s{seed}'for name in ('control','low','medium','high')for seed in (382,383)}
    verified=(proof.get('kernel')==state['kernel'] and proof.get('pilot_steps_verified')==819200 and
              {m['id']for m in proof.get('pilot_models',[])}==expected)
    steps=sum(min(102400,trials.get(key,{}).get('steps',0))for key in expected)
    total='100% бюджета проверено' if verified else f'≥{100*steps/819200:.1f}%' if steps else 'шаги пока не подтверждены'
    summary=read(reports/'crown_economy_pilot_summary_20261030.json') or {}
    scores={r['id']:f"{r['wins']}/{r['games']}"for r in summary.get('rows',[])} if verified and summary.get('kernel')==state['kernel'] else {}
    lines=['# Текущий этап','',f'**PPO: расход спелов по башням — {total}.** 8 прогонов, общий бюджет 819 200 шагов.','',
           '| Настройка штрафа | Seed 382 | Seed 383 | Победы validation |','|---|---:|---:|---:|']
    for name,label in [('control','Контроль'),('low','Спящий король: 0,1'),('medium','Спящий король: 0,4'),('high','Король: 0,4 + прочие башни: 0,1')]:
        values=[]
        for seed in (382,383):
            row=trials.get(f'pilot_{name}_s{seed}')
            values.append('100% проверено' if verified else f"≥{100*row['steps']/102400:.1f}%"if row else'—')
        lines.append(f"| {label} | {values[0]} | {values[1]} | {scores.get(name,'—')} |")
    timing=read(reports/'bank_timing_stage_20261030.json') or {}
    init=read(reports/'bank_initialization_training_20261030.json')
    if init:
        stream=read(Path(init['root'])/'live_stream/progress.json') or init.get('progress_snapshot',{})
        count=sum(r.get('updates',0)for r in stream.get('initializations',{}).values())
        init_proof=read(reports/'bank_initialization_verified_20261030.json') or {}
        init_status=('6400 обновлений проверены'if init_proof.get('kernel')==init['kernel'] and
                     init_proof.get('supervised_updates_verified')==6400 else
                     f'≥{100*count/6400:.1f}% обновлений'if count else'обновления пока не подтверждены')
        lines+=['',f'**BC-подбор инициализации:** {init_status}; 4 варианта × 2 seed, локальные навыки. '
                '[План и данные](reports/bank_initialization_study_20261030.md).']
    margin=read(reports/'bank_initialization_margin_training_20261030.json')
    if margin:
        progress=read(Path(margin['root'])/'live_stream/progress.json') or {}
        count=sum(r.get('updates',0)for r in progress.get('initializations',{}).values()) if progress.get('kernel')==margin['kernel'] else 0
        label=f'≥{100*count/6400:.1f}% обновлений'if count else margin.get('status','подготовлен')
        lines+=['',f'**Новый BC на исправленных данных:** {label}; 1060 состояний, запас по задержке и несколько безопасных клеток.']
    teacher=(f"{timing['mixed_examples']} состояний, 4 навыка; проверяем запас по задержке перед PPO."
             if timing.get('mixed_examples')else
             (f"{timing['teacher_examples']} состояний из 36 воспроизведённых защит; "if timing.get('teacher_examples')else'Собираем безопасные траектории; ')+
             'уточняем границу постановки с шагом 250 мс.')
    next_action=('Все профили провалили семейные пороги; большого продолжения нет. Проверяем смешанное BC и сброс старого приора.'
                 if verified and not proof['pilot_selection']['gate_met']else
                 'Сравнить 8 прогонов по победам, всем 6 семействам, запасу и полезности спелов; 2–4 прошедших порог кандидата — на полный бюджет.')
    if margin:
        teacher='1060 состояний, 4 навыка. Для накопления: запас 750 мс; пустые Fireball/Zap покрыты во всех слотах каждой группы.'
        next_action='Проверить новый короткий BC; затем замкнутую защиту и вторую оплаченную волну. Большое продолжение пока не допущено.'
    lines+=['','Проценты показывают объём, не качество; «—» означает отсутствие подтверждённых данных. Проверка Kaggle раз в 20 минут.','',
        '**Учитель:** '+teacher+' Общий учитель для PPO ещё не готов.','',
        '**Реальные бои выключены.** Последняя проба BC200: 1 поражение, 4 боя отменены.','',
        '**Дальше:** '+next_action,'',
        '[Параметры опыта](reports/crown_economy_pilot_20261030.md) · [Учитель и разбор](reports/bank_timing_and_live_failure_20261030.md) · '
        '[Прежние 24 прогона](reports/mana_campaign_results_20261029.md) · [План](plan_2.md)','']
    return '\n'.join(lines)


def render_mana(reports,campaign):
    lines=['# Текущий этап','',
        '**24 коротких PPO-обучения: накопление эликсира и осмысленный выбор карты.**',
        'Реальные бои остановлены. На аккаунтах 1 и 3 исчерпана GPU-квота; используем аккаунт 2 с лимитом двух заданий одновременно.','',
        '| Серия | Прогресс | Состояние |','|---|---:|---|']
    bank=read(reports/'bank_teacher_pilot_training_20261027.json')
    cannon=read(reports/'bank_cannon_search_launch_20261029.json')
    if bank:
        pct,note=progress(bank)
        title='**Учитель банка исключён: найден неверный WAIT.** ' if bank.get('promotion_blocked') else '**Новый этап: учим сохранять банк и вовремя защищаться.** '
        lines[2:2]=[title+
            f'8 PPO-прогонов; {pct}, {note}. '
            '60 парных учебных примеров, отложена целая группа позиций. '
            '[Дизайн](reports/bank_teacher_20261029.md).','']
    if cannon:
        cannon_status=('Оба поиска завершены и собраны:7956 вариантов. ' if cannon.get('all_collected') else 'Два поиска работают на CPU Kaggle. ')
        lines[2:2]=['**Сейчас: исправляем учителя через поиск защиты пушкой.** '
            '72 ситуации: одиночный Balloon и Balloon+Baby Dragon, разные банк/рука, все36 клеток центра и поддержки. '
            +cannon_status+'[Контрпример и следующий этап](reports/bank_cannon_teacher_repair_20261029.md).','']
    timing=read(reports/'bank_timing_stage_20261030.json')
    if timing:
        timing_note=('Найдена ошибка упаковки первого CPU-поиска; исправленный пакет прошёл изолированную проверку. '
                     if timing.get('packaging_fault_found') and timing.get('repair_status')!='submitted' else
                     'CPU-поиск36 контекстов и шести задержек принят Kaggle; ')
        lines[2:2]=['**Сейчас: учим сохранять эликсир через проверку безопасного ожидания.** '
            +timing_note+'Учитель траекторий ещё не использован в обучении. '
            'Локально подтверждена защита без урона после3с накопления. '
            '[Дизайн и следующий шаг](reports/bank_timing_and_live_failure_20261030.md).','',
            '**BC200 в реальной игре:** 1 поражение; ещё4 боя отменены по команде пользователя. '
            '6 подтверждённых Fireball,0 Musketeer; в97,7% решений после5-й секунды доступно менее4 эликсира. '
            'Бот выключен. [Разбор](reports/bank_timing_and_live_failure_20261030.md).','']
    crown=read(reports/'crown_economy_pilot_training_20261030.json')
    if crown:
        pct,note=progress(crown)
        lines[2:2]=['**Новый опыт: расход спелов по башням — 8 коротких PPO-прогонов.** '
            f'{pct}; {note}. 4 настройки ×2 seed ×102400 шагов. '
            'Добивание и попадание по юнитам исключены из нового штрафа. '
            '[Параметры и проверки](reports/crown_economy_pilot_20261030.md).','']
    for kind,label in [('reserve','Накопление эликсира'),('balance','Меньше давления на частоту карт'),('wait','Исследование WAIT и доступных карт')]:
        stem=f'mana_{kind}_pilot'
        state=read(reports/f'{stem}_training_20261027.json')
        proof=read(reports/f'{stem}_verified_20261027.json')
        if proof and proof.get('pilot_steps_verified')==819200:
            percent,note='100%','веса и все шаги проверены'
        elif state:percent,note=progress(state)
        else:percent,note='0%','подготовлена'
        lines.append(f'| {label} | {percent} | {note} |')
    lines+=['','Каждая серия: 4 настройки × 2 seed × 102 400 шагов. Всего 2 457 600 шагов. '
        'Процент показывает объём обучения, не качество. Проверка Kaggle раз в 20 минут.','',
        '**Общее для всех:** пушка только в центре, бонус урона своих башен 0,15, штраф пустых спеллов 0,1, задержка команды 1350 мс.','']
    if campaign.get('ranking_report'):
        label='все24 прогона проверены; ни один профиль не прошёл порог всех групп' if campaign['status']=='completed_no_eligible_pair' else campaign['status']
        lines += [f"**Отбор:** {label}. [Таблица результатов](reports/mana_campaign_results_20261029.md).",'']
        if campaign['status']=='completed_no_eligible_pair':
            lines += ['Лучший по победам: уменьшение баланса карт до0,03 — **37/144 (25,7%)**, контроль **28/144 (19,4%)**. '
                'Усиление накопления и исследование WAIT проблему расхода эликсира не решили. Большое продолжение не запущено.','']
    else:
        lines += ['**Дальше:** после проверки всех 24 прогонов сравнить 10 разных настроек. '
            'Общий контроль учитывается один раз. Два лучших профиля с результатом на обоих seed '
            'и без нулевой защиты одной из групп получают по 4 096 000 шагов при доступной GPU-квоте.','']
    full=read(reports/'bank_teacher_full_pair0_20261029_training_20261027.json') or read(reports/'mana_full_pair0_20261029_training_20261027.json')
    if full:
        snapshot=read(Path(full['root'])/'live_stream/progress.json') or full.get('progress_snapshot',{})
        trials=snapshot.get('trials',{})
        lines+=['| Полное обучение | Прогресс |','|---|---:|']
        for child in full['children']:
            row=trials.get(child['id'],{});steps=row.get('steps',0)
            lines.append(f"| {child['id']} | ≥{100*steps/4096000:.1f}% |")
        lines.append('')
    lines+=['**Ограничение:** прыжок хога через реку ещё отличается от игры. '
        'Эти опыты проверяют расход эликсира на закреплённой текущей физике; новых реальных винрейтов нет.','',
        '[Параметры и схема отбора](reports/mana_campaign_20261029.md) · '
        '[Разбор последнего реального боя](reports/balanced_offline_diagnosis_20261029.md) · '
        '[План](plan_2.md) · [Прежние результаты](reports/training_dashboard_history_20261028.md)','']
    return '\n'.join(lines)


def render_bank_campaign(reports,campaign):
    lines=['# Текущий этап','',
        '**Сравниваем исходные веса на исправленных демонстрациях защиты и накопления.**','',
        '| Исходная модель | BC-обновления | Состояние |','|---|---:|---|']
    for item in campaign['studies']:
        state=read(reports/item['status_file']) or {}
        verified=read(reports/item['verified_file']) or {}
        stream=read(Path(item['root'])/'live_stream/progress.json') or {}
        names={f'{name}_s{seed}'for name in ('retain01','retain001','retain0','zero01')for seed in (392,393)}
        complete=(verified.get('kernel')==item['kernel'] and verified.get('supervised_updates_verified')==6400
                  and {r['id']for r in verified.get('runs',[])}==names)
        count=(sum(min(800,r.get('updates',0))for key,r in stream.get('initializations',{}).items()if key in names)
               if stream.get('kernel')==item['kernel']else 0)
        pct='100% проверено'if complete else f'≥{100*count/6400:.1f}%'if count else'—'
        status='результаты проверены'if complete else state.get('status','подготовлен')
        lines.append(f"| {item['label']} | {pct} | {status} |")
    depth=read(reports/'bank_fit_depth_training_20261101.json')
    if depth:
        stream=read(Path(depth['root'])/'live_stream/progress.json')or{}
        expected={f'{mode}{budget}_s{seed}':budget for mode in ('mixed','isolated')for budget in (800,8000)for seed in (402,403)}
        count=sum(min(expected[name],max(0,r.get('updates',0)))for name,r in stream.get('fit_depth',{}).items()
            if name in expected and r.get('actual_cuda'))if stream.get('kernel')==depth['kernel']else 0
        proof=read(reports/'bank_fit_depth_verified_20261101.json')or{}
        complete=(proof.get('schema')=='verified-bank-fit-depth-v1'and proof.get('kernel')==depth['kernel']
            and proof.get('actual_cuda_verified')is True and proof.get('supervised_updates_verified')==35200
            and {r['id']for r in proof.get('runs',[])}==set(expected))
        pct='100% проверено'if complete else f'≥{count/352:.1f}%'if count else '—'
        status=('локальные результаты проверены'if complete else 'обновления завершены; проверяем веса'if count==35200
            else 'обучается на GPU'if count else depth.get('status','подготовлен'))
        lines.append(f"| Длительность × смесь навыков | {pct} | {status} |")
    pair=read(reports/'bank_correction_pair_training_20261101.json')
    pair_summary=None
    if pair:
        snapshot=read(Path(pair['root'])/'live_stream/progress.json')or{}
        proof=read(reports/'bank_correction_pair_verified_20261101.json')or{}
        names={f'{kind}_kl{kl}_s{seed}'for kind in ('control','corrections')for kl in ('001','01')for seed in (402,403)}
        complete=(proof.get('schema')=='verified-bank-correction-pair-v1'and proof.get('kernel')==pair['kernel']
            and proof.get('actual_cuda_verified')is True and proof.get('supervised_updates_verified')==6400
            and {r['id']for r in proof.get('runs',[])}==names)
        count=sum(min(800,max(0,r.get('updates',0)))for name,r in snapshot.get('corrections',{}).items()
            if name in names and r.get('actual_cuda'))if snapshot.get('kernel')==pair['kernel']else 0
        pct='100% проверено'if complete else f'≥{count/64:.1f}%'if count else '—'
        status={'full_validation_running':'идёт сравнение полных матчей',
            'full_validation_verified':'полные матчи проверены','verifying_cuda_outputs':'проверяем веса'}.get(pair.get('status'),pair.get('status','подготовлен'))
        lines.append(f"| Исправления защиты: с примерами / без | {pct} | {status} |")
        lines+=['','**Исправления защиты:** 51 воспроизведённый пример, 8 прогонов; '
            'сравнение на 648 полных симуляционных матчах с одинаковыми сценариями. '
            '[Разбор решений](reports/bank_correction_diagnosis_20261101.md).']
        if complete:
            from scripts.kaggle.local_artifact_transfer import file_sha
            summary=read(reports/'bank_correction_pair_full_validation_20261101.json')or{}
            if (summary.get('schema')=='verified-correction-full-games-v1'and summary.get('games_verified')==648
                    and summary.get('verified_sha256')==file_sha(reports/'bank_correction_pair_verified_20261101.json')):
                pair_summary=summary
                lines+=['','| Вариант | Победы в симуляторе, два seed | Порог всех семейств |','|---|---:|---:|']
                for row in summary['groups']:
                    kind,kl=row['case'].split('_kl')
                    label=('Контроль'if kind=='control'else'Исправления')+' · KL '+('0,01'if kl=='001'else'0,1')
                    lines.append(f"| {label} | {row['wins']}/144 | {'да'if row['all_family_floor_passed']else'нет'} |")
                lines+=['','144 испытания используют 72 повторяющихся сценария. Это validation, не реальные игры. '
                    '[Полное сравнение](reports/bank_correction_pair_full_validation_20261101.md).']
            else:
                games=0
                for model in range(9):
                    for block in range(6):
                        part=read(Path(pair['root'])/'full_validation'/f'model_{model}_block_{block}.json')
                        if part and len(part.get('records',[]))==12:games+=12
                lines+=['',f'**Проверка кандидатов:** {games}/648 матчей завершено; итоговые винрейты ещё не подведены.']
    previous=read(reports/'bank_previous_behavior_summary_20261101.json')
    behavior_rows=[]
    for name,label in (('margin','balanced'),('continued','Предыдущий BC'),('v25','v25')):
        behavior=read(reports/f'bank_{name}_behavior_summary_20261101.json')or{}
        if behavior.get('schema')=='bank-behavior-local-summary-v1'and behavior.get('episodes_verified')==384:
            best=behavior['ranking'][0]
            behavior_rows.append(f"| {label} · {best['case']} | {best['safe']}/{best['episodes']} | {best['mean_damage']:.0f} |")
    if behavior_rows:
        lines+=['','**Новые серии: локальная защита шара с поддержкой** (это не винрейт):','',
            '| Лучший вариант серии | Без урона | Средний урон башням |','|---|---:|---:|',*behavior_rows,
            '','Два seed, две задержки и повторяющиеся сцены; ни один вариант пока не надёжен во всех условиях.']
    if depth:
        lines+=['','**Проверка обучаемости:** 800/8000 обновлений × смесь/отдельная защита × два seed. '
            'Изолированная защита — диагностический вариант, не кандидат для реальной игры. '
            '[Протокол](reports/bank_fit_depth_design_20261101.md).']
        match=read(reports/'bank_fit_depth_match_state_20261101.json')or{}
        finished=read(reports/'bank_fit_depth_full_validation_20261101.json')or{}
        proof_path=reports/'bank_fit_depth_verified_20261101.json'
        from scripts.kaggle.local_artifact_transfer import file_sha
        if (finished.get('schema')=='verified-fit-depth-full-games-v1'and finished.get('games_verified')==360
                and proof_path.is_file()and finished.get('verified_sha256')==file_sha(proof_path)):
            lines+=['','| Длительность смешанного BC | Победы, два seed |','|---|---:|']
            for row in finished['groups']:lines.append(f"| {row['case']} | {row['wins']}/144 |")
            lines+=['','Это те же 72 validation-сценария на каждом seed. '
                '[Матчи](reports/bank_fit_depth_full_validation_20261101.md) · '
                '[Карты и расход](reports/bank_fit_depth_decisions_20261101.md).']
        elif match.get('status')=='full_validation_running':
            games=0
            for model in range(5):
                for block in range(6):
                    part=read(Path(depth['root'])/'mixed_full_validation'/f'model_{model}_block_{block}.json')
                    if part and len(part.get('records',[]))==12:games+=12
            lines+=['',f'**Сравнение длительности в полных матчах:** {games}/360 завершено.']
        elif match.get('status')=='waiting_for_verified_weights':
            lines+=['','**Следом:** 360 полных матчей для исходника и четырёх смешанных моделей; ожидается проверка весов.']
        elif match.get('status')=='inspection_required':
            lines+=['','**Сравнение длительности:** требуется проверка локального оценщика; итоговых результатов пока нет.']
    if previous and previous.get('episodes_verified')==384:
        row=previous['ranking'][0]
        lines+=['',f"**Прежний BC, локальная проверка:** лучший вариант {row['case']} — {row['safe']}/{row['episodes']} защит без урона. "
            '[Все варианты](reports/bank_previous_behavior_20261101.md).']
    full=read(reports/'bank_full_validation_previous_20261101.json')
    if not pair_summary and full and full.get('schema')=='verified-bank-full-validation-v1' and full.get('episodes_verified')==216:
        lines+=['','**Прежние веса, полные матчи в симуляторе:**','',
            '| Модель | Победы на одинаковых 72 сценариях |','|---|---:|']
        for row in full['models']:
            lines.append(f"| {row['label']} | {row['wins']}/{row['games']} — {100*row['wins']/row['games']:.1f}% |")
        lines+=['','Это validation, не реальные бои. '
            'Прежний BC уменьшил пустые спеллы, но потерял победы; на большой бюджет не отобран. '
            '[Карты и экономика](reports/bank_full_validation_previous_20261101.md).']
    captured=read(reports/'bank_match_states_20261101.json')
    if captured and captured.get('snapshots_and_train_only_verified'):
        search=read(reports/'bank_match_search_state_20261101.json')or{}
        found={}
        for root in search.get('shards',[]):
            for path in Path(root).glob('context_*.json'):
                row=read(path)
                if row and row.get('baseline_reproduced_exactly'):found[row['index']]=row
        if search.get('postprocess_phase')=='complete':
            lines+=['','**Поиск исправлений завершён:** 106 ситуаций, 51 независимо воспроизведённое улучшение; '
                '1111 учебных состояний. [Как отбирали действия](reports/bank_correction_frontier_20261101.md).']
        elif found:
            lines+=['',f"**Поиск исправлений:** {len(found)}/{captured['states']} ситуаций разобраны; "
                f"в {sum(bool(r['successful_actions'])for r in found.values())} найдены предварительные улучшения при обеих задержках. "
                'Это примеры для следующего обучения, не новые модели. '
                '[Первое воспроизведённое исправление](reports/bank_match_action_search_20261101.md).']
        else:
            lines+=['',f"**Следующий шаг:** собраны {captured['states']} воспроизводимых моментов ожидания из "
                f"{captured['games']} тренировочных матчей. Сравниваем WAIT с постановкой защиты; верных меток пока нет."]
    lines+=['','Первые три серии: 4 способа инициализации × 2 seed × 800 обновлений. '
        'Это BC на локальных демонстрациях, не PPO и не игровой винрейт. Проценты — только подтверждённые обновления.','',
        '**Проверка поведения:** замкнутая защита шара с поддержкой при 1350/2025 мс; '
        'затем полные симуляционные матчи. Большие обучения — после проверки кандидатов.','',
        '**Данные:** 1060 состояний, центральная пушка, запас 750 мс для банковских сцен, '
        'Fireball/Zap во всех слотах каждой группы.']
    jump=read(reports/'hog_edge_jump_summary_20261101.json')or{}
    if jump.get('schema')=='hog-edge-jump-comparison-summary-v1'and jump.get('xy_errors_recomputed'):
        checked=jump['check'];errors=checked['mean_error_tiles']
        lines+=['',f"**Прототип прыжка хога:** на {checked['replayed']} дополнительных переходах ошибка пути "
            f"{errors['baseline']:.2f} → {errors['prototype']:.2f} клетки; физика обучений пока не менялась. "
            '[Проверка и ограничения](reports/hog_edge_jump_20261101.md).']
    lines+=['',
        '**Реальные бои выключены.** Наблюдение за Kaggle раз в 20 минут.','',
        '**Отбор по экономике:** запас во время угрозы, потери башен и пустые спеллы. '
        '[Разбор прежних моделей](reports/bank_economy_behavior_20261101.md).','',
        '[План опыта](reports/bank_campaign_20261101.md) · [Предыдущие результаты](reports/bank_initialization_results_20261030.md) · [План](plan_2.md)','']
    return '\n'.join(lines)


def render_policy_adaptation(reports,state):
    stream=read(Path(state['root'])/'live_stream/progress.json')or{}
    if stream.get('kernel')!=state['kernel']:stream=state.get('progress_snapshot',{})
    runs=stream.get('trials',{})if stream.get('kernel')==state['kernel']else{}
    proof=read(reports/'policy_adaptation_verified_20261101.json')or{}
    names={f'pilot_{name}_s{seed}'for name in ('balanced','bank','bc800','bc8000')for seed in (412,413)}
    complete=(proof.get('kernel')==state['kernel']and proof.get('pilot_steps_verified')==819200
        and proof.get('policy_adaptation_audit',{}).get('verified')is True
        and {r['id']for r in proof.get('pilot_models',[])}==names)
    lines=['# Текущий этап','',
        '**PPO после BC: четыре исходника, одинаковые атаки и награды, два новых seed.**','',
        '| Исходник | Seed412 | Seed413 |','|---|---:|---:|']
    for name,label in (('balanced','balanced'),('bank','Прежний BC'),('bc800','BC800'),('bc8000','BC8000')):
        values=[]
        for seed in (412,413):
            row=runs.get(f'pilot_{name}_s{seed}',{})
            values.append('100% проверено'if complete else f"≥{100*min(102400,row['steps'])/102400:.1f}%"if row.get('steps',0)>0 else'—')
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines+=['',f"**Состояние:** {'все PPO-бюджеты и веса проверены'if complete else state.get('status','подготовлено')}.",
        'План: 819 200 шагов среды, по 102 400 на каждый прогон. Это PPO на полных тренировочных играх; новых BC-обновлений нет.','',
        '**Зачем:** проверить, улучшает ли обучение по исходам игр защиту после BC и сохраняет ли экономику. '
        'Большое продолжение выбирается после результатов пилота.']
    prior=read(reports/'bank_fit_depth_full_validation_20261101.json')or{}
    if prior.get('schema')=='verified-fit-depth-full-games-v1'and prior.get('games_verified')==360:
        lines+=['','**Предыдущая проверка в симуляторе:**','',
            '| Обучение по примерам | Победы |','|---|---:|']
        for r in prior['groups']:lines.append(f"| {r['case']} | {r['wins']}/144 |")
        lines+=['','Те же 72 сценария повторяются для двух seed. BC8000 тратит около 23–24 эликсира на пустые спеллы за матч '
            'против 10–11 у BC800. [Полные матчи и расход](reports/bank_fit_depth_decisions_20261101.md).']
    lines+=['','**Реальные бои выключены.** Kaggle проверяется раз в 20 минут.','',
        '[Протокол PPO](reports/policy_adaptation_design_20261101.md) · '
        '[Разбор BC](reports/bank_fit_depth_diagnosis_20261101.md) · [План](plan_2.md)','']
    return '\n'.join(lines)


def render_economy_experiments(reports,calibrated,adaptation):
    lines=['# Текущий этап','',
        '**PPO: исправляем расход эликсира и сравниваем исходные веса.**','',
        '| Опыт | Вариант | Первый seed | Второй seed |','|---|---|---:|---:|']
    studies=[('Штраф пустого спелла',calibrated,(422,423),
              (('control','0,1 — контроль'),('neutral','0,3'),('medium','0,5'),('strong','1,0')),
              'calibrated_spell_verified_20261102.json','calibrated_spell_summary_20261102.json')]
    if adaptation:
        studies.append(('Исходные веса',adaptation,(412,413),
                        (('balanced','balanced'),('bank','Прежний BC'),('bc800','BC800'),('bc8000','BC8000')),
                        'policy_adaptation_verified_20261101.json','policy_adaptation_summary_20261101.json'))
    summaries=[];statuses=[]
    for label,state,seeds,cases,proof_file,summary_file in studies:
        stream=read(Path(state['root'])/'live_stream/progress.json')or{}
        runs=stream.get('trials',{})if stream.get('kernel')==state['kernel']else{}
        proof=read(reports/proof_file)or{}
        complete=(proof.get('kernel')==state['kernel']and proof.get('pilot_steps_verified')==819200
                  and {r['id']for r in proof.get('pilot_models',[])}=={f'pilot_{name}_s{seed}'for name,_ in cases for seed in seeds}
                  and (proof.get('calibrated_spell_cost_verified')or proof.get('policy_adaptation_audit',{}).get('verified')))
        for name,title in cases:
            values=[]
            for seed in seeds:
                steps=runs.get(f'pilot_{name}_s{seed}',{}).get('steps',0)
                values.append('100% проверено'if complete else f'≥{min(100,100*steps/102400):.1f}%'if steps else'нет счётчика')
            lines.append(f"| {label} | {title} | {' | '.join(values)} |")
        statuses.append(f"{label}: {state.get('status','подготовлено')}.")
        summary=read(reports/summary_file)
        if complete and summary and summary.get('kernel')==state['kernel']:summaries.append((label,summary))
    lines+=['',*statuses,'','На каждый опыт — 8 прогонов по 102 400 шагов. Отсутствие счётчика не означает остановку GPU.','',
        '**Найдено:** при полном запасе пустой Fireball получает +0,8 награды относительно ожидания; Zap +0,4. '
        'Проверено в пустой арене. Штраф 0,5 меняет знак; улучшение игры ещё нужно измерить. '
        '[Проверка и план опыта](reports/empty_spell_incentive_20261102.md).']
    for label,summary in summaries:
        lines+=['',f'**{label}: проверенный validation**','',
                '| Вариант | Победы | Пустые спеллы, эликсир / матч |','|---|---:|---:|']
        for row in summary['rows']:
            lines.append(f"| {row['id']} | {row['wins']}/{row['games']} | {row['mean_empty_spell_elixir']:.2f} |")
    lines+=['','Большой бюджет — после сравнения побед по семействам, защиты и полезных спеллов. '
            'Реальные бои выключены; Kaggle проверяется раз в 20 минут.','',
            '[План](plan_2.md) · [Предыдущие 360 матчей](reports/bank_fit_depth_decisions_20261101.md)','']
    return '\n'.join(lines)


def render(reports):
    reports=Path(reports)
    adaptation=read(reports/'policy_adaptation_training_20261101.json')
    calibrated=read(reports/'calibrated_spell_training_20261102.json')
    if calibrated:return render_economy_experiments(reports,calibrated,adaptation)
    if adaptation:return render_policy_adaptation(reports,adaptation)
    bank=read(reports/'bank_campaign_20261101.json')
    if bank:return render_bank_campaign(reports,bank)
    crown=read(reports/'crown_economy_pilot_training_20261030.json')
    if crown:return render_crown(reports,crown)
    reports=Path(reports)
    campaign=read('tmp/kaggle/mana_campaign_20261029/campaign_state.json')
    if campaign:return render_mana(reports,campaign)
    lines=['# Текущий этап','',
        '**Улучшаем PPO после ошибок реальных боёв. Новая модель для live пока не принята.**', '',
        '| Короткое обучение | Прогресс | Результат / состояние |','|---|---:|---|']
    mechanics=read(reports/'recorded_episode_replay_20261028.json')
    offline=read(reports/'balanced_offline_decisions_20261029.json')
    if offline and offline.get('schema')=='saved-live-decision-audit-v1':
        counts=offline['commands']['counts']
        lines[4:4]=['**29 октября: реальные бои остановлены по просьбе пользователя; продолжаем офлайн.** '
            f"balanced: {counts.get('confirmed',0)} подтверждённых команд, исход единственного боя неизвестен. "
            'Добавлена защита нового слота; накопление эликсира остаётся проблемой. '
            '[Разбор и изменения](reports/balanced_offline_diagnosis_20261029.md).','']
    if mechanics and mechanics.get('schema')=='recorded-episode-replay-report-v1':
        note='**Проверка реальных эпизодов:** исправлены скорости снарядов пушки и лучницы. '
        fireball=read(reports/'native_fireball_crown_overlap_verified_20261028.json')
        if fireball and fireball.get('schema')=='recorded-fireball-crown-overlap-v1':
            note+='[Исправлены угловые попадания Fireball по Princess](reports/fireball_crown_overlap_20261028.md). '
        hog=read(reports/'hog_side_clip_replay_verified_20261029.json')
        if hog and hog.get('schema')=='native-hog-side-clip-audit-v1':
            note+='[Исправлен боковой обзор хога](reports/hog_sight_and_river_20261029.md):5 из36 эпизодов после реки лучше, ухудшений нет. Прыжок ещё неверный; полный эпизод расходится до7,27 клетки.'
        else:
            note+='Путь хога расходится до7,31 клетки; исправление прыжка/выбора цели ещё не принято. [Сравнение и график](reports/recorded_mechanics_comparison_20261028.md).'
        lines[4:4]=[note,'']
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
    central=read(reports/'central_tower_pilot_summary_20261027.json')
    if central:
        values=', '.join(f"{r['id']}: {r['wins']}/{r['games']}" for r in central['rows'])
        next_step='Порог продолжения пройден; готовится проверка больших запусков.' if central['eligible_for_full'] else 'Порог продолжения не пройден. Следующий малый опыт меняет долю защитных стартов; статус запуска — в таблице выше.'
        if read(reports/'normal_range_pilot_verified_20261027.json'):
            next_step='Защитные старты также проверены и не прошли порог. Сейчас исправляем физику по реальным эпизодам; нового большого обучения нет.'
        lines+=[f'**Центральная пушка и урон башен проверены:** {values}. {next_step} [Результаты](reports/central_tower_pilot_results_20261027.md).','']
    elif read(reports/'combined_reward_pilot_summary_20261027.json'):
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
    if offline and offline.get('schema')=='saved-live-decision-audit-v1':
        lines.append('| balanced · 29 октября, остановлена | 1; ещё4 отменены | исход неизвестен | — |')
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
    experiment_note='**Новый опыт:** те же защитные старты и награды на исправленной дальности атак. Пакет принят Kaggle; подтверждённые шаги — в таблице выше. Проверены четыре команды обучения, точный resume и48 технических игр. [Детали](reports/normal_range_pilot_20261028.md).'
    if read(reports/'normal_range_pilot_verified_20261027.json'):
        experiment_note='**Последние опыты завершены:** защитные старты с прежней и исправленной дальностью не прошли порог продолжения. Нового большого обучения нет; следующий шаг — исправление пути/прыжка хога и повторный малый опыт на закреплённой физике.'
    lines+=['','Старая серия прервалась из-за связи и отказа входа. У второй старой серии исход6-го неизвестен, ещё4 игры не начаты.','',
        '**Три приоритета:** пушка ограничена центром; бонус фактического урона своих башен включён; дальность обычной атаки и остановка движения исправлены. [Проверка245 выстрелов](reports/normal_range_correction_20261028.md): старая геометрия179, новая245. Это проверка физики, не качество модели. Удержание цели остаётся отдельным экспериментом; полное соответствие игре не доказано.','',
        experiment_note,'',
        '[Все прежние таблицы train/val/test и этапы](reports/training_dashboard_history_20261028.md). Исторический снимок содержит прежние статусы; актуальные — выше.','']
    return '\n'.join(lines)
