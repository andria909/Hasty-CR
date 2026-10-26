"""Render the short dashboard from audited metrics pinned to the four final weights."""
import argparse
import json
from pathlib import Path

ROSTER=(('defense','spell_precision','Защита · spell_precision'),
        ('defense','economy_soft','Защита · economy_soft'),
        ('joint','economy_soft','Атака + защита · economy_soft'),
        ('joint','tower_survival','Атака + защита · tower_survival'))


def percent(value):return '—'if value is None else f'{100*value:.2f}%'.replace('.',',')
def number(value,places=1):return f'{value:.{places}f}'.replace('.',',')


def rate(value):
    if not value['known']:return 'Неизвестно: '+str(value['unknown'])
    return f"{percent(value['rate'])} · {value['success']}/{value['known']}"+(f"; неизвестно {value['unknown']}"if value.get('unknown')else'')


def winrate(report,baseline=None):
    if report is None:return '—'
    rows=[report['opponents'][baseline]]if baseline else list(report['opponents'].values())
    if any(r['observed_simulation']['unknown']for r in rows):return 'Есть неизвестные исходы'
    wins=sum(r['observed_simulation']['wins']for r in rows);games=sum(r['games']for r in rows)
    return f'{percent(wins/games)} · {wins}/{games}'


def load_models(reports):
    completion=json.loads((reports/'ppo_selected_four_full_completion.json').read_text())
    if completion.get('all_four_full_budgets_verified')is not True:raise ValueError('Four completed full budgets required')
    pinned={(r['mode'],r['profile']):r for r in completion['models']};models=[]
    for mode,profile,label in ROSTER:
        boundary=pinned[(mode,profile)]
        if (boundary['steps'],boundary['update'])!=(2048000,2000):raise ValueError('Incomplete model cannot show100%')
        item=dict(mode=mode,profile=profile,label=label,key=mode+'_'+profile,full=None,train=None)
        for kind,schema in (('full','verified-selected-ppo-final-summary-v1'),('train','verified-selected-ppo-train-summary-v1')):
            path=reports/('ppo_'+item['key']+'_'+kind+'_metrics.json')
            if not path.exists():continue
            report=json.loads(path.read_text())
            if (report.get('schema')!=schema or report.get('checkpoint_sha256')!=boundary['checkpoint_sha256']or
                report.get('mode')!=mode or report.get('profile')!=profile):
                raise ValueError('Dashboard metrics do not describe the pinned final weights')
            item[kind]=report
        models.append(item)
    return models


def history_progress_lines(reports):
    summary=reports/'history_pilot_summary_20261026.json'
    if summary.exists():
        r=json.loads(summary.read_text())
        lines=['','**История кадров: обучение завершено на 100%, улучшения нет.**',
            '','| Вариант | Победы на validation |','|---|---:|']
        for row in r['rows']:
            lines.append(f"| {row['id']} | {row['wins']}/{row['games']} ({100*row['wins']/row['games']:.1f}%) |")
        lines+=['','Исходник:42/72 (58,3%). Два seed повторяют72 контекста; это не144 независимых состояния. Большое обучение этих вариантов не запускается.',
            '[Подробности](reports/history_pilot_results_20261026.md).']
        board=reports/'board_potential_pilot_training_20261026.json'
        if board.exists():
            b=json.loads(board.read_text());trials=b.get('progress_snapshot',{}).get('trials',{})
            steps=sum(t.get('steps',0)for t in trials.values())
            lines+=['',f'**Новый эксперимент: награда за размен юнитов.** Отправлен на Kaggle; подтверждённые шаги {steps}/819200 ({100*steps/819200:.1f}%). Четыре коэффициента, два seed. [Состояние](reports/defense_plan/board_potential_pilot_training_20261026.json).']
        bank=reports/'bank_conservation_training_20261027.json'
        if bank.exists():
            state=json.loads(bank.read_text());trials=state.get('progress_snapshot',{}).get('trials',{})
            steps=sum(row.get('steps',0)for row in trials.values())
            lines+=['',f'**Контроль расхода эликсира:** аккаунт3, отправлен; подтверждённые шаги {steps}/819200 ({100*steps/819200:.1f}%). Overflow0/0,15 ×JS0/0,1; четыре профиля, два seed. [Состояние](reports/defense_plan/bank_conservation_training_20261027.json).']
        diagnosis=reports/'air_policy_train_diagnostic_20261027.json'
        if diagnosis.exists():
            lines+=['','**Диагностика воздуха:** на трёх TRAIN-seed новый strong держал мушкетёра в руке2023 шага, но лишь на двух имел ≥4 эликсира. Контроль штрафа overflow и JS-баланса отправлен отдельным заданием на освободившийся аккаунт3. [Измерения](reports/air_policy_train_diagnostic_20261027.md).']
        if (reports/'air_reservation_intervention_20261027.json').exists():
            lines+=['','**Проверка накопления:** ожидание4 эликсира и принудительный выбор мушкетёра не дали побед на трёх TRAIN-seed. Одной частоты карты недостаточно; проверяем размещение и размен. [Контроль](reports/air_reservation_intervention_20261027.md).']
        return lines
    path=reports/'history_pilot_training_20261026.json'
    if not path.exists():return []
    study=json.loads(path.read_text());trials=study.get('progress_snapshot',{}).get('trials',{})
    lines=['','**История кадров: короткое обучение на Kaggle.**1/2 кадра ×обычный/усиленный пул, '+
        'два seed;819200 шагов.','',
        '| Конфигурация | Полученная доля бюджета двух seed |','|---|---:|']
    for frames in (1,2):
        for focus,label in (('uniform','обычный пул'),('air_golem','воздух/голем3×')):
            keys=[f'pilot_history{frames}_{focus}_s{s}'for s in(82,83)]
            value=(f"{100*sum(trials.get(k,{}).get('steps',0)for k in keys)/204800:.1f}%"
                if any(k in trials for k in keys)else'нет данных')
            lines.append(f'| {frames} кадр(а),{label} | {value} |')
    lines+=['','[Состояние истории](reports/defense_plan/history_pilot_training_20261026.json). '+
        '41planes/309scalars; данные из двух публичных кадров. Live не запускался.']
    return lines


def slots_evaluation_lines(reports):
    path=reports/'continuous_slots_pool_launch_20261026.json'
    if not path.exists():return []
    launch=json.loads(path.read_text());proof=reports/'continuous_slots_pool_verified_20261026.json'
    if proof.exists():
        r=json.loads(proof.read_text());lines=['','**Новые модели:600 проверочных матчей завершены.**',
            '','| Модель | Test-победы | Турнир-победы |','|---|---:|---:|']
        labels={'full_balanced_s44':'PPO balanced','full_strong_s44':'PPO strong','parent':'Исходник','v25':'v25'}
        for name in r['tests']:
            t=r['tests'][name];s=r['standings'][name]
            lines.append(f"| {labels.get(name,name)} | {t['counts'].get('win',0)}/{t['games']} | {s['wins']}/{s['games']} |")
        lines+=['','[Подробные результаты новых весов](reports/continuous_slots_pool_20261026.md). Live-приёмки нет.']
        return lines
    progress=launch.get('progress_snapshot',{}).get('evaluation',{}).get('games_lower_bound')
    detail=(f"Подтверждено не менее{progress['completed']}/600 завершённых игр. "if progress else'Итоговых результатов пока нет. ')
    return ['','**Новые модели:600 проверочных матчей отправлены на Kaggle.** '+detail+
        'Два новых PPO, исходник иv25;480 test-игр +120 турнирных.']


def render_reality(models,status,reports):
    if status.get('stage')=='continuous_focus_cohort':
        p=reports/'continuous_focus_training_20261026.json';training=json.loads(p.read_text())
        trials=training.get('progress_snapshot',{}).get('trials',{})
        lines=['# Текущий этап','',
            '**Новый цикл PPO:** воздух/голем3× чаще против обычной выборки; entropy0,01/0,03. '+
            'Восемь коротких проб по102400 шагов. Две большие по4096000 разрешены только после проверки всех архетипов.','',
            'Стартовые веса выбираются между исходником и двумя corrected-slot моделями на72 РАЗНЫХ validation-состояниях. '+
            'Отбор:0,5×среднее +0,5×худший архетип; test не выбирает коэффициенты.','',
            '| Текущая проба | Последний полученный процент |','|---|---:|']
        for name,row in trials.items():
            if name.startswith('pilot_')and row['percent']>=100:continue
            lines.append(f"| {name} | {row['percent']:.1f}% |")
        if not trials:lines+=['','GPU-задание отправлено; ожидаются данные о выборе стартовых весов/первых шагах. Процент не угадывается.']
        verified=reports/'continuous_focus_verified_20261026.json'
        if verified.exists():
            v=json.loads(verified.read_text())
            lines+=['','**Проверено:**819200 шагов коротких проб; '+str(v['full_steps_verified'])+' шагов больших обучений. '+
                ('Гейт полного обучения пройден.'if v['full_gate_met']else'Гейт не пройден: большой бюджет не запускался.')]
        lines+=['','Живые бои остановлены. Ни одна модель автоматически не продвигается. '+
                '[Состояние](reports/defense_plan/continuous_focus_training_20261026.json).','',
                'Предыдущая600-матчевая оценка: balanced41/60 в турнире и14/20 противv25, но81/120 наслучайных пушах '+
                'против96/120 уисходника. [Все карты и результаты](reports/continuous_control_pool_20261026.md).']
        lines+=history_progress_lines(reports)+slots_evaluation_lines(reports)
        return '\n'.join(lines)+'\n'
    if status.get('stage') in ('continuous_full_game_ppo','continuous_full_game_slots_ppo'):
        slot_mode=status.get('stage')=='continuous_full_game_slots_ppo'
        training=json.loads((reports/('continuous_slots_training_20261026.json'if slot_mode else 'continuous_training_20261026.json')).read_text())
        trials=training.get('progress_snapshot',{}).get('trials',{})
        finished=training.get('status')=='training_complete_budgets_verified'
        full_started=any(name.startswith('full_')for name in trials)
        completed_pilots=8 if finished else sum(name.startswith('pilot_')and row['percent']>=100 for name,row in trials.items())
        lines=['# Текущий этап','',
            '**Полные матчи PPO на Kaggle, 2×Tesla T4.** Четыре конфигурации × два seed по102400 шагов; '+
            'затем две лучшие по4096000. Всего9011200 шагов. '+
            ('Бюджеты завершены и проверены; независимая оценка ещё требуется.' if finished else
             'Большие обучения начались.'if full_started else 'Идёт короткий подбор.'),'',
            f'**Короткие пробы завершены: {completed_pilots}/8.**','',
            '| Текущая проба | Последний полученный процент |','|---|---:|']
        for name,row in trials.items():
            if name.startswith('pilot_')and row['percent']>=100:continue
            lines.append(f"| {name} | {row['percent']:.0f}% |")
        control=reports/'continuous_training_verified_20261026.json'
        if slot_mode and control.exists():
            prior=json.loads(control.read_text())
            lines+=['','**Старый контроль с общей блокировкой завершён:** '+
                '; '.join(f"{m['id']}: validation {m['validation']['wins']}/{m['validation']['games']} побед"for m in prior['models'])+
                '. Это другие настройки исполнения; веса не продвинуты. '+
                '48 validation-строк —24 начальных состояния, проигранных дважды; в следующем цикле seed уникальны.']
            if (reports/'continuous_control_pool_launch_20261026.json').exists():
                verified_pool=reports/'continuous_control_pool_verified_20261026.json'
                if verified_pool.exists():
                    pool=json.loads(verified_pool.read_text())
                    lines+=['','**600 матчей проверены. Старые контрольные веса, новый исполнитель по слотам:**','',
                        '| Модель | Случайные пуши | Турнир | Противv25 |','|---|---:|---:|---:|']
                    for key,label in (('full_balanced_s44','Контроль balanced'),('full_strong_s44','Контроль strong'),('parent','Исходник'),('v25','v25')):
                        test=pool['tests'][key];standing=pool['standings'][key]
                        against='—'if key=='v25'else str(pool['matrix'][key]['v25'].get('win',0))+'/20'
                        lines.append(f"| {label} | {test['counts'].get('win',0)}/120 | {standing['wins']}/60 | {against} |")
                    lines+=['','Ни одна модель не продвинута. [Все карты и результаты](reports/continuous_control_pool_20261026.md).']
                else:
                    lines+=['','**Независимая проверка:**600 полных игр двух контрольных моделей, исходника иv25; '+
                        '480 игр со случайными пушами +120 турнирных. Исполнитель у всех исправленный, по слотам. '+
                        '[Запуск](reports/defense_plan/continuous_control_pool_launch_20261026.json).']
        if (reports/'continuous_train_family_diagnostics_20261026.json').exists():
            lines+=['',('**Автопереход включён, ждёт проверенные текущие веса:** 'if(reports/'continuous_focus_handoff_20261026.json').exists()
                else '**Следующий цикл подготовлен, ещё не запущен:** ')+'воздух/голем3× чаще, '+
                'два уровня entropy, отбор по среднему и худшему архетипу на72 validation-играх. '+
                'Причина — провал TRAIN против воздуха/голема при хорошем среднем. '+
                '[Протокол](data/ppo_tuning/continuous_focus/protocol.json), '+
                '[диагностика](reports/defense_plan/continuous_train_family_diagnostics_20261026.json).']
        parallel=reports/'clock_fireball_pilot_training_20261026.json'
        if parallel.exists():
            study=json.loads(parallel.read_text())
            summary=reports/'clock_fireball_pilot_summary_20261026.json'
            if summary.exists():
                small=json.loads(summary.read_text())
                lines+=['','**Короткая серия завершена:819200 шагов проверены.** '+
                    'Ни один вариант не прошёл гейт воздуха/голема; больших продолжений нет.',
                    '','| Настройка | Validation-победы |','|---|---:|']
                labels={'current_control':'База','current_miss':'База + штраф FB',
                    'seconds_preserved_control':'Горизонт ×2','seconds_preserved_miss':'Горизонт ×2 + штраф FB'}
                for row in small['rows']+[dict(id='parent',**small['parent'])]:
                    lines.append(f"| {labels.get(row['id'],'Исходник')} | {row['wins']}/{row['games']} ({100*row['wins']/row['games']:.2f}%) |")
                lines+=['','144 строки =72 validation-контекста ×2 обученных seed, не независимый test. '+
                    'Штраф уменьшил число FB, точность наведения почти не изменилась. '+
                    '[Карты и подробности](reports/clock_fireball_pilot_results_20261026.md).']
            else:
                lines+=['','**Параллельная короткая серия:** '+study['status']+
                    '; gamma/lambda × штраф пустого Fireball,8 проб,819200 шагов. '+
                    'Старт с исходника; автоматических больших запусков нет.']
            trials=study.get('progress_snapshot',{}).get('trials',{})
            if trials and not summary.exists():
                lines+=['','| Конфигурация короткой серии | Доля бюджета двух seed |','|---|---:|']
                for case in ('current_control','current_miss','seconds_preserved_control','seconds_preserved_miss'):
                    steps=sum(trials.get(f'pilot_{case}_s{s}',{}).get('steps',0)for s in(72,73))
                    lines.append(f'| {case} | {100*steps/204800:.1f}% |')
        if (reports/'continuous_control_uncertainty_20261026.json').exists():
            lines+=['','**Разброс:** старые14/20 противv25 дают приблизительно45–95% по seed-bootstrap; '+
                'надёжное превосходство не доказано. [Подробности](reports/continuous_control_uncertainty_20261026.md).']
        if (reports/'continuous_train_reward_audit_20261026.json').exists():
            lines+=['','**Разбор наград:**17 871 TRAIN-return пересчитаны без расхождений; '+
                'в последних4713 играх штраф переполнения всегда0. Следующая отдельная гипотеза — '+
                'gamma/lambda с учётом250мс шага. [Аудит](reports/continuous_reward_audit_20261026.md).']
        if (reports/'full_elixir_opening_probe_20261026_v3.json').exists():
            lines+=['','**Пустое поле:** зависание не повторилось, но подтверждены пустые Fireball. '+
                'Штраф подготовлен, по умолчанию выключен; текущие GPU-пакеты прежние. '+
                '[Проба и следующая серия](reports/full_elixir_opening_probe_20261026.md).']
        lines+=['','**После завершения:** отдельный оценщик600 матчей для новых весов готов; '+
            'GPU-оценка ещё не отправлена. Test и турнир не используются для выбора коэффициентов.']
        if (reports/'public_observation_aliasing_20261026.json').exists():
            lines+=['','**История кадров:** [PPO и validation подключены](reports/observation_history_gap_20261026.md). '+
                'Старые GPU-пакеты не менялись; live выключен.']
        lines+=['','Наблюдение раз в20 минут; запись логов можно продолжать без потери счётчиков.',
            'Проценты — снимок полученных логов, не оценка по времени. '+
            ('[Состояние](reports/defense_plan/continuous_slots_training_20261026.json).'if slot_mode else '[Состояние](reports/defense_plan/continuous_training_20261026.json).'),'',
            '**Что изменено:** непрерывные противники с настоящей рукой/циклом/эликсиром; '+
            'штраф за переполнение и подбор мягкого баланса карт. '+
            ('Команда225мс;2100мс блокируется ТОЛЬКО использованный слот, остальные доступны. '+
             'Решения каждые250мс. Live ждёт только свежий тик и резервирует стоимость отправленных карт. '
             if slot_mode else 'Задержка команды225мс + общая блокировка2100мс (контрольный старый запуск). ')+
            'Исправлены измеренные скорости снарядов; точки кайта приближены.','',
            '**Проверки:**40 целевых тестов прошли; точный Kaggle-пакет завершил4 контрольных матча '+
            '(исходник проиграл4/4). Отдельно3 старых теста tower troops не проходят без клиентских TOML; '+
            'не выдаём весь тестовый набор за зелёный.','',
            '**Приёмки пока нет:** полная физика/таргетинг и перенос в live не подтверждены. '+
            'Живые бои остановлены. Исходник и v25 сохранены.','',
            '[Среднее использование всех карт у20 моделей](reports/card_usage_20261025.md). '+
            'Предыдущий короткий JS-пилот с0,02 не дал заметного выравнивания или новых навыков.']
        lines+=history_progress_lines(reports)+slots_evaluation_lines(reports)
        return '\n'.join(lines)+'\n'
    live=json.loads((reports/'live_card_usage_20261025.json').read_text())
    complete=status.get('card_usage_status')=='complete_verified'
    lines=['# Где мы сейчас','',
        '**Сейчас:** проверяем перенос симуляции в игру. Живой бот и очередь остановлены. '+
        f"Реальная проба исходной joint economy_soft: победы/поражения — {live['wins']}/{live['losses']}; один прерванный матч исключён.",'',
        ('**Статистика 20 моделей:** готова и проверена. [Все карты, средние и сравнение](reports/card_usage_20261025.md).'
         if complete else '**Статистика 20 моделей:** запущены 400 симуляционных матчей на Kaggle (по 20 против v25); итоговая таблица ещё не готова.'),'',
        '**Реальная игра — среднее число подтверждённых розыгрышей за матч:**','',
        '| Hog | Musketeer | Cannon | Knight | Skeletons | Electro Spirit | Fireball | Zap |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
        '| '+' | '.join(number(live['per_card'][c]['mean_per_completed_match'],2)for c in
            ('hog_rider','musketeer','cannon','knight','skeletons','electro_spirit','fireball','zap'))+' |','',
        'Skeletons — один розыгрыш карты; число появившихся тел учитывается отдельно. '+
        '[Реальная статистика](reports/defense_plan/live_card_usage_20261025.json).','',
        '**Найденные расхождения:** live ждёт ACK руки 1,30 с перед следующим действием; при неподтверждении — 4 с. '+
        'Benchmark исполняет команду через 150 мс. Турнир использовал одну восьмикарточную колоду и level11; '+
        'реальные колоды и уровни отличались. [Исполнитель](reports/defense_plan/native_executor_parity_audit_20261025.json), '+
        '[уровни](reports/defense_plan/live_level_confound_audit_20261025.json).','',
        '**Далее:** согласовать входы и задержки, проверить механику ключевых взаимодействий, '+
        'оценивать на разнообразных колодах. Подготовлены 324 новых tactical-упражнения. На Kaggle запущены три пробы по 102400 шагов: контроль, мягкий баланс 8 карт и баланс 6 без заклинаний. Качество ещё не проверено; состояние и последние полученные проценты — в `reports/defense_plan/realism_status.json`.','',
        '## Ранее обученные модели','',
        'Все четыре прошли 2 048 000 шагов. Ниже историческая среда: legacy geometry и Fireball 10 клеток/с. '+
        '«Без урона» — ни одна своя башня не потеряла HP; это отдельные упражнения, а не винрейт.','',
        '| Модель | Train без урона | Val без урона | Test без урона | Против v25 | Все 15 |',
        '|---|---:|---:|---:|---:|---:|']
    for model in models:
        values=[]
        for split in ('train','validation','test'):
            data=model['train']['metrics']if split=='train'and model['train']else model['full'].get(split)if model['full']else None
            values.append(rate(data['defense']['overall']['no_damage'])if data else '—')
        lines.append('| '+model['label']+' | '+' | '.join(values+[winrate(model['full'],'v25'),winrate(model['full'])])+' |')
    lines+=['','Подробные train/val/test-метрики и турниры сохранены в `reports/defense_plan/ppo_*_metrics.json`. '+
        'Исходные веса и v25 сохранены. Ни корреляция рейтингов с live, ни полная приёмка механик пока не доказаны.']
    return '\n'.join(lines)+'\n'


def render(models,refinement=None,mechanics=None):
    ready=sum(m['full']is not None for m in models);trained=sum(m['train']is not None for m in models)
    lines=['# Где мы сейчас и что получилось','',
        f'**Этап:** все 4 модели обучены. Val/test и матчи проверены у {ready}/4; отдельные train-замеры — у {trained}/4.','',
        '| Модель | Обучение | Готовые замеры | Против v25 | Против всех 15 |','|---|---:|---|---:|---:|']
    if refinement:
        evaluated=refinement.get('evaluation_summary')
        state=('обучение завершено; независимая проверка '+refinement['evaluation']['observed_status']
               if refinement.get('evaluation') else refinement['remote_status'])
        if refinement.get('balanced'):state='новая фаза: '+refinement['balanced']['observed_status']
        note=('**Сейчас:** дообучение сохранённой PPO «атака + защита · economy_soft» на Fireball: '
              'добивание башни/здания и попадание по башне с юнитом одним шаром. '
              'Kaggle: '+state+'. Исходные веса сохранены; '+
              ('первый кандидат не принят из-за снижения винрейта. 'if evaluated else 'результаты нового кандидата ещё не проверены. ')+
              '[Статус](reports/defense_plan/fireball_refinement_status.json).')
        lines[2:2]=[note,'']
        selectivity=refinement.get('selectivity')
        if selectivity:
            lines[4:4]=['**Коррекция лишних Fireball:** '+selectivity['observed_status']+
                f". В train-контроле пустой полной башни: исходник {selectivity['parent_empty_tower_casts']}/{selectivity['control_episodes']}, "
                f"первый кандидат {selectivity['candidate_empty_tower_casts']}/{selectivity['control_episodes']} бросков за первые 4 с.",'']
            if selectivity.get('negative_validation'):
                negative=selectivity['negative_validation'];tower=selectivity['validation']['tower_finish']
                lines[6:6]=[f"Validation коррекции: лишний первый Fireball {round(negative['first_action_fireball_rate']*negative['episodes'])}/{negative['episodes']}, "
                    f"добивание башни {tower['successes']}/{tower['episodes']}. Этот кандидат тоже не принят.",'']
        candidate=refinement.get('candidate_summary')
        if candidate or evaluated:
            extra=[('**Первый Fireball-кандидат, независимый test прежней среды (10 клеток/с):**'if evaluated else '**Fireball, validation (отбор кандидата; ещё не независимый test):**'),'',
                   '| Задача | Исходник | После дообучения |','|---|---:|---:|']
            for key,label in (('tower_finish','Добить башню'),('building_finish','Добить здание'),('tower_unit','Башня + юнит одним шаром')):
                before=evaluated['fireball_test']['before'][key]if evaluated else candidate['validation_before'][key]
                after=evaluated['fireball_test']['after'][key]if evaluated else candidate['validation_after'][key]
                extra.append(f"| {label} | {before['successes']}/{before['episodes']} | {after['successes']}/{after['episodes']} |")
            lines[4:4]=extra+['']
            if evaluated:
                matches=evaluated['matches'];v25=matches['by_opponent']['v25'];historical=matches['historical'];parent=matches['parent']
                before=evaluated['old_test_before']['defense']['overall']['no_damage']
                after=evaluated['old_test_after']['defense']['overall']['no_damage']
                lines[4+len(extra)+1:4+len(extra)+1]=[
                    f"**Регрессия первого кандидата:** все 15 соперников: 219/300 → {historical['wins']}/300; "
                    f"v25: 19/20 → {v25['wins']}/20; против исходника: {parent['wins']}/20. "
                    f"Защита без урона: {before['success']}/708 → {after['success']}/708. "
                    '[Проверенный отчёт](reports/defense_plan/fireball_refinement_full_metrics.json).','']
    for m in models:
        state=('Val/test и матчи готовы'if m['full']else'Val/test и матчи ожидают отчёта')+('; train готов'if m['train']else'; train ожидает отчёта')
        lines.append(f"| {m['label']} | 100% | {state} | {winrate(m['full'],'v25')} | {winrate(m['full'])} |")
    lines+=['','100% — бюджет обучения, 2 048 000 шагов. Настройки награды указаны после точки. `—` — проверенного результата ещё нет.','',
        '## Защита от заданного пуша','','«Без урона» — ни одна наша башня не потеряла HP. HP и эликсир — среднее за эпизод; эликсир восстанавливается. Таймаут — обрыв по лимиту времени, незавершённая защита.','',
        '| Модель | Набор | Эпизодов | Таймаутов | Без урона | Потеря HP всех башен | Эликсир | Принцесса на линии атаки жива | Наш король жив |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in models:
        for split,label in (('train','Train'),('validation','Val'),('test','Test')):
            data=m['train']['metrics']if split=='train'and m['train']else(m['full'][split]if split!='train'and m['full']else None)
            if data is None:lines.append(f"| {m['label']} | {label} | Ждём отчёт | — | — | — | — | — | — |");continue
            d=data['defense']['overall']
            lines.append(f"| {m['label']} | {label} | {d['episodes']} | {d['timeouts']} | {rate(d['no_damage'])} | {number(d['mean_tower_hp_loss'])} | {number(d['mean_elixir_spent'],2)} | {percent(d['princess_survived']['rate'])} | {percent(d['king_survived']['rate'])} |")
    lines+=['','Train — учебные сценарии; Val — проверочные; Test — отложенные. По Test веса и коэффициенты не подбираются.','',
        '| Модель · test | Простой пуш без урона | Средний | Сложный | Стресс |','|---|---:|---:|---:|---:|']
    for m in models:
        if m['full']:lines.append('| '+m['label']+' | '+' | '.join(rate(m['full']['test']['defense']['by_difficulty'][d]['no_damage'])for d in ('simple','medium','complex','impossible'))+' |')
    lines+=['','## Атака в отдельных упражнениях','','Хог — фактическое попадание в нужную башню; добивание — завершающий Fireball/Zap. У защитных моделей это контроль: отдельной атаке их не обучали.','',
        '| Модель | Набор | Хог попал | Добил заклинанием | Пропущено окно атаки |','|---|---|---:|---:|---:|']
    for m in models:
        for split,label in (('train','Train'),('validation','Val'),('test','Test')):
            data=m['train']['metrics']if split=='train'and m['train']else(m['full'][split]if split!='train'and m['full']else None)
            if data is None:lines.append(f"| {m['label']} | {label} | Ждём отчёт | — | — |");continue
            a=data['attack'];lines.append(f"| {m['label']} | {label} | {rate(a['hog_confirmed_hit'])} | {rate(a['spell_confirmed_finish'])} | {rate(a['missed_window'])} |")
    lines+=['','[Проверка добивания на двух train-примерах](reports/defense_plan/ppo_train_spell_finisher_probe.json): точный Fireball — 2/2, tower_survival — 0/2.','',
        '## Полноценные матчи в симуляторе','','На соперника — 20 матчей: 10 шафлов с обменом сторон. v25 входит в итог 300 матчей. Цель 80% против каждой прошлой модели пока не достигнута.','',
        '| Соперник | '+' | '.join(m['label']for m in models)+' |','|---|'+'---:|'*len(models)]
    baselines=next((list(m['full']['opponents'])for m in models if m['full']),[])
    for b in baselines:lines.append('| '+b+' | '+' | '.join(winrate(m['full'],b)for m in models)+' |')
    lines+=['','## Подробные проверенные отчёты','']
    for m in models:
        links=[]
        for key,label in (('full','Val/test и матчи'),('train','Train')):
            if m[key]:links.append(f"[{label}](reports/defense_plan/ppo_{m['key']}_{key}_metrics.json)")
        if links:lines.append(m['label']+': '+', '.join(links)+'.')
    lines+=['','Train: 1484 сценария × 3 шафла = 4452 эпизода, веса не меняются. Живые бои выключены; соответствие симулятора реальной игре ещё не принято.']
    if mechanics:
        clock=mechanics['device_clock_tick_rate_hz']['median'];speeds=[]
        for key,label in (('HogRider','хог'),('Knight','рыцарь')):
            row=mechanics['units'][key]
            measured=row['speed_millitiles_per_second_at_20hz']['median']/1000
            expected=row['simulator_unbuffed_speed_millitiles_per_second']/1000
            speeds.append(label+' '+number(measured,3)+' против '+number(expected,1)+' клетки/с')
        lines+=['','**Скорости по сохранённому парсеру:** часы ≈'+number(clock,2)+' тиков/с; на ровных участках '+
            ', '.join(speeds)+' в симуляторе. Таргетинг/бафы/разгон не приняты. '
            '[Замер и ограничения](reports/defense_plan/recorded_native_tick_speeds.json).']
    recovery=(refinement or {}).get('balanced_match_validation')
    if recovery:
        current=['**Текущий этап:** Fireball v3 обучен на 100% (80 эпох). Чекпойнт эпохи 50 прошёл validation-матчи; независимый test ещё не выполнен. Основная модель сохранена.','',
            '| Fireball · validation | Успех |','|---|---:|']
        for key,label in (('tower_finish','Добивание башни'),('building_finish','Добивание здания'),('tower_unit','Башня и юнит одним шаром')):
            row=recovery['validation'][key]
            current.append(f"| {label} | {row['successes']}/{row['episodes']} |")
        current+=['','Лишний первый Fireball: 0/96 контрольных ситуаций.','',
            '| Симуляционные validation-матчи | Победы | Поражения |','|---|---:|---:|']
        for key,label in (('candidate_v25','Новая против v25'),('teacher_v25','Исходная против v25'),('candidate_parent','Новая против исходной')):
            row=recovery['game_summaries'][key]
            current.append(f"| {label} | {row['wins']} | {row['losses']} |")
        current+=['','Все три серии — по 20 матчей, ничьих нет. Среда: Fireball 12 клеток/с и native geometry; прежние таблицы ниже относятся к старой среде. Кандидат не принят окончательно, живые бои выключены. '+
            '[Проверенный отчёт](reports/defense_plan/fireball_balanced_match_validation.json).','',
            '**Далее:** независимые упражнения и турнир против 19 соперников с новой контрольной оценкой исходной модели.','',
            '## Ранее завершённые этапы','']
        if refinement.get('native_evaluation',{}).get('gpu_jobs_submitted')is False:
            current[2:2]=['Независимый тест подготовлен, GPU ещё не запущены: Kaggle не подтвердил загрузки данных; ответы проверки доступа 401/403. '+
                '[Статус запуска](reports/defense_plan/fireball_native_evaluation_launch_status.json).','']
        elif refinement.get('native_evaluation',{}).get('gpu_jobs_submitted')is True:
            current[2:2]=['**Независимый тест запущен:** кандидат и исходник, по 2 Tesla T4; CUDA и веса подтверждены. '+
                'У каждого 804 сценария; матчей — 380 у кандидата и 360 у исходника. Итоговые метрики ещё не проверены. '+
                '[Статус](reports/defense_plan/fireball_native_evaluation_launch_status.json).','']
        lines[2:2]=current
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,default=Path('reports/defense_plan'))
    parser.add_argument('--output',type=Path,default=Path('done.md'))
    args=parser.parse_args()
    status=args.reports/'fireball_refinement_status.json'
    mechanics=args.reports/'recorded_native_tick_speeds.json'
    reality=args.reports/'realism_status.json'
    if reality.exists():
        args.output.write_text(render_reality(load_models(args.reports),json.loads(reality.read_text()),args.reports))
    else:
        args.output.write_text(render(load_models(args.reports),json.loads(status.read_text()) if status.exists() else None,
            json.loads(mechanics.read_text())if mechanics.exists()else None))
