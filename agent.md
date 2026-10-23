# Hasty-CR: рабочая карта проекта

## Сейчас: четыре полных PPO после подбора коэффициентов

Все16 опытовseed42 и6seed43 завершены, выгружены и проверены.
Победители среднего двухseed: defense spell_precision/economy_soft;
joint economy_soft/tower_survival. Выбор безtest и безrawreward.
reports/defense_plan/ppo_coefficient_confirmed_selection.json,
ppo_coefficient_confirmed.md иppo_coefficient_confirmation_checkpoints.json.

Отправленыversion1 и фактически обучаются4 kernels
OWNER/hastycr-ppo-coef-full-20261023:
- andriashadrin: defense spell_precision;
- vertel0: defense economy_soft;
- andreiperelaiko: joint economy_soft;
- demjanklimov: joint tower_survival.
ВездеTeslaT4 и ростsteps подтверждены:
reports/defense_plan/ppo_selected_full_launch.json all_verified=True.
Пакет tmp/kaggle/ppo_coefficient_full_20261023/; PID/пути observer/collector
в OWNER/monitoring.json. Не отправлятьповторно; опрос раз1200сек.
При старте третьего восстановлен только локальныйstream после auth/SSE404;
kernel не перезапускался. СтарыеSSE первого/второго этапов остановлены только
после полнойвыгрузки; файлы сохранены.

Каждыйполныйrun:2048000 новых шагов, fixedteacher своего режима,
новыйoptimizer, seed42, sourcearchive40ba4ec0..., сохранение100updates,
trainwallclock18000сек/kernel24000сек. При incomplete_budget сохранить
последниеweights и продолжать толькоexact same-source boundary;
full_command учитывает additionalupdates defense / finalupdate joint.
Полныйбюджет и конечнаяприёмкакачества ещё НЕ достигнуты. v25 неизменна.


Готова новая EVAL-only версия пакета:
tmp/kaggle/ppo_selected_evaluation_20261023_v2/, archive0544f61b...;
reports/defense_plan/ppo_selected_evaluation_readiness_v2.json.
Все146 training-файлов неизменны. Новый sim/evaluation_terminal_damage.py
добавляет opt-in terminal attribution в sim.eval_strategy --terminal-attribution:
legacy metrics сохраняются, v2 записывает источник добивания только по
наблюдаемому HP delta во время исходного KingDestroyed. Для5 парных случаев
проверены побайтовые observations/rewards/masks/legacy metrics/snapshot.
Неизвестные источники остаютсяunknown. Никакоеtraining/reward/optimizer
не изменено. Пакет v1 сохранён, ниv1ниv2 не загружались/не оценивались наGPU.
36тестов observer/attack/evaljob прошли; изолированно загрузились306/309
weights, code_fingerprint иCLIflag проверены. Для новой итоговой оценки
использоватьv2. Будущий job builder должен фиксировать конечный update2000
из selected_full/result.json, а не старый selector истории best_validation:
в новых fulljobs eval-every0, полных результатов пока нет.



## Завершённый подбор коэффициентов PPO (история)


Первый подбор ЗАВЕРШЁН: все16опытов,102400шагов каждый, API COMPLETE,
все4collected/receipt.complete=True. Локальныйаудит весов/optimizer/снимков
восьми сред выполнен для всех16, v25 SHA неизменна. Отбор итаблица:
reports/defense_plan/ppo_coefficient_screen_selection.json и
reports/defense_plan/ppo_coefficient_screening.md.

Текущий этап — seed43,6опытов на3аккаунтах. Jobs version1:
andriashadrin,vertel0,andreiperelaiko/hastycr-ppo-coef-confirm-20261023.
Пакет tmp/kaggle/ppo_coefficient_confirmation_20261023/. Все3фактически
на2T4,ростsteps обоихworkers подтверждён:
reports/defense_plan/ppo_coefficient_confirmation_launch.json all_verified=True.
Не отправлятьповторно. Наблюдатели/collector PID и пути в OWNER/monitoring.json,
опрос1200сек. ВторойSSEпервоначально404, переподключён только локальныйstream,
теперь CUDA/steps подтверждены. Старыеobserver первогоэтапа остановлены только
после terminalCOMPLETE+полнойвыгрузки; удалённыеjobs не менялись.

После трёх полных confirmation collected запустить
scripts.kaggle.prepare_ppo_selected_full: повторно проверит обаseed, выберет
2defense+2joint и подготовит4полных job. Пока полныеjobs не отправлены.
Финалистыseed43: defense economy_soft/spell_precision/economy_strong;
joint economy_soft/tower_survival/survival_balanced. Никакого test-ретюна.


Новый прямой запрос пользователя: сначала102400 шагов на конфигурацию,
затем по2 лучших конфигурации полностью в режимахdefense иjoint (всего4PPO).
Не запускать новыеGRPO; предыдущие модели/замеры сохранить. Выбор только
поvalidation, не поtest и не по reward разных профилей.
Протокол/8профилей/фиксированные128-context validation: data/ppo_tuning/.
В каждом режиме одинаковый weights-only teacher: текущийvalidation-best
PPO1 (SHA0aa0648d...) илиjointPPO3 (SHAd9b0a524...). Optimizer новый.
Новый пакет: tmp/kaggle/ppo_coefficient_study_20261023/. Все4 inputs ready,
все4 kernel hastycr-ppo-coef-screen-20261023 version1 отправлены; на каждом
подтверждены2TeslaT4 и ростsteps у обоих workers. Всего16коротких trials,
по102400шагов. Доказательство: reports/defense_plan/ppo_coefficient_screening_launch.json.
Все4 study jobs API COMPLETE, все16train-бюджетов достигнуты. Идёт выгрузка:
vertel0/demjanklimov collected/receipt.complete=True; оставшиеся два collector
PID/логи вmonitoring.json, скачивание продолжалось. Не перезапускать обучение.
При сетевых сбоях коллектор сохраняет частичные файлы и повторяет загрузку;
cache использует только файлы, совпавшие с receipt SHA. Terminal collectors иSSE наблюдатели запущены,
PID/пути вOWNER/monitoring.json; status раз1200сек. Затем top3/mode seed43,
потом полное обучение2лучших/mode. Прежние eval завершены и сохранены;
200матчей+4020skill проверены: reports/defense_plan/five_model_round_robin_verified.json.

Пользователь явно разрешил регулярныйgitpush вorigin.62 накопленныхкоммита
уже отправлены вmain; source backup snapshot — branch
backup/workspace-source-20261023-101624. Токены остаются внеGit.
Все старые расчёты закончить/сохранить; подготовленный80-match parser-control
v25 job ещё НЕ отправлен и уступает приоритет подборуPPO.

## Обучение завершено, идёт полная оценка

Все4 основныхtraining kernels API COMPLETE и конечные счётчики проверены:
PPO1 2048000,GRPO2 1000650,jointPPO3 2048000,jointGRPO4 1000297.
Подтверждение: reports/defense_plan/four_training_completion.json.
Старые описания RUNNING обучения ниже — история. Ничего не переобучать
и не перепушивать поверх завершённых jobs из-за устаревшего SSE progress.
Конечные веса сохранены в collected/ и collected_final_/collected_retry_;
validation-best для общей оценки уже закреплены, v25 не изменена.

Запущены новыеfinaleval jobs: andriashadrin/hastycr-defense-final-eval-20261023
(PPO1+дополнительныйконтрольGRPO1),vertel0/hastycr-grpo2-final-eval-20261023,
andreiperelaiko/hastycr-joint-ppo-final-eval-20261023 и
demjanklimov/hastycr-joint-grpo-final-eval-20261023. Actual CUDA подтверждены.
Defence-only:708test+15×20duels на модель; joint:1056validation+804test+15×20duels.
Качество не принято, часть результатов ещё ожидается. Проверять раз20мин;
естественные SSE события читать можно. Не повторять активные GPU jobs.

Новый запрос — круговой турнир4основныхмоделей+v25 (200матчей), затем
таблица всех5. Подагент ppo300_live5 готовит отдельныйnewkernel и common
804-case skillbenchmark для всех5 на одномfrozen source. Дополнительный
GRPO1 в эти5 не входит. LiveMuMu по-прежнему0/5 и заблокирован SVIP.

## Последний запрос: PPO300 test и5 реальных игр

Пользователь явно разрешил ровно5 новых реальных матчей с
tmp/kaggle/defense_train_stage2_20261022/recovered_terminal_retry_1/ppo/best_validation_000300.pt,
SHA d4a583bc730b1b0373135c99c409f18ef475d8c3247dd29f639e74441e579cea.
Пока0/5: MuMu stopped и требует оплату истёкшего SVIP. Не покупать подписку,
не очищать данные и не создавать другой профиль вместо существующего.
Readiness и команды продолжения: tmp/live/ppo300_20261023_readiness/.
Полный mac.png содержит частный фон браузера: не публиковать/не добавлять в Git.
memory_series теперь передаёт выбранный checkpoint всем remaining runners;
никакого fallback на v25. После восстановления MuMu проверить свежие ADB/лобби
и запустить только оставшиеся5 с тем же checkpoint. Каноническую v25 не менять.

Отдельно запущена промежуточная GPU-оценка PPO300:
andriashadrin/hastycr-ppo300-test-v25-20261023,708test эпизодов и20парных игр
с v25. Это явный запрос проверить промежуточные веса, не финальная приёмка.
Пакет и результаты: tmp/kaggle/ppo300_test_v25_20261023/.
Подагент ppo300_offline_eval следит за ней; дубликат не запускать.

## Дополнительные общие политики на аккаунтах 3 и 4

Отправлены версии1: andreiperelaiko/hastycr-joint-ppo-20261023 и
demjanklimov/hastycr-joint-grpo-20261023. В каждой ОДНА политика на обе задачи,
309 признаков, task-mode joint, отдельный новый optimizer, инициализация v25.
Пакеты и submission receipts: tmp/kaggle/joint_strategy_20261023/ по владельцу.
CUDA Tesla T4 и рост шагов обоих тренеров подтверждены:
reports/defense_plan/joint_strategy_connection_check.json. Не дублировать задания.
Новые collectors используют --profile strategy --poll-seconds1200.
Все четыре модели проверять раз в20мин; пользователю только проценты.
Первоначальные два задания не изменять, секреты брать только из окружения
изолированных процессов. Это обучение смешанным навыкам, не приёмка стратегии.

## Отдельный GRPO на втором токене

По прямому запросу пользователя запущен vertel0/hastycr-grpo-account2-20261023,
версия1. Аккаунт второго токена — vertel0, индекс1 в приватном реестре
~/.kaggle/access_tokens.json. Рабочий ~/.kaggle/access_token не менять.
Аутентификация второго аккаунта передаётся KAGGLE_API_TOKEN только дочерним
локальным процессам, никогда не включать значение в команды, логи или пакеты.
Пакет/job: tmp/kaggle/grpo_account2_20261023/; исходники source_v2 закреплены,
boundary350/269282. Реальные GPU/ростsteps проверены, live_stream/progress.json
содержит свежие счётчики. Collector запущен --profile grpo-account2 с отдельным
токеном. Первый kernel НЕ остановлен; не запускать дубликаты ни на одном аккаунте.
Оценка этого GRPO требует отдельного подтверждённого результата и validation.

## Текущий запуск восстановления защиты

Исходный stage2 завершился ERROR: GRPO на378/292766 шагах, PPO на1164/1191936.
Сохранения проверены: PPO update1150/1177600, GRPO boundary350/269282.
Продолжение запущено в **andriashadrin/hastycr-defense-semantic-recovery**, версия1,
API подтвердил RUNNING. Не перезапускать и не заменять при RUNNING/QUEUED.
Это фактический slug из ответа Kaggle; желаемый dated slug не был создан.
Требуется добрать PPO2048000 иGRPO1000000, затем свежая validation selection.
Состояние: tmp/kaggle/defense_recovery_20261022/collected.status.json;
collector запущен модулем scripts.kaggle.collect_defense_recovery.
Текущий controller пишет детальные train-логи в output-файлы, не в stdout;
SSE может быть пустым, это не доказательство остановки обучения.
Последние сохранения и receipt: tmp/kaggle/defense_train_stage2_20261022/recovered_terminal_retry_1.
Источник recovery заморожен в source_v2 и отдельном приватном dataset;
исправление Princess/Firecracker escort проверено без изменения архива.
Никакие live-матчи не запускать. v25 и исходный stage2 не заменять.
Исторические сведения о RUNNING stage2 ниже устарели; этот раздел приоритетен.

## Активное большое обучение

Пакет продолжения: `tmp/kaggle/defense_recovery_20261022/source_v2`;
контроллер `scripts/kaggle/run_defense_recovery.py`. Это явная миграция
семантики среды (not_exact_resume), а не обход проверки exact resume.
В Kaggle архивы могут автоматически распаковываться: передавать
`--source-extracted-root` явно на известный каталог source/.
Самодостаточные выбранные веса пишутся в selected/ppo.pt и selected/grpo.pt.
Итоговая оценка: приватный dataset
`andriashadrin/hastycr-defense-eval-repaired-20261022` уже ready;
контроллер evaluate_stage2.py принимает --recovery-result-sha256.
Исходники recovery загружены отдельно как
`andriashadrin/hastycr-defense-recovery-source-20261022`; перед повторной
попыткой проверить текущий процесс загрузки и удалённый статус.
Полные чекпойнты stage2 нельзя получить из пустого metadata listing во
время RUNNING; terminal-only recovery watcher уже запущен.

**Уточнение:** общий kernel RUNNING, но subprocess GRPO завершился с ошибкой
на группе378/292766 шагах: invalid_script_trigger_timeout для
`pair:firecracker:hog_rider:skeletons/back/x1/right`. PPO продолжает обучение.
Не перезапускать/не заменять kernel, пока PPO работает. После завершения
нужен recovery весов из metadata outputs: parent может упасть до этапа
формирования selected_checkpoint/SHA. Подробности в
`reports/defense_plan/stage2_grpo_failure.json`.

`andriashadrin/hastycr-defense-stage2-20261022`, версия 1, RUNNING.
Цели: PPO 2048000 новых шагов, GRPO 1000000 завершённых шагов, весь
train-каталог, warmstart весов из завершённого первого этапа. Это новый
этап с расширенной средой и новым optimizer, не точный resume старых
траекторий. Лимит 21600 с каждому тренеру, kernel timeout24000 с.
Состояние: `tmp/kaggle/defense_train_stage2_20261022/monitor_status.json`;
живой прогресс: `live_stream_reconnect_4/progress.json` рядом. Первый поток
оборвался по SSL, обучение продолжалось; восстановлен только просмотр логов. Источник заморожен в
`dataset/source.tar.gz`, каждый файл проверяется по manifest.json.
**Не запускать повторно и не заменять notebook при RUNNING/QUEUED.**
При смене исходников для exact resume брать замороженный bundle этого
запуска. Рабочая v25 неизменна. Ни один живой матч не возобновлять.
После обучения предусмотрены validation и20 парных симуляционных игр
каждой модели с v25; это ещё не сравнение со всеми15 прошлыми моделями.

## Завершённый первый этап

`andriashadrin/hastycr-defense-stage1-20261022`, версия 2: COMPLETE.
PPO 200 обновлений/102400 шагов, GRPO 100 обновлений/57095 шагов.
Веса скачаны и проверены. Парная validation 24 эпизода —
`reports/defense_plan/stage1_validation_summary.json`; это не итоговая приёмка.
Первый этап повторно не запускать. Активный второй этап описан выше.

## Текущее состояние защитного RL

Источник требований — `plan.md`, выполненные этапы — `done.md`.
Каталог атак экспортируется `scripts/defense/export_catalog.py`; полный
hold-only аудит — `scripts/defense/audit_catalog.py` (это проверка среды,
не обучение). Рабочая v25 сохраняется без изменений.
Приватный защитный CUDA smoke: `andriashadrin/hastycr-defense-smoke-20261022`.
Версия 1: тесты и оба тренера завершились с кодом 0, по 5 итераций;
финальная проверка артефактов упала на отсутствующем sys.path модуля sim.
Исправление загрузчика и обновлённый пакет данных находятся в
`tmp/kaggle/defense_smoke_20261022/`. Не запускать вторую копию без проверки
удалённого статуса. Контрольная сумма v25 проверяется упаковщиком.
Устаревшие состояния более ранних заданий ниже являются историей.

## План и журнал выполнения

Пользовательский план: `plan.md`. Подтверждённый прогресс и расположение
кода/отчётов: `done.md`. После каждого выполненного этапа обновлять `done.md`:
что изменено, где лежит, чем проверено, что осталось. Не отмечать подготовку
как успешное обучение. Секреты в журнал не записывать.
Уточнение терминального условия: смерть любого короля немедленно завершает
траекторию. Обычное доигрывание без новых карт применяется, пока короли живы.

## Приватный тест CUDA на Kaggle

Пользователь разрешил тестовое обучение на 1–5 итераций. Отправлено приватное
задание `andriashadrin/hastycr-cuda-smoke-20261022`: ровно5 PPO rollout/update
итераций (2envs ×8rollout ×5 =80steps), CUDA обязательна, исходная копияv25.
Статус/результаты: `tmp/kaggle/smoke_20261022/monitor_status.json`, код задачи
`tmp/kaggle/smoke_20261022/job/smoke_train.py`, загрузки `output/` рядом.
Первый запуск подтвердил CUDA=true и2TeslaT4, GPU matmul finite=true. Затем упал до обучения: Kaggle распаковал source.tar.gz, старый загрузчик искал архив. Исправлено разрешение распакованной структуры; отправлена версия2 того же kernel. Актуальный монитор/результаты второй версии: `tmp/kaggle/smoke_20261022/v2/monitor_status.json` и `v2/output/`. На момент повторной отправки версия2 QUEUED; пять PPO итераций ещё не подтверждены.
Конечный monitor.py проверяет статус и скачивает логи/чекпойнт при завершении.
CLI: `/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/kaggle`.
Токен хранится вне репозитория; не печатать, не копировать в dataset/notebook.
Не отправлять второе задание поверх этого без проверки статуса. Лимит600сек
выполнения; рабочийv25 не заменять. Бои на Mac остаются остановленными.

## Остановка пользователем

Пользователь остановил текущую серию. Бот и очередь завершены: 25 матчей,
12 побед,13 поражений,48% winrate. Журнал:
`tmp/live/memory_v25_20261022_series30/series.json`, status=stopped_by_user.
**Не возобновлять оставшиеся пять и не запускать новые матчи без нового
явного запроса.** Описания активной серии ниже сохранены как история.

## Активная серия v25 с памятью, 22 октября 2026

По новому запросу пользователя запущены **30 матчей**. Первый — исправленный
матч из `tmp/live/memory_v25_20261022_final3`; два прежних поражения прототипа
в эти 30 не входят. **Не запускать второй процесс или новую серию** без сверки
`tmp/live/memory_v25_20261022_series30/series.json` и процессов `memory_play` /
`memory_series`. Супервизор `scripts/gamestate/memory_series.py` переводит игру
через проверенную кнопку OK в лобби, ждёт `no_battle`, нажимает Battle и ждёт
`MATCH_START`. После первых трёх завершений он запускает runner только на
оставшиеся 27; суммарный лимит остаётся 30. При ошибке меню не повторяет
нажатия вслепую. Итоговые снимки и счётчик побед/поражений в папке серии;
неясный результат отмечается unknown, а не угадывается.

Активный путь: `scripts/gamestate/memory_play.py`, v25, native v8,
`coordinate_mode=physical_live`, без CV в наблюдении. Старый `run-macos.sh`
по-прежнему использует CV и по умолчанию v11: **не использовать для этого
запуска**. Чекпойнт v25 не менялся. Native v8 опционально читает свою руку,
цикл, баланс эликсира и игровые часы (`--own-state`); v6 API не изменён.
Подробности и ограничения в `scripts/gamestate/agent.md`.

В первых двух пробах были ошибки интеграции: остановка при отсутствии
подтверждения карты и переворот Y, отключавший защитные действия. Исправлены
таймаут (другие слоты рассматриваются по свежему состоянию) и явный режим
physical_live. Это не доказывает качество самой стратегии: полная история
карт соперника ещё отсутствует, неизвестные tactical profiles сохраняют
измеренные HP/позицию, но не вымышленные характеристики. Такие случаи
отмечаются INPUT_WARNING. Данные о старом мосте ниже — история разработки.

## Текущее изменение: переход на ARM64-парсер (22 октября 2026)

Пользователь уточнил: сохранить **v25**. Проверенная копия чекпойнта и архив
старого кода/BuildABot лежат вне проекта:
`/Users/andriashadrin/projects/Hasty-CR-preserved-v25-20261022-131231/`.
SHA-256 сохранён в `manifest.json`, копия чекпойнта сверена с оригиналом.
Очистка CV и переключение активного бота ещё не выполнены: описание старого
live-контура ниже пока соответствует коду, а не целевому устройству системы.

Добавлен математический счётчик своего эликсира
`scripts/gamestate/elixir.py` и `CurrentStateReader.query_with_elixir()`.
Поддерживаются подтверждённый доход, пересчёт запоздавших событий,
резервирование команд до подтверждения, отмена при известном отказе,
прогноз баланса и ожидания стоимости. В ответе API есть отдельный доступный
бюджет с учётом резервов; профильные тесты: 39 passed.
Инструкция и ограничения — в `scripts/gamestate/agent.md`: счётчику нужны
игровые часы и подтверждённые расходы, их native API пока не возвращает.
Не подставлять host-time или первые появления юнитов вместо этих данных.
Новые бои для проверки не запускались.

### Согласование парсера и симулятора

Отчёт: `reports/parser_sim_alignment_20261022/report.md`.
`scripts/gamestate/model_bridge.py` даёт вход v25 через общий с обучением
`sim.env.observe`/`legal_mask`, с проверкой сторон, реальных HP и собственных
входных данных; `MemoryPolicy` выполняет inference без нажатий. Старый бот
и launcher ещё не переключены на память. Запасной public-loader исправлен:
он больше не переводит Speed в секунды второй раз; `UnitSpec` хранит raw
Speed, `make_unit` переводит его в mt/s. Скоростной признак старого RL-моста
тоже согласован с Entity.

Обнаружены несогласованные координаты башен и старое отражение Y наблюдения
относительно действий. Для v25 этот контракт не менять молча. Снарядный
аудит старого повтора дал 32 предварительно сравнимых попадания: 25 совпадений
при выборе цели из снимка; гипотеза учёта радиуса атакующего даёт 32/32.
Это косвенные метки, не доказанная общая точность. Основная дальность атаки
и таргетинг не изменены. Клиентского дампа по-прежнему нет, readiness=false;
широкий контрольный прогон содержит 120 старых падений после исправления
скорости против 126 до, новых падений в выбранных пяти файлах не выявлено.
Точные результаты и границы выводов сохранять вместе с отчётом.

Этот файл — короткий маршрут для человека или агента, продолжающего разработку на другом устройстве. Основное описание и установка — в [README.md](README.md). Репозиторий пользователя: `origin` (`andria909/Hasty-CR`); исходный проект: `upstream` (`hastylmao/Hasty-CR`). Пушить только в `origin`, не в `upstream`.

## Что где лежит

| Путь | Назначение |
| --- | --- |
| `sim/` | Детерминированный симулятор, физика, окружение PPO, тренировка и парное сравнение чекпойнтов. |
| `sim/env.py` | Наблюдения, действия, награды и задержка действий в симуляции. |
| `sim/public_state.py` | Публичные признаки оппонента: карты, эликсир, скорость/позиции и угрозы. |
| `sim/meta_decks.py` | Пул колод соперника для `--opponent meta`; включает приближённые live-кейсы. |
| `scripts/brain/` | Политика, RL-мост, ограничения безопасности, оценка заклинаний и трекинг боя. |
| `scripts/cr_bot.py` | Живой цикл: снимок экрана → распознавание → решение → ADB-нажатие → журнал. |
| `run-macos.sh` | Запуск на macOS: наблюдение, ручной бой или автоматическая очередь. |
| `data/live_cases/` | Проверенные описания проигранных матчей и *синтетические* колоды для обучения. |
| `tests/` | Регрессии симуляции и живого RL-моста. |
| `reports/` | Результаты замеров. `tmp/` — локальные журналы/обучение, Git его игнорирует. |

## Текущий игровой контекст

Наша фактическая колода — Hog Rider, Cannon, Musketeer, Fireball, Knight, Electro Spirit, Skeletons, Zap (не классические Ice Golem / Ice Spirit / Log). Цель — надёжная оборона в центре с пушкой и мушкетёром, дешёвый кайт по типу цели, затем темповая атака Hog; не тратить весь эликсир без угрозы. Заклинания должны попадать по прогнозируемому положению цели и давать выгодный размен или добивать башню.

Последний зафиксированный матч перед остановкой бота — `20261013_225020_m009`, поражение 1:3 на `portable_v25_cannon_economy_best.pt`. Его журнал сохранён в `data/live_cases/20261013_225020_m009_source.json`, описание — в `data/live_cases/20261013_225020_m009.json`. В журнале есть действия и редкие снимки HP башен, но **нет** полного наблюдения перед каждым решением. Поэтому это не готовая экспертная PPO-траектория: синтетическая колода лишь добавляет похожее давление в пул `meta`. Восемь карт взяты из наблюдавшихся угроз, но точный состав колоды соперника не подтверждён.

Версии v26–v29 и JSON-отчёты — эксперименты, а не автоматически улучшенный live-релиз. Для v29 есть отрицательный разбор в `reports/portable_v29_livecase_probe.md`: на независимом блоке результат хуже v25. Не заменяйте рабочий чекпойнт только по результату симуляции: сравните на одинаковых seed с `brain,meta`, затем наблюдайте короткие live-тесты. Последняя остановка была по запросу пользователя; самовольно не возобновляйте бои.

## Быстрый старт на другом Mac

```bash
python3.12 -m venv .venv-live
.venv-live/bin/pip install -e '.[live,dev]'
.venv-live/bin/python -m pytest tests -q
```

Для live нужны установленный Clash Royale в Android-эмуляторе, ADB-подключение, локальные `vendor/`-зависимости и чекпойнт. `vendor/`, `tmp/`, локальные конфиги `scripts/brain/*.json`, VPN и учётные данные в Git не входят; сверяйтесь с разделом macOS в README и `vendor/manifest.json`. Проверка устройства: `adb devices`; при нескольких устройствах передать `--serial`.

```bash
./run-macos.sh --serial DEVICE_ID --checkpoint checkpoints/portable_v25/portable_v25_cannon_economy_best.pt --shadow
```

`--shadow` только наблюдает (по умолчанию), `--tap-friendly` нажимает карты, но не кнопку Battle, `--ladder --matches N` сам запускает рейтинговые игры; это риск блокировки аккаунта. Не запускать `--ladder` без явного запроса пользователя. Остановить живой цикл можно Ctrl-C; проверить отсутствие процесса перед новым запуском.

## Обучение и проверка

```bash
.venv-live/bin/python -m sim.train_ppo --help
.venv-live/bin/python -m sim.train_ppo --opponent meta --brain-share 0.5 --steps 200000 --envs 6 --init checkpoints/portable_v25/portable_v25_cannon_economy_best.pt --name experiment
.venv-live/bin/python -m sim.eval_checkpoints --baseline checkpoints/portable_v25/portable_v25_cannon_economy_best.pt --candidate tmp/rl/experiment_best.pt --opponents brain,meta --episodes 80 --output reports/experiment_eval.json
```

Тренер пишет веса в `tmp/rl/`; проверьте созданный файл перед оценкой. Не считайте одиночную победу, reward или win-rate по одному сопернику доказательством прогресса. Проверяйте задержку от наблюдения до нажатия и отдельные случаи: ранний Hog врага, тяжёлый пуш, защитный Fireball, здание для Hog, удержание эликсира, добивание башни.

Перед коммитом: `git status --short`, `git diff --check`, тесты, проверка на токены/ключи. `*.pt`, `checkpoints/` и `tmp/` игнорируются; для передачи выбранного чекпойнта на другое устройство добавляйте только конкретный проверенный файл (`git add -f <точный путь>`), а не весь `tmp/`. Не включайте VPN-ключи, данные аккаунта и cookies.

## Правила этой разработки

На длинных задачах сохранять не менее 50% каждого активного окна квоты Codex: проверять через `$usage` до начала и после этапов, при остатке ≤50% остановиться и спросить пользователя. Ничего не удалять и не чистить без отдельного разрешения на конкретные файлы. Изменения лучше делать обратимыми. Новую модель продвигать в live только после тестов и явного сравнения с базой.

## Проверка live CV, 21 октября 2026

Текущие эталоны восьми карт находятся в `data/card_references/current_20261021`; карты руки и превью читает `scripts/brain/cards.py`. Исправления поля подключены через `scripts/live_perception.py`: фильтр границ текущей арены, исключение почти идентичных гипотез, ограниченное уточнение слабых зданий теми же ONNX-весами и проверка стороны здания по полоске HP. Профильные проверки: 110 passed, 5 skipped; новые бои и обучение не запускались.

Результаты поля и ограничения: `reports/board_cv_fix_20261021_final/report.txt`. Переносимые реальные кадры и предварительная разметка: `data/validation/board_regression_20261021/`. Это диагностические связанные кадры, не независимая оценка всего детектора. Оставшиеся ошибки сохранены в `unresolved.json`: свой Hog как вражеский Knight, эффект смерти как Mini PEKKA, полный пропуск войск под эффектами. Не заявлять идеальное распознавание поля. Повторный офлайн-прогон: `.venv-live/bin/python scripts/audit_board_cv.py --output reports/NEW_BOARD_AUDIT`.

KataCR проверен отдельно офлайн: `scripts/brain/katacr.py`, `scripts/eval_katacr.py`, отчёт `reports/katacr_eval_20261021/report.txt`. Оба опубликованных v0.7.13 скачаны в `tmp/katacr_eval_20261021`; источники/хэши — `data/detectors/katacr_v0713.json`. На MPS: 148 кадров, среднее 122 мс. На 4 размеченных кадрах — 9/9 объектов, но в общем прогоне 59 противоречий своей колоде, включая ложные классы на надписях и ошибки стороны. В live KataCR не подключён. При проверке исправлена разметка кадра 3: враг mega_minion, а не knight; `labels_v2.json` и пересчитанные метрики в отчёте KataCR заменяют прежние проценты на labels v1. Оригиналы сохранены. Проверки адаптера/метрики/поля/покрытия: 36 passed, 1 skipped.


Офлайн-проверка исходной пары KataCR v0.7.13 на авторском датасете (22 октября 2026): `scripts/eval_katacr_original.py`, полный список `images/part2/yolo_annotation.txt` — 6939 кадров, 117026 объектов. Отчёт `reports/katacr_original_20261022_full/report.txt`; юниты: precision 92.82%, recall 75.89%, mAP50 87.62%, mAP50-95 67.95%; сторона верна у 97.86% правильно найденных юнитов. Это родная проверка автора, возможны общие исходники с обучением; не независимая оценка новой игры. Отдельный последовательный замер 64 кадров при работающем MuMu: среднее 254 мс, p95 349 мс. Нового обучения, боёв и продвижения в live не было.

## ARM64-парсер памяти и живой замер, 22 октября 2026

Инструкция: [scripts/gamestate/agent.md](scripts/gamestate/agent.md). Скрипт `scripts/gamestate/query_arm64.py` возвращает текущие ID/имя/сторону/HP/координаты войск, зданий и башен без скриншота. Native-reader `scripts/gamestate/native/arm64_entities.c` читает память только через `pread`; Python проверяет root, ARM64, versionCode160402020 и закреплённый SHA-256. Мир боя находится автоматически при каждом запросе. Не отключать проверки сборки и не смешивать с прежним x86-reader.

Один запрос из корня проекта после сборки/установки native-reader по инструкции:

```bash
.venv-live/bin/python -m scripts.gamestate.query_arm64 --serial 127.0.0.1:26625
```

Живой матч с разрешения пользователя: v25, ровно1 бой, 35 розыгрышей, поражение0:1; бот штатно завершился по `max_matches=1`. Парсер работал параллельно, бот всё ещё использовал BuildABot/CV. Из600 запросов в начале матча для итоговой метрики взяты543 после подтверждённого MATCH_START: median21,71мс, p95=44,10мс, max57,31мс, 543/543≤100мс, отклонений0;6–18 объектов включая башни/снаряды. Это полный запрос с Mac до разобранного состояния, без первоначальной проверки версии/хэша. Отчёт: [reports/arm64_live_query_20261022.json](reports/arm64_live_query_20261022.json); локальные исходные JSONL и лог матча указаны внутри.

Замер ограничен первой минутой одного живого матча. Это текущие позиции тел, не доказанный полный поток постановок карт: первые появления, групповые карты, дочерние юниты и мгновенные заклинания требуют отдельного учёта. Side0/1 не означает фиксированно «мы/враг». Скорость парсера не является скоростью нынешнего CV-бота. Профильные тесты:10 passed (`test_arm64_entities.py`, `test_query_arm64.py`). Не возобновлять бои автоматически после завершённого теста.
