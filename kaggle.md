# Как запускается обучение на Kaggle с ноутбука

Ноутбук собирает исходники и чекпойнты, загружает приватные входные данные и отправляет задачу через Kaggle API. Само обучение выполняется на GPU Kaggle. После завершения ноутбук скачивает результаты и проверяет контрольные суммы. MuMu и живые матчи для этого не нужны.

## Текущая серия 29 октября:24 PPO-прогона

[Параметры и пути](reports/mana_campaign_20261029.md), [текущие проценты](done.md), [конфигурация автоматического отбора](data/ppo_tuning/mana_campaign_runner_20261029.json).

На аккаунтах1 и3 получен явный отказ по недельной GPU-квоте. На аккаунте2 приняты `hastycr-mana-balance-1029` и `hastycr-mana-reserve-1029`; `hastycr-mana-wait-1029` ждёт освобождения одного из двух слотов. `queue_capacity_pilot` повторяет только явный отказ по числу сессий, один раз и на том же handle. После неоднозначного сетевого ответа повторного запуска нет: сначала проверяется существующий запрос.

Новые процессы используют SDK2.2.5 из `tmp/kaggle/sdk_225_20261028` через `PYTHONPATH`; основной CLI/токен не менялись. Для одного сбоя `StartBlobUpload`404 использован `scripts/kaggle/retry_blob_upload.py`: повтор загрузки в тот же приватный dataset через официальный web API Kaggle. Это не новая версия обучения и не смена параметров. Отказы по квоте таким способом не повторяются.

Наблюдатели работают раз в20 минут. `watch_mana_campaign` дожидается всех трёх полностью проверенных серий, сводит одинаковые контроли в одну настройку и готовит полное обучение двух лучших прошедших порог профилей. До отправки оно снова проходит проверки frozen inputs и exact resume. Прогресс локально обновляется командой:

```bash
.venv-live/bin/python -m scripts.defense.update_done
```

Она читает сохранённые результаты, не опрашивает Kaggle и не запускает игру. Повторно запускать существующие notebook нельзя: receipts и PID лежат в `tmp/kaggle/mana_campaign_20261029/`.

## Локальное окружение и доступ

Работаем из корня проекта:

```bash
cd /Users/andriashadrin/projects/Hasty-CR
HASTY_KAGGLE=/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/kaggle
HASTY_KAGGLE_PY=/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/python
```

- `.venv-live/bin/python` — Python проекта: упаковка, локальные проверки, работа с PyTorch.
- `$HASTY_KAGGLE` — отдельное окружение CLI для запросов к Kaggle.
- `$HASTY_KAGGLE_PY` — Python того же окружения, для скриптов с Kaggle SDK.
- Аккаунт: `andriashadrin`.
- Токен уже настроен вне репозитория: `~/.kaggle/access_token`, права `0600`. Его содержимое не печатаем, не добавляем в Git, архивы, dataset или notebook.
- Реестр дополнительных токенов разных аккаунтов: `~/.kaggle/access_tokens.json`, также `0600`. Он не является файлом конфигурации CLI. Для отдельного дочернего процесса нужный токен передаётся через `KAGGLE_API_TOKEN`, без изменения рабочего `access_token` и без токена в аргументах команды.

Проверка доступа без вывода токена:

```bash
"$HASTY_KAGGLE" kernels list --mine --csv
```

## Что отправляется на сервер

1. **Приватный dataset с исходниками:** архив `source.tar.gz`, `manifest.json` с SHA-256 файлов, необходимые данные симулятора.
2. **Приватный dataset с весами:** только нужные чекпойнты и документы их происхождения.
3. **Kernel/script:** управляющий Python-файл и `kernel-metadata.json`. Скрипт проверяет входы, распаковывает проект и запускает тренеры.

Dataset — входные файлы; kernel — выполняемая задача. Изменение кода на ноутбуке не меняет уже запущенный kernel.

Для нового обучения текущие исходники можно собрать так:

```bash
.venv-live/bin/python scripts/kaggle/pack_defense.py \
  --output tmp/kaggle/MY_NEW_RUN/dataset \
  --owner andriashadrin \
  --slug hastycr-defense-next-inputs
```

`MY_NEW_RUN` и slug нужно заменить уникальными именами. Упаковщик использует список разрешённых файлов, проверяет v25 и создаёт новую директорию. Сам по себе этот шаг не запускает обучение.

Для продолжения старого обучения используются его замороженные исходники. Новый рабочий checkout может иметь другую геометрию, encoder или механику; его нельзя незаметно подставлять под старый optimizer и траектории.

## Загрузка приватного dataset

В директории загрузки должен быть `dataset-metadata.json`, например:

```json
{
  "id": "andriashadrin/hastycr-defense-next-inputs",
  "title": "hastycr defense next inputs",
  "licenses": [{"name": "other"}],
  "isPrivate": true
}
```

```bash
"$HASTY_KAGGLE" datasets create -p tmp/kaggle/MY_NEW_RUN/dataset -q
"$HASTY_KAGGLE" datasets status andriashadrin/hastycr-defense-next-inputs
```

Дождаться `ready`. `--public` не используем. Для нового пакета создаём новый slug. Вложенные папки CLI по умолчанию пропускает: исходники передаём архивом, а чекпойнты текущего recovery загружены плоскими файлами; bootstrap восстанавливает их пути по проверенному receipt.

Kaggle может автоматически распаковать `source.tar.gz` в `source/`. Контроллер получает этот известный путь явно; рекурсивно искать произвольную папку проекта не нужно.

## Подготовка kernel и выбор GPU

В папке задания лежит `kernel-metadata.json`:

```json
{
  "id": "andriashadrin/hastycr-defense-next",
  "title": "hastycr defense next",
  "code_file": "bootstrap.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": [
    "andriashadrin/hastycr-defense-next-inputs"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
```

`id`, `dataset_sources` и фактический bootstrap должны соответствовать новому пакету. Заголовок должен превращаться в тот же slug, что и `id`: иначе Kaggle может создать задачу по другому адресу. После отправки сохраняем **фактический URL из ответа**.

Script kernel загружает `code_file`. Произвольный JSON рядом с ним не следует считать доступным на сервере: наш bootstrap встраивает конфигурацию непосредственно в Python-файл.

Команда запуска новой подготовленной задачи:

```bash
"$HASTY_KAGGLE" kernels push \
  -p tmp/kaggle/MY_NEW_RUN/job \
  --accelerator NvidiaTeslaT4 \
  --timeout 24000
```

В наших предыдущих запусках этот accelerator предоставлял две Tesla T4. Контроллер дополнительно проверяет `torch.cuda.is_available()` и минимум две CUDA-карты; CPU fallback запрещён. PPO получает `cuda:0`, GRPO — `cuda:1`, процессы запускаются параллельно.

Перед push проверяем, нет ли уже `RUNNING`/`QUEUED` задачи с этим slug. Работающее обучение не заменяем повторным push.

## Где задаётся размер обучения

Конкретные параметры продолжения находятся в `commands()` файла [run_defense_recovery.py](scripts/kaggle/run_defense_recovery.py):

| Параметр | PPO | GRPO |
|---|---:|---:|
| Общая цель этапа | 2 048 000 шагов | 1 000 000 шагов |
| GPU | `cuda:0` | `cuda:1` |
| Параллельные среды / размер группы | 8 сред | 8 траекторий |
| Rollout PPO | 128 шагов на среду | — |
| Learning rate | 0,00005 | 0,00003 |
| Сохранение | каждые 50 обновлений | каждые 50 групп |
| Validation | каждые 100 обновлений, 32 эпизода | каждые 100 групп, 32 эпизода |

Для PPO одно обновление = `8 × 128 = 1024` шага. При продолжении передаём число **оставшихся** обновлений. Для GRPO длина группы меняется, поэтому завершение определяется фактическими `environment_steps`, а не процентом от числа групп.

## Статус, логи и проценты

Текущая задача продолжения:

```bash
HASTY_KERNEL=andriashadrin/hastycr-defense-semantic-recovery
"$HASTY_KAGGLE" kernels status "$HASTY_KERNEL"
"$HASTY_KAGGLE" kernels logs "$HASTY_KERNEL" --follow
```

Общий `RUNNING` подтверждает работу kernel, но не гарантирует, что оба дочерних тренера живы.

Обновлённый локальный controller для **следующих** запусков выводит отдельно PPO/GRPO: PID, начало и завершение процесса, шаги, loss, validation reward и `training_percent`. Процент считается от полной цели с учётом checkpoint: `completed_steps / target_steps × 100`.

**У текущей версии semantic-recovery старый закреплённый controller:** подробные логи пишутся в output-файлы, поэтому live stdout может быть пустым. Обновление локального файла не добавляет проценты в уже работающую задачу. Для следующей упаковки нужно включить новый `scripts/kaggle/run_defense_recovery.py` как `controller.py` и пересчитать его manifest SHA.

Запись live-лога в новую локальную папку:

```bash
"$HASTY_KAGGLE_PY" scripts/kaggle/stream_training.py \
  --kernel "$HASTY_KERNEL" \
  --output tmp/kaggle/MY_NEW_RUN/live_stream \
  --reconnects 100
```

SSL/404/сетевой сбой наблюдения сам по себе не означает падение обучения. Повторяем запрос статуса той же задачи, не запускаем вторую копию.

## Получение результатов и продолжение после ошибки

Полные output-файлы текущего API доступны после терминального статуса. Для recovery используем:

```bash
"$HASTY_KAGGLE_PY" -m scripts.kaggle.collect_defense_recovery \
  --kernel "$HASTY_KERNEL" \
  --output tmp/kaggle/MY_NEW_RUN/collected \
  --watch-seconds 28800
```

Сборщик ждёт завершения, сохраняет `recovery_result.json`, метрики, логи, выбранные веса и последние точки продолжения. Каталог вывода должен быть новым. Для текущей задачи такой сборщик уже запущен; второй запуск не нужен.

Если kernel упал, сохранённые веса не теряются автоматически. Для исходного stage2 отдельный инструмент [recover_defense_outputs.py](scripts/kaggle/recover_defense_outputs.py) извлекает последние PPO-checkpoints и завершённую GRPO boundary, даже если итоговый отчёт не успел перечислить веса.

- **Warmstart:** загружаются веса, optimizer и среда начинают новый этап.
- **Resume:** восстанавливаются также optimizer, RNG, sampler, счётчики и сохранённые состояния среды.
- **Смена механики среды:** явная проверенная миграция с пометкой `not_exact_resume`; нельзя называть её точным продолжением прежних траекторий.

SHA проверяем локально и по закреплённым manifest/receipt. Установленный SDK не сообщает удалённый SHA для каждого output-файла, поэтому независимую серверную проверку хеша не заявляем.

## Итоговая оценка

Когда достигнуты цели по шагам и закончена свежая validation selection, [prepare_defense_evaluation_job.py](scripts/kaggle/prepare_defense_evaluation_job.py) собирает отдельное приватное задание оценки. До этого он отказывает в подготовке.

Протокол: 236 test-сценариев × 3 seed и по 20 парных матчей с каждой из 15 прошлых моделей. Чекпойнт выбирается на validation, test не используется для подбора весов, наград или длительности окон. Процент успешных train-эпизодов не заменяет этот протокол.

## Текущие файлы и задачи

- Продолжение: [hastycr-defense-semantic-recovery](https://www.kaggle.com/code/andriashadrin/hastycr-defense-semantic-recovery).
- Подготовленный job, конфигурация и receipt отправки: `tmp/kaggle/defense_recovery_20261022/`.
- Источник продолжения: `source_v2/` внутри этой папки; загруженная копия controller и её manifest — `source_dataset/`.
- Проверенные сохранения исходного stage2: `tmp/kaggle/defense_train_stage2_20261022/recovered_terminal_retry_1/`.
- Приватные входы текущего recovery: `andriashadrin/hastycr-defense-recovery-source-20261022` и `andriashadrin/hastycr-defense-stage2-recovered-20261022`.
- Выполненные этапы и результаты: [done.md](done.md). План и критерии качества: [plan.md](plan.md).

Рабочая v25 сохраняется отдельно и не заменяется результатом обучения автоматически. Живые бои остаются остановленными.

## Отдельный GRPO на втором аккаунте

По просьбе пользователя запущен [vertel0/hastycr-grpo-account2-20261023](https://www.kaggle.com/code/vertel0/hastycr-grpo-account2-20261023). Второй токен реестра относится к `vertel0`; первый рабочий токен не менялся.

Это отдельное продолжение с последнего доступного completed boundary: update350,269282 шага. Живое состояние первого kernel перенести нельзя, первый kernel не остановлен. На втором аккаунте подготовлен свой приватный dataset с теми же закреплёнными исходниками и checkpoint. Тренер сохраняет прежний `cuda:1` контракт; сервер подтвердил две Tesla T4 и Torch2.11.0+cu128.

- Runner: [run_grpo_account.py](scripts/kaggle/run_grpo_account.py).
- Пакет, job и receipt: `tmp/kaggle/grpo_account2_20261023/`.
- Live-счётчики: `live_stream/progress.json` внутри этой папки, проценты есть в `events.jsonl`.
- Для collector используется `--profile grpo-account2` и аутентификация вторым токеном только в окружении его дочернего процесса. Сборщик уже запущен.
- Проверка реального обучения, GPU и роста шагов: [grpo_account2_connection_check.json](reports/defense_plan/grpo_account2_connection_check.json).
# Совместные модели на аккаунтах 3 и 4

По дополнительному запросу запускаются две общие политики: одна PPO для
атаки и защиты на `andreiperelaiko`, одна GRPO для обеих задач на
`demjanklimov`. Это отдельные модели от исходной защиты на аккаунтах1/2.
Номера токенов в приватном JSON: индексы2 и3. Значения в пакеты не включать.

Версии1 отправлены в `andreiperelaiko/hastycr-joint-ppo-20261023` и
`demjanklimov/hastycr-joint-grpo-20261023`. Пакеты по владельцам:
`tmp/kaggle/joint_strategy_20261023/`. `launch_config.json` закрепляет SHA,
`submission_receipt.json` хранит фактический URL, `observers.json` — PID
локальных наблюдателей. Не создавать дубликаты при RUNNING/QUEUED.

Runner: `scripts/kaggle/run_strategy_account.py`. Один CUDA:0 subprocess
на аккаунт, `--task-mode joint`,309 признаков, warmstart v25 без замены
оригинала, новый optimizer. Цели PPO2048000 и GRPO1000000 шагов.
Обучение смешанным навыкам не заменяет итоговую проверку стратегии.

Collector запускается с `--profile strategy --poll-seconds1200`, только
читает статус и после завершения сохраняет результаты/веса. Streaming
логи доставляются сервером; отдельные API-проверки делать раз в20мин.
Процент считать по фактическим завершённым шагам и указанной цели,
не по группам GRPO или validation. `trainer_completed` сам по себе не
доказывает достижения цели: проверить training_result.json у PPO,
environment_steps и checkpoint_phase финального GRPO, конечность весов.

После terminal collection каждого совместного задания выполнить аудит
(пример для PPO; для GRPO заменить владельца на demjanklimov):

```sh
.venv-live/bin/python -m scripts.kaggle.audit_strategy_training \
  --collected tmp/kaggle/joint_strategy_20261023/andreiperelaiko/collected \
  --launch-config tmp/kaggle/joint_strategy_20261023/andreiperelaiko/launch_config.json \
  --source-manifest tmp/kaggle/joint_strategy_20261023/andreiperelaiko/dataset/manifest.json \
  --output tmp/kaggle/joint_strategy_20261023/andreiperelaiko/training_audit.json
```

`ready_for_validation` подтверждает готовность весов к оценке, но не
качество и не приёмку. Аудит только читает собранные данные. Файл отчёта
создаётся без перезаписи существующего. SHA receipt проверяет внутреннюю
согласованность выгрузки, независимо подтверждённого remote SHA нет.

После успешного аудита общих моделей `scripts.kaggle.select_strategy_checkpoint`
принимает те же `--collected`, `--launch-config`, `--source-manifest` и новый
`--output`. Он фиксирует уже выбранный тренером validation-best в policy.pt,
подтверждая его score финальным checkpoint и полный train-target отдельным
аудитом. Это не выбор поtest и не приёмка качества. Отсутствующий/неполный
validation-best не заменяется случайным последним файлом.

Оценка общих моделей имеет отдельный источник:
`tmp/kaggle/joint_strategy_evaluation_20261023_v2/`, упаковщик
`scripts/kaggle/pack_strategy_evaluation.py`. Старый defense-only пакет
с306 признаками к ним напрямую неприменим. Протокол:
`data/strategy/evaluation_protocol.json`,1056 validation и804 test
эпизода на модель; SHA передаётся `sim.eval_strategy` вместе с SHA выбранных
весов. В test нельзя выбирать веса или подбирать дедлайны/награды.

Для сравнения со старыми моделями `sim.eval_defense_duel` принимает
`--candidate-mode joint_enemy_presence_v1`:309 входов, отсутствующий учебный
таймер и явная публичная эвристика переключения навыков. Старые модели
сохраняют свои режимы из allowlist. Это оценка данного способа развёртывания,
не доказательство выученного стратегического переключения. По протоколу —
20 симуляционных матчей на каждого из15 прошлых соперников, со сменой сторон.
Quality flags и ограничения реальных механик сохраняются.

Для отдельного защитного GRPO на втором аккаунте сборщик
`scripts.kaggle.prepare_defense_evaluation_job` поддерживает
`--result-mode account2`. Он принимает terminal `account2_result.json`,
собственные1000000 шагов и свежую validation selection, включает только
выбранный `grpo.pt`, а controller использует `--account2-result-sha256`.
Не создавать для него фиктивный PPO или успешный recovery_result.json.
Оценка сохраняет весь исходный test и15 соперников; source dataset должен
быть доступен именно владельцу нового evaluation kernel. Оба исходных
алгоритма в обычном режиме recovery по-прежнему оцениваются отдельно на2GPU.


## Подбор коэффициентов PPO

Текущий пакет: `tmp/kaggle/ppo_coefficient_study_20261023/`. Четыре задания
`OWNER/hastycr-ppo-coef-screen-20261023` уже отправлены, повторно их не пушить.
Наблюдатели используют интервал 1200 секунд; подтверждение CUDA и первых
реальных шагов — `reports/defense_plan/ppo_coefficient_screening_launch.json`.

После terminal COMPLETE и скачивания всех четырёх `OWNER/collected`:

```bash
.venv-live/bin/python -m scripts.kaggle.prepare_ppo_confirmation \
  --study tmp/kaggle/ppo_coefficient_study_20261023 \
  --output tmp/kaggle/ppo_coefficient_confirmation_20261023
```

Команда проверяет результаты и готовит шесть прогонов seed43, ничего не
отправляя. Используются прежние приватные datasets и те же teacher.
Перед отправкой каждого нового kernel проверяются готовность dataset и
отсутствие уже запущенного задания с этим slug. После отправки нужны
подтверждение CUDA, растущие шаги и terminal collector с соответствующим
`launch_config.json`.

После скачивания всех шести повторных прогонов:

```bash
.venv-live/bin/python -m scripts.kaggle.prepare_ppo_selected_full \
  --study tmp/kaggle/ppo_coefficient_study_20261023 \
  --confirmation tmp/kaggle/ppo_coefficient_confirmation_20261023 \
  --output tmp/kaggle/ppo_coefficient_full_20261023
```

Получатся четыре отдельных задания: две защиты и две общие политики.
Каждая заново получает фиксированные начальные веса своего режима и
2048000 train-шагов, а не наследует optimizer короткого опыта. Проверенные
короткие результаты остаются нетронутыми. Файлы уже существующих пакетов
не перезаписываются. Выбор использует validation и среднее двух seed,
без test и без сравнения reward разных коэффициентов.

Полный контроллер сохраняет checkpoint каждые100 обновлений и при
лимите18000 секунд. Если точные2048000 шагов не достигнуты, результат
`incomplete_budget` не считается завершённой моделью. Команда
`full_command` учитывает, что у защитного тренера updates — дополнительные
обновления, а у общего — конечный номер. Для возобновления необходимо
отдельно собрать новый пакет с сохранённой границей, тем же frozen source
и настройками; текущий full runner запускает новый этап с teacher и сам
автоматически продолжение не отправляет.

## Итоговая оценка новых полных PPO

После `COMPLETE` и полной проверенной выгрузки соответствующего полного
запуска используется новый builder, а не selector старой истории best-validation.
Фиксируется заранее заданный `train/update_002000.pt` с ровно2048000шагами.
Пример для первого аккаунта (для остальных заменить владельца во всех ссылках):

```bash
.venv-live/bin/python -m scripts.kaggle.prepare_selected_ppo_evaluation \
  --collected tmp/kaggle/ppo_coefficient_full_20261023/andriashadrin/collected \
  --launch-config tmp/kaggle/ppo_coefficient_full_20261023/andriashadrin/launch_config.json \
  --source-manifest tmp/kaggle/ppo_coefficient_study_20261023/shared/manifest.json \
  --evaluation-package tmp/kaggle/ppo_selected_evaluation_20261023_v2 \
  --output tmp/kaggle/ppo_selected_final_eval_20261023/andriashadrin \
  --kernel andriashadrin/hastycr-selected-ppo-final-eval-20261023 \
  --evaluation-dataset andriashadrin/hastycr-selected-ppo-final-inputs-20261023 \
  --selected-dataset andriashadrin/hastycr-selected-ppo-final-weights-20261023
```

Builder проверяет terminal receipt, веса и источник, создаёт приватные
`evaluation_dataset/`, `selected_dataset/`, `job/` и receipt. Не загружает
данные и не запускает GPU. Перед реальной отправкой отдельно проверяются
новые slug и готовность обоих datasets; действующие jobs не заменяются.

У всех четырёх политик одинаковый heldout-каталог:1056validation и804test
(708защиты+96атаки), плюс300игр против15старых моделей. Для306сети таймер
иtask-признак отбрасываются, используется её native physical_live; для309
сохраняется cells_v2. Это разные входные контракты, явно записанные в отчёте.
Legacy-метрики сохраняются, терминальные добивания дополнительно учитываются
наблюдателемv2; неизвестный источник не записывается как подтверждённыйуспех.

Для итоговых файлов можно использовать terminal collector с
`--profile strategy --poll-seconds 1200`: новый controller сохраняет их под
`strategy_evaluation/`. После скачивания нужна сверка `result.json/parts`
с SHA всех трёх отчётов; один статус COMPLETE не является приёмкой качества.
