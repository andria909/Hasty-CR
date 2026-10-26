# ARM64: чтение текущего поля Clash Royale

## BlueStacks Air: проверено 25 октября 2026

Бесплатный BlueStacks Air на этом Mac запускает установленный официальный Clash Royale
до меню; выполнен один разрешённый реальный матч. Для ADB shell добавлен локально
собранный `native/adb_su.c`: `/system/xbin/hasty-su`, root:shell, mode 04750.
Он доступен shell/root, не Android-приложениям. Парсер сам root не устанавливает.
Оригинальный образ сохранён в
`/Users/andriashadrin/Downloads/BlueStacks-repair-20261024/parser-root-20261025/Root.qcow2.original`.
Изменённый системный образ может быть заменён обновлением BlueStacks; данные игры не очищались.

Запуск из корня репозитория (serial перепроверять через `adb devices`):

```bash
.venv-live/bin/python -m scripts.gamestate.query_arm64_v2 \
  --serial emulator-5554 --su /system/xbin/hasty-su --own-state \
  --binary /data/local/tmp/hasty_arm64_entities_20261025_bs_v8
```

Python API: `CurrentStateReader(adb, serial, binary=path, own_state=True,
su="/system/xbin/hasty-su")`. Без `su` сохраняется прежний прямой root-ADB путь.
Формат результата и проверки UID/ABI/versionCode/libg SHA не изменены.
В меню возвращается `no_battle`, не фиктивное пустое поле со статусом ok.

Отчёт: `reports/bluestacks_parser_live_20261025.json`; сырые 3000 запросов и
снимки экрана: `tmp/bluestacks_parser_20261025/live_match_01/`.
В матче 1414 снимков поля: median 36.60 ms, p95 45.00 ms, max 67.46 ms.
Это запрос → готовый JSON, без первоначального attach; не задержка постановки карты.
Период опроса был 100 ms. Median непосредственно native read — 0.051 ms.
Сопоставлены HP башен, сторона, начальная рука и полный эликсир со скриншотами.
Обнаружены тела обеих сторон, здания, башни, снаряды и заклинания.
Один матч не доказывает полноту для всех карт или событий постановки.

Own-state валиден в 1206 снимках. В 208 возвращён `transition_or_ambiguous`:
189 при замершем финальном тике 2440 и 19 во время боя (тики 1737–1782,
две группы с валидными снимками между ними). Не заполнять эти пропуски старой
рукой/эликсиром и не выдавать их за достоверное наблюдение. Публичное поле
в этих строках оставалось доступно. Причина 19 промежуточных отказов не установлена.
32 теста транспорта, старого API и CLI прошли. Новые бои автоматически не запускаются.


## Текущий live-путь v8 и серия 30

`CurrentStateReader(adb, serial, own_state=True)` выбирает отдельный
`/data/local/tmp/hasty_arm64_entities_20261022_v8`, флаг `--auto-state-once`.
Обычный `query()` без own_state сохраняет v6. Проверки ABI, versionCode и
libg hash не отключены. Native только читает через pread.

Дополнительные поля: `frame.battle_tick`, `own_state.status/world/tick/own_side`,
`hand_indices`, `cycle_indices`, `elixir_raw`. Частота часов 20 Hz,
эликсир = raw / 10000. Векторы руки и очереди обязаны вместе быть перестановкой
0..7; проверяются повторно и с backlinks к миру/коллекции. На переходах
возвращается unavailable/transition_or_ambiguous. Два валидных private-hand
(неоднозначный повтор) не допускаются. Чужой скрытый эликсир не экспортируется.

Смещения закреплённой сборки: world+0x60 tick; world+0xA8 player_state;
player_state+0x08 backlink; +0xE0/+0xE8 player/king pointers; player+0x78 side;
+0x10 collection backlink; +0x210/218/21C hand vector/capacity/size;
+0x220/228/22C cycle vector/capacity/size; +0x2F8 own mana raw.
Список player+0x288 — порядок замеченных карт, **не порядок колоды**.
На аккаунте визуально проверен слот 1: Hog, Cannon, Zap, Musketeer,
Electro Spirit, Fireball, Skeletons, Knight. Индексы руки относятся к этому
порядку. При смене колоды прежняя калибровка недействительна.

`memory_play` выполняет решения v25 и два Android tap на карту, но сам не
открывает очередь. Для текущей серии меню управляет `memory_series`, общий
лимит 30, журнал `tmp/live/memory_v25_20261022_series30/series.json`.
Нельзя параллельно запускать другой бот. Android 1440x2560; Battle 720,1987;
результат OK 950,2320. Переключение меню только после MATCH_END, снимка и
подтверждения перехода в no_battle. На финальном результате очередь завершается.

Live-мост явно использует physical_live: компенсирует legacy 31-y в общем
observe/mask, чтобы физические позиции соответствовали экранным действиям.
Стандартный legacy_training сохранён для совместимости старых offline-тестов.
Это отдельный явно выбранный контракт, веса модели не переобучались.

Замеры PLAY.timing: полный запрос памяти, native read, подготовка признаков,
inference, выбор карты ADB, целевое нажатие ADB, полный запрос→возврат input.
PLAY_ACK — смена руки, не измерение точного появления тела. Скриншоты только
для диагностики: `--screenshot-every 5`, после input и ACK; не вход модели.
`screenshots.jsonl` хранит время получения PNG и границы игровых тиков вокруг
capture. Это не точный момент экспозиции/рендера. Агрегатор:

```bash
.venv-live/bin/python -m scripts.gamestate.summarize_memory_run RUN_DIR --output reports/NEW_MEMORY_TIMINGS.json
```

Ограничения live-пробы: неполная история соперника явно разрешена флагом
адаптера, неизвестные профили сохраняют тело/HP, а отсутствующие traits
нулевые и помечены warning. Нельзя выдавать такую пробу за полностью
согласованный с обучением вход. Строгие defaults адаптера остаются строгими.
Утверждения ниже о недоступности часов/руки относятся к старому v6 API.

Этот путь предназначен для установленного официального ARM64-клиента в MuMu на Mac. Старые `memread.py`, `frame.py` и `native/hcr_reader.c` относятся к другому, x86_64-клиенту: их ограничения нельзя отключать ради ARM64.

## Файлы

- `query_arm64.py`: Python API `CurrentStateReader.query()`, одиночный запрос и конечный замер задержки.
- `query_arm64_v2.py`: hardened CLI wrapper for SDK/PATH ADB discovery and readable errors when the game process is absent. It uses the same pinned v6/v8 native reader and keeps all root, ABI, version and library-hash checks.
- `native/arm64_entities.c`: Android-процесс читает `/proc/PID/mem` через `open(O_RDONLY)` и `pread`, возвращает JSON. Не внедряет код, не меняет память игры и не нажимает кнопки.
- `arm64_entities.py`: проверки версии/хэша/ABI, диагностический Python-reader и учёт первых появлений.
- `capture_arm64.py`: непрерывная запись native-потока, событий появления и скриншотов. Требует адрес коллекции текущего боя; для обычного запроса используйте `query_arm64.py`, который находит её автоматически.

## Предварительные условия

ADB уже работает от root, игра открыта, ABI `arm64-v8a`. Текущие ограничения:

- `versionCode=160402020`;
- SHA-256 `libg.so`: `347afc57c8d168bb67d0defa488a89f76297025cac1e8544b5182bb7e70ff433`;
- проверенный serial на этой машине: `127.0.0.1:26625`, но после перезапуска проверить `adb devices` и MuMu CLI;
- Python: `.venv-live/bin/python`; локальный ADB: `/tmp/Hasty-android-tools/platform-tools/adb`.

При несовпадении версии или хэша парсер останавливается. Не подставлять старые смещения под новую сборку. Скрипт сам не включает root, VPN и не запускает матч. Запуск боя требует разрешения пользователя.

## Сборка и установка считывателя

Из корня репозитория, используя существующий Zig-компилятор (на этом Mac установлен в `tmp/android_log_probe/compiler/ziglang/zig`):

```bash
tmp/android_log_probe/compiler/ziglang/zig cc -target aarch64-linux-musl -static -O2 -Wall -Wextra -Werror scripts/gamestate/native/arm64_entities.c -o tmp/android_log_probe/arm64_entities_v6
/tmp/Hasty-android-tools/platform-tools/adb -s 127.0.0.1:26625 push tmp/android_log_probe/arm64_entities_v6 /data/local/tmp/hasty_arm64_entities_20261022_v6
/tmp/Hasty-android-tools/platform-tools/adb -s 127.0.0.1:26625 shell chmod 700 /data/local/tmp/hasty_arm64_entities_20261022_v6
```

На другой машине нужен доступный Zig с поддержкой `aarch64-linux-musl`; путь к нему можно заменить. Бинарники и компилятор в `tmp/` не входят в Git. Не перезаписывать работающий бинарник: для нового варианта использовать новый путь и `--binary`.

## Запуск

Один запрос текущего поля:

```bash
.venv-live/bin/python -m scripts.gamestate.query_arm64 --serial 127.0.0.1:26625
```

Для Android Studio AVD можно использовать v2 и явно указать serial/ADB:

```bash
.venv-live/bin/python -m scripts.gamestate.query_arm64_v2 \
  --serial emulator-5556
```

Google Play AVD не поддерживает `adb root`; AOSP/userdebug нужен для чтения памяти.
На проверенном API 35 AOSP и Google Play AVD установленный Clash Royale падает
при старте (`lskumcyej.e: 02`), поэтому новый live-запрос там пока невозможен.
v2 не обходит root или проверки клиента и не запускает игру.

600 запросов с периодом 100 мс, ожидание начала боя до 120 секунд; каталог вывода должен быть новым:

```bash
.venv-live/bin/python -m scripts.gamestate.query_arm64 --serial 127.0.0.1:26625 --requests 600 --period-ms 100 --wait-for-battle 120 --output tmp/MY_NEW_CAPTURE
```

Скрипт не начинает бой. В ожидании он ищет читаемый мир; результат боя или повтор тоже могут содержать мир. Чтобы проверить именно живой бой, сначала убедиться в лобби, затем начать разрешённый матч и подтвердить его по экрану/журналу. Скрипт не классифицирует live/replay автоматически.

Python API из корня проекта:

```python
from scripts.gamestate.query_arm64 import CurrentStateReader

reader = CurrentStateReader('/tmp/Hasty-android-tools/platform-tools/adb',
                            '127.0.0.1:26625')
state = reader.query()
if state['status'] == 'ok':
    for unit in state['units']:
        print(unit['entity_id'], unit['name'], unit['side'],
              unit['x_tiles'], unit['y_tiles'], unit['hp'])
```

Создание reader выполняет проверку root, ABI, версии, PID, отображений и хэша библиотеки один раз. Каждый `query()` заново находит мир и коллекцию внутри native-процесса. После перезапуска игры создать reader заново: PID и адрес библиотеки действуют только внутри текущего процесса.

## Данные и координаты

`status`: `ok`, `no_battle` или `rejected`. Ошибки транспорта/неожиданный формат вызывают исключение; старое состояние не подставляется. `rejected` означает изменение состава/идентичности объектов во время чтения или нарушение проверок. Повторный запрос решает вызывающий код; скрытых повторов в замере нет.

`units` содержит объекты пространства ID 5: войска, здания и башни. `frame.entities` дополнительно содержит видимые этому считывателю снаряды. Поля:

- `entity_id`: ID тела в пределах мира боя; вместе с миром/PID образует идентичность. Может повторяться в другом матче.
- `name`: внутреннее имя, например `HogRider`, `ElectroSpirit`.
- `side`: индекс игрока движка 0/1. Это **не** постоянное обозначение «мы/враг»: в проверенном живом матче мы были side=1. Сторону и ориентацию нужно привязать к текущему экрану.
- `x`, `y`: текущие мировые координаты, 1000 единиц = одна клетка. `x_tiles`, `y_tiles` делятся на 1000. Это не пиксели экрана и не автоматически нормализованный верх/низ относительно пользователя.
- `hp`, `max_hp`: здоровье; native `-1` означает отсутствие прочитанного компонента здоровья.
- `card_id`: карта-источник; у варвара из тарана может быть ID тарана.
- `level_index`: индекс уровня только у тел, `null` у снарядов.
- `deployment_tick_candidate`, `play_sequence_candidate`, `deck_index_candidate`: исследуемые поля. Не считать документированным API команды постановки.

Первое наблюдение нового ID — появление тела, а не гарантированная точка/момент розыгрыша карты. Групповая карта создаёт несколько тел, таран позже создаёт варваров; мгновенные заклинания могут отсутствовать в этой коллекции. `target_position_candidate` у Fireball/Arrows — исследуемые поля цели снаряда. Нельзя объявлять полный охват всех карт/заклинаний.

## Как находится и проверяется память

Для закреплённой сборки: `libg_base + 0x1A586A8` → root; `root+0x28` → context; `context+0x90` → world; `world+0x40` → collection. Tagged pointers маскируются `0x00FFFFFFFFFFFFFF`. Проверяется vtable коллекции `libg_base+0x1936450`.

У коллекции: массив указателей `+8`, capacity `+0x10`, count `+0x14`. Проверяются размеры, уникальность указателей, принадлежность объектов коллекции, формат имён, диапазоны координат/стороны. После чтения повторно проверяются заголовок, массив и идентичность объектов. Это снижает риск смешанного кадра, но **не делает снимок атомарным относительно тика движка**.

## Что измеряется

`round_trip_ms`: монотонные часы Mac, от запуска запроса до получения ответа ADB, разбора JSON, проверки ответа и выделения/перевода координат юнитов. Включает запуск нового subprocess/ADB/native-процесса и поиск мира. Первичная проверка версии/хэша, ожидание боя, интервал между запросами, запись JSONL и вывод в терминал исключены.

`frame.read_us`: время только чтения snapshot внутри Android, без транспорта и большей части подготовки. Его нельзя выдавать за задержку полного запроса.

Это не метрика «нажатие карты → подтверждение её постановки» и не «скриншот → CV-координаты». В первом живом ручном тесте нажатие → первое обнаружение тела заняло 1,25–1,29 с; источник задержки не разложен на сетевое принятие, обработку и появление. Обнаруженная очередь команд в повторе заранее содержит будущие действия; она не доказывает раннее получение команд в live.

`requests.jsonl` сохраняет каждый ответ, `identity.json` — процесс и библиотеку, `summary.json` — accepted/rejected/no_battle, median/p95/max и число запросов ≤100/150 мс. Период 100 мс — частота запросов, а не время выполнения. Результат боя может сохранять читаемое поле: проверять временной диапазон по журналу матча, не смешивать его с активной игрой.

Профильные тесты:

```bash
.venv-live/bin/python -m pytest tests/test_arm64_entities.py tests/test_query_arm64.py -q
```

## Подтверждённый живой замер

22 октября2026: один разрешённый живой матч с параллельно работающим ботом v25,35 розыгрышей, завершение по лимиту1. Отчёт `reports/arm64_live_query_20261022.json`:543 запроса после подтверждённого начала матча, все корректны; median21,71мс, p95=44,10мс, max57,31мс, все≤100мс. Native-read median0,023мс. На поле6–18 объектов включая башни/снаряды. Первые57 запросов до подтверждённого MATCH_START исключены из основной метрики; все600 сохранены в raw JSONL. Это первая минута одного боя, а не стресс-тест и не доказательство полноты всех карт. Бот продолжал использовать прежний CV-путь: переключение его на память здесь не выполнено.

## Математический счётчик своего эликсира

`scripts/gamestate/elixir.py` не использует CV. `ElixirTracker.from_match_start()`
начинает с 5 только при известном начале боя. Стандартная модель: одна единица
за 2.8 игровых секунды, x2 с 120-й секунды, x3 с 240-й, максимум 10;
переходы интегрируются по частям, накопление поверх лимита теряется. Это модель
обычного режима, согласованная с симулятором, а не измерение памяти. Для других
режимов задаётся явный `schedule`; часы должны учитывать паузы.

```python
from scripts.gamestate.elixir import ElixirTracker

elixir = ElixirTracker.from_match_start()  # только известное начало
# Один подтверждённый розыгрыш, не один объект/юнит и не просто отправленный tap:
elixir.confirm_play(event_id='match-17/play-1', cost=4, game_seconds=3.0)
row = reader.query_with_elixir(elixir, game_seconds=4.0)
# row['own_elixir']: value, minimum, maximum, exact, game_seconds, source
```

При подключении посреди боя: `ElixirTracker(game_seconds=elapsed)` возвращает
диапазон [0,10], `value=None`, пока значение не определено. `can_afford(cost)`
использует нижнюю границу. Розыгрыши дедуплицируются по ID события;
несогласованный расход отклоняется, часы не могут идти назад. `invalidate()`
сбрасывает уверенность после пропущенных событий/переподключения; `reconcile()`
позволяет привязать счётчик к доверенному измерению. При каждом новом матче
нужен новый счётчик. Доход от сборщика/эликсирного голема и иные специальные
эффекты учитываются через `confirm_gain()` по подтверждённому событию;
если событие пропущено или величина неизвестна, нужен сброс уверенности.

**Пока native API не отдаёт игровые часы и полный поток своих розыгрышей.**
Их передаёт вызывающий код; автоматического точного live-эликсира из одних
позиций сейчас нет. `query_with_elixir` добавляет оценку только для `status=ok`,
не выдаёт её за поле native-памяти и не меняет обычный `query()`.

### Команды в полёте, доход и прогноз

Перед отправкой ADB-команды резервируйте её стоимость. Резерв не считается
подтверждённой тратой, но вычитается из доступного бюджета: `can_afford()`
и `seconds_until_affordable()` учитывают его. Если отправка завершилась ошибкой,
отменять резерв можно лишь когда известно, что команда не была принята игрой.
Таймаут транспорта сам по себе этого не доказывает. Автоматического истечения
резервов нет, чтобы задержка подтверждения не разрешила повторную трату.

```python
elixir.reserve_play(action_id='tap-1', cost=4, game_seconds=3.0)
# Здесь вызывающий код отправляет команду. Затем получает подтверждение игры:
elixir.confirm_play(event_id='play-1', action_id='tap-1',
                    cost=4, game_seconds=3.4)
# При подтверждённом отказе вместо confirm_play:
# elixir.cancel_play('tap-1')

elixir.confirm_gain(event_id='collector-credit-1', amount=1, game_seconds=8.0)
wait = elixir.seconds_until_affordable(4)
future = elixir.forecast(game_seconds=10.0)  # не изменяет текущее состояние
```

Расходы и доход допускают запоздавшие подтверждения: история пересчитывается
по **игровому времени события**, включая восстановление, потерянное у лимита.
При одинаковом времени событий используется порядок передачи вызывающим
кодом; расход и доход у лимита требуют правильного порядка. Повтор события
не изменяет баланс; конфликтующий ID, невозможный расход или неверное
подтверждение команды отклоняются без частичной записи. ID отменённой или
подтверждённой команды повторно использовать нельзя. `reconcile()` и
`invalidate()` задают новую временную опору: неизвестные события до неё
отклоняются, известные дубли сохраняют дедупликацию; резервы остаются до
разрешения соответствующих команд.

`own_elixir` дополнительно содержит `reserved`, `pending_actions`,
`available_minimum`, `available_maximum`. `value` описывает модельный баланс,
а доступный для новой команды бюджет — `available_minimum`. `exact` означает
точное значение **внутри модели при полноте входных событий**, а не доказанную
точность относительно игры. Пользовательский `schedule` получает отдельный
`source='mathematical_custom_mode'`.

Прогноз предполагает отсутствие новых расходов и доходов. Если стоимость
вместе с резервами превышает 10, ожидание равно `math.inf`: сначала нужно
разрешить ожидающие команды. Прогноз не является чтением будущего состояния.

Проверка счётчика и его подключения к парсеру:

```bash
.venv-live/bin/python -m pytest tests/test_own_elixir.py tests/test_query_arm64.py tests/test_arm64_entities.py -q
```

## Мост в v25 и аудит симулятора

Результаты и ограничения: `reports/parser_sim_alignment_20261022/report.md`.
`scripts/gamestate/model_bridge.py` использует общий с обучением encoder/mask:
`NativeModelAdapter(own_side=...)`, `OwnState`, `MemoryPolicy`. Не вызывает CV,
не нажимает экран и не меняет старый launcher. `placement_world()` возвращает
мировые милли-клетки, **не пиксели**. Существующий `grid_to_point` сохраняется;
расхождение Y наблюдения и действий задокументировано, а не молча исправлено.

```python
from pathlib import Path
from scripts.gamestate.model_bridge import NativeModelAdapter, OwnState, MemoryPolicy

# clock/hand/next/events/active должны быть подтверждены вызывающим кодом.
context = OwnState(game_seconds=clock, hand=tuple(hand), next_card=next_card,
                   elixir=elixir.estimate(), opponent_events=tuple(events),
                   public_history_complete=history_complete, battle_active=active)
view = NativeModelAdapter(own_side=my_side, stats_level=11).encode(reader.query(), context)
policy = MemoryPolicy(Path('checkpoints/portable_v25/portable_v25_cannon_economy_best.pt'))
decision = policy.decide(view)  # только inference; policy переиспользовать между кадрами
```

Неизвестные HP/имена/слоты и отсутствующие башни отклоняются; уничтоженные башни
задаются отдельно подтверждёнными `(engine_side, lane)` в `destroyed_towers`.
При неизвестном своём эликсире, неполной публичной истории или неподтверждённой
активности боя доступно только ожидание. Имена эволюций без профиля не заменять
базовой картой без отдельной проверки характеристик. Уровень статов задаётся
явно: `level_index` ещё не стал проверенным универсальным номером уровня.

Аудит существующих записей (без ADB/боёв):

```bash
.venv-live/bin/python -m scripts.gamestate.audit_sim_alignment --trace tmp/android_log_probe/live_query_v6_20261022_run2/requests.jsonl --start-row 57 --output reports/NEW_ALIGNMENT_AUDIT
.venv-live/bin/python -m scripts.gamestate.audit_projectile_targets --trace tmp/android_log_probe/native_replay_01/frames.jsonl --output reports/NEW_PROJECTILE_AUDIT.json
```

Первый проверяет скорости и геометрию; второй формирует косвенные кандидаты
целей по снарядам/HP. Это не проверенные engine target IDs и не полная оценка
точности таргетинга. Гипотеза радиуса атакующего проверяется только в аудиторе,
основная физика дальности не изменена. Исправлена подтверждённая двойная
конверсия скорости в публичном fallback: UnitSpec хранит raw Speed, Entity — mt/s.
