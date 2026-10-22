# Геометрия: независимый аудит и минимальная миграция

Проверен **основной** `/Users/andriashadrin/projects/Hasty-CR`, не устаревшая копия ветки. Только чтение кода/сохранённых логов и вычисления наблюдений: без `Match.step`, устройств, боёв, обучения и изменений runtime. Активные frozen stage2/evaluation пакеты не менялись.

## Что означает позиция

`native/arm64_entities.c:138,162` читает signed int32 непосредственно из entity `+0x7c/+0x80`; сторона находится в `+0x78`. Python `arm64_entities.py:149` экспортирует те же числа без pixel conversion и без поправки размера башни. Это **позиционный якорь игрового entity**, не координата отрисованного спрайта. Но эти поля сами по себе не доказывают, что якорь равен центру collision shape/footprint: ни форма коллизии, ни её локальный offset reader не экспортирует. Называть эти значения доказанным центром хитбокса нельзя.

Источники измерений: `reports/saved_arm64_motion_20261022_v2/{report.md,audit.json}`. В сохранённой выборке:

| Anchor | Native entity position | Current simulator |
|---|---|---|
| Top princess L/R | (3500/14500,6500) | (3500/14500,7500) |
| Bottom princess L/R | (3500/14500,25500) | (3500/14500,24500) |
| Top king | (9000,3000) | (9500,3500) |
| Bottom king | (9000,29000) | (9500,28500) |

Глобальный сдвиг координат это не исправит. Нельзя нелинейно растягивать координаты войск ради совмещения башен: это изменит дистанции/скорости/радиусы без основания.

## Три разных преобразования, которые нельзя смешивать

1. **Физический мир:** `arena.tile(x,y) = ((x+.5)*1000,(y+.5)*1000)`, размеры 18000×32000, y вниз. Именно это делает `adapter.grid_to_point`. `to_tiles` возвращает координату в клетках, **не индекс клетки**: для центра клетки 3 получается 3.5.
2. **Смена места игрока:** native bridge вращает мир `(18000-x,32000-y)` вокруг (9000,16000); `grid_to_point(...,side=-1)` зеркалит индексы `(17-x,31-y)` **до** добавления полуклетки. Эти два преобразования эквивалентны для центров клеток.
3. **Существующий encoder:** `env.observe` бинит `round(x/1000), round(31-y/1000)` и только потом зеркалит индексы для side=-1. `physical_live` подаёт временную proxy-позицию `31000-y`, чтобы отменить старый Y flip. Число 31000 здесь — компенсация encoder, **не** физический поворот вокруг середины поля. Bridge сохраняет реальные native positions отдельно.

### Подтверждённые ошибки контракта

**King symmetry:** текущий bottom king (9500,28500) при world180 становится (8500,3500), тогда как top king стоит в (9500,3500). Ошибка x=1000. У native kings x=9000, поэтому world180 совпадает. При нынешних anchors расстояния от king до L/R princess различаются; это влияет на физику, а не только на картинку.

**Квантование:** `round` использует ties-to-even, тогда как action world points уже имеют полуклетку. Четыре действия `(3,23),(4,23),(3,24),(4,24)` дают разные targets `(3500/4500,23500/24500)`, но все четыре в текущем `physical_live` попадают в observation bin `(4,24)`.

Проведён исчерпывающий перебор **576 action cells на каждый режим**, без тиков: один `Match` только создан, добавлено одно простое тело, вызван настоящий ROOT `observe`. Для physical_live применена ровно proxy-компенсация из bridge/defense_env.

| Encoder | Unique encoded cells /576 | Совпали с ожидаемой клеткой режима | Multiplicity of encoded cells |
|---|---:|---:|---|
| legacy_training | 160 | 160 с `(x,31-y)` | 32×2, 128×4 |
| physical_live | 170 | 170 с `(x,y)` | 4×1, 46×2, 120×4 |

Для legacy прямых совпадений с исходной `(x,y)` ноль из-за его отдельного Y flip; **это не метрика точности распознавания**. Проблема aliasing присутствует в обоих режимах. Различие 160/170 связано в том числе с clipping крайних bins. Исправление floor-квантования отдельно от изменения Y orientation: одно не заменяет другое.

## Минимальная конфигурация без monkeypatch

Предлагается два независимых versioned контракта:

- `ArenaGeometry`: immutable экземпляр с explicit ID/SHA, размерами, river/bridge параметрами, crown anchors и deployment/pocket параметрами с provenance. `legacy_v1` в точности сохраняет текущие числа; opt-in `native_anchors_candidate_v1` меняет **только шесть измеренных crown anchors**, остальные параметры пока наследует legacy с явными unresolved flags. Reader не доказал footprint, pocket, bridge width и collision offsets.
- `PolicyCoordinates`: отдельно orientation и binning. Существующие `legacy_training` и `physical_live` должны оставаться bitwise compatibility aliases. Новый opt-in `physical_live_cells_v2` сохраняет physical Y, меняет только binning на `floor(world/1000)` с прежним clip. Имена/shape/actions v25 не меняются; файл весов не переписывается. Но новые bins меняют входные значения, поэтому это новый контракт, требующий отдельной оценки/обучения, а не молчаливый «фикс v25».

При cell_floor_v2 action cell center имеет ожидаемый обратный индекс, и 180° world rotation согласуется с `(17-x,31-y)` без ties-to-even. Проверять все 576 cells и обе стороны. Внешние координаты native могут временно выходить за board: не подменять исходные world positions clipping-ом encoder; отражать отдельное предупреждение.

### Где передать geometry

1. `Match(...,geometry=legacy_v1)` передаёт тот же объект в `Battle`; `_build_towers` использует instance anchors. Defaults не меняются.
2. Не ограничиваться `_build_towers`: заменить anchor lookup в `Match.play_card` для burrow home (`match.py:494`), в `Battle` для ability anchors (`engine.py:1536–1539`) и drill resurfacing (`engine.py:3222–3223`). Предпочтительно реальные `match.towers`/переданный registry, а не вторые копии глобальных anchors.
3. `deploy_area_ok(...,geometry=...)`, `Match.play_card`, механическая маска и `_placeable` обязаны использовать **один** профиль. Pocket сейчас строится вокруг globals башен; после перемещения anchors иначе маска и реальные постановки разойдутся. При этом сам размер pocket остаётся недоказанным.
4. `_PLACEMENT_CACHE` должен включать geometry fingerprint. Pathfinding импортирует river/bridge constants **по значению** (`pathfind.py:24`) и кэширует по goal/blockers; если разрешать разные terrain-профили, передать geometry в walkability/field и включить fingerprint в `_FIELD_CACHE`. Для первого anchor-only профиля terrain не меняется: не заявлять, что этим уже исправлен river/bridge.
5. Передать geometry и policy coordinate IDs в DefenseEnv, native bridge, duel evaluator и checkpoint/config/resume; запретить неявную смену при resume. Старые .pt без metadata получают только explicit legacy default с warning. Старые frozen data не переименовывать и не перепаковывать задним числом.
6. `memory_play.tap_points` — отдельная калибровка screen pixels. Сейчас реальная постановка использует grid_x/y, а не placement_world x/y. Смена world anchors не является разрешением менять экранную калибровку; проверить её отдельным контрактом.

## Приёмка следующей реализации

- Native fixture со всеми шестью anchor positions проходит roundtrip нормализации для обоих own_side; destroyed lanes тоже отражаются правильно.
- Default legacy replay/observation/action/mask сохраняются bitwise; файл v25 и frozen archive SHA остаются прежними.
- Два Match разных geometry одновременно не влияют друг на друга; cache keys не пересекаются; snapshot/restore сохраняют profile IDs/SHA.
- Native candidate anchors зеркально симметричны; floor-cell encoder даёт 576 уникальных bins, exact roundtrip и seat symmetry. Отдельный negative test сохраняет известные legacy aliases, чтобы compatibility не поменялась случайно.
- Механическая маска и фактический `play_card` согласованы по deployment/pocket для обоих мест; приближённый pocket не объявляется измеренным.
- Отдельное сравнение новых/старых контрактов по native placements/траекториям. Реальные target UIDs пока отсутствуют: совпадение geometry и скорости не доказывает targeting. Не менять общий множитель скорости на основании этой работы.

## Привязка доказательств к коду

SHA256 прочитанных файлов ROOT:

- `sim/arena.py`: `1a0e91ae883908e78db934e6b34da864456050333f2f4fda1ad32a501eeec836`
- `sim/env.py`: `25a8b2e60eee035147cb59e9305dc08ab72c375e64e672d4da4f823a508554bb`
- `sim/adapter.py`: `3d12212efeb1583179382281729c7e2bd52daeaec94079e29265fde9bbf7f428`
- `scripts/gamestate/model_bridge.py`: `79019bb8c891ee87a866cdb2ecf1220d139ba9a76bdb59e4d1e7c03d7f148349`

Это проект следующего изменения и конкретные воспроизводимые дефекты; не отчёт о выполненной миграции или доказанной точности боевой симуляции.
