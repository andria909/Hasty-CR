# Результаты короткого обучения

Проверено CUDA-шагов: 819200. Два тренировочных seed повторяют72 validation-состояния;144 строки не являются144 независимыми контекстами.

| Профиль | Победы | Воздух | Голем | Мушкетёр за игру | Hog за игру |
|---|---:|---:|---:|---:|---:|
| Исходник | 44/72 | 0/12 | 0/12 | 0.12 | 1.94 |
| joint_control | 89/144 | 1/24 | 0/24 | 0.25 | 1.85 |
| joint_bc800 | 73/144 | 3/24 | 5/24 | 2.81 | 5.41 |
| joint_replay100 | 73/144 | 2/24 | 5/24 | 4.40 | 5.22 |
| joint_replay500 | 51/144 | 4/24 | 8/24 | 4.65 | 5.34 |

## Локальный навык: перед PPO и выбранный PPO-чекпойнт

| Профиль | До PPO | Выбранный чекпойнт PPO |
|---|---:|---:|
| joint_control | 0/24 | 0/24 |
| joint_bc800 | 0/24 | 0/24 |
| joint_replay100 | 0/24 | 0/24 |
| joint_replay500 | 0/24 | 0/24 |

12 локальных состояний повторяются для двух seed. Это успешные действия из учебного упражнения, не победы в матчах. Чекпойнт выбирается по полному validation, не по этой таблице.

## Fireball и обе линии

| Профиль | Fireball за матч | Пустые | По ≥3 юнитам | Целей за матч | Потери левой / правой башни |
|---|---:|---:|---:|---:|---:|
| joint_control | 0.44 | 0.24 | 0.07 | 0.48 | 0.326 / 0.339 |
| joint_bc800 | 4.65 | 0.73 | 0.83 | 8.40 | 0.583 / 0.488 |
| joint_replay100 | 6.01 | 1.28 | 1.29 | 12.02 | 0.584 / 0.413 |
| joint_replay500 | 6.00 | 0.99 | 1.38 | 12.10 | 0.626 / 0.583 |

Потери — средняя доля исходного HP своей башни. Это не частота атак по линии. Все карты по каждой линии отдельно сохранены в JSON mean_card_lane_plays.

## Каждый локальный навык отдельно

| Профиль | Навык | До PPO | Выбранный PPO |
|---|---|---:|---:|
| joint_control | air_cannon_efficient | 0/8 | 0/8 |
| joint_control | air_defense | 0/0 | 0/0 |
| joint_control | air_electro_spirit_recovery | 0/8 | 0/8 |
| joint_control | air_fireball_recovery | 0/4 | 0/4 |
| joint_control | air_reserve_deploy | 0/0 | 0/0 |
| joint_control | air_reserve_wait | 0/0 | 0/0 |
| joint_control | air_zap_recovery | 0/4 | 0/4 |
| joint_control | fireball_group | 0/0 | 0/0 |
| joint_control | fireball_restraint | 0/0 | 0/0 |
| joint_bc800 | air_cannon_efficient | 0/8 | 0/8 |
| joint_bc800 | air_defense | 0/0 | 0/0 |
| joint_bc800 | air_electro_spirit_recovery | 0/8 | 0/8 |
| joint_bc800 | air_fireball_recovery | 0/4 | 0/4 |
| joint_bc800 | air_reserve_deploy | 0/0 | 0/0 |
| joint_bc800 | air_reserve_wait | 0/0 | 0/0 |
| joint_bc800 | air_zap_recovery | 0/4 | 0/4 |
| joint_bc800 | fireball_group | 0/0 | 0/0 |
| joint_bc800 | fireball_restraint | 0/0 | 0/0 |
| joint_replay100 | air_cannon_efficient | 0/8 | 0/8 |
| joint_replay100 | air_defense | 0/0 | 0/0 |
| joint_replay100 | air_electro_spirit_recovery | 0/8 | 0/8 |
| joint_replay100 | air_fireball_recovery | 0/4 | 0/4 |
| joint_replay100 | air_reserve_deploy | 0/0 | 0/0 |
| joint_replay100 | air_reserve_wait | 0/0 | 0/0 |
| joint_replay100 | air_zap_recovery | 0/4 | 0/4 |
| joint_replay100 | fireball_group | 0/0 | 0/0 |
| joint_replay100 | fireball_restraint | 0/0 | 0/0 |
| joint_replay500 | air_cannon_efficient | 0/8 | 0/8 |
| joint_replay500 | air_defense | 0/0 | 0/0 |
| joint_replay500 | air_electro_spirit_recovery | 0/8 | 0/8 |
| joint_replay500 | air_fireball_recovery | 0/4 | 0/4 |
| joint_replay500 | air_reserve_deploy | 0/0 | 0/0 |
| joint_replay500 | air_reserve_wait | 0/0 | 0/0 |
| joint_replay500 | air_zap_recovery | 0/4 | 0/4 |
| joint_replay500 | fireball_group | 0/0 | 0/0 |
| joint_replay500 | fireball_restraint | 0/0 | 0/0 |

Удержание Fireball на пустом поле не компенсирует потерю защиты воздуха или группового прицела. Все группы входят в локальный holdout целиком.

Минимум два профиля порог не прошли: большое продолжение этого исследования не запускается.
Это validation для отбора; test и реальные бои не проводились. Частота карты сама по себе не доказывает качество.
