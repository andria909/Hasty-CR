# Результаты короткого обучения

Проверено CUDA-шагов: 819200. Два тренировочных seed повторяют72 validation-состояния;144 строки не являются144 независимыми контекстами.

| Профиль | Победы | Воздух | Голем | Мушкетёр за игру | Hog за игру |
|---|---:|---:|---:|---:|---:|
| Исходник | 44/72 | 0/12 | 0/12 | 0.12 | 1.94 |
| joint_control | 89/144 | 1/24 | 0/24 | 0.25 | 1.85 |
| joint_bc800 | 89/144 | 4/24 | 9/24 | 2.50 | 4.76 |
| joint_replay100 | 65/144 | 11/24 | 5/24 | 4.66 | 5.39 |
| joint_replay500 | 65/144 | 6/24 | 6/24 | 4.76 | 5.56 |

## Локальный навык: перед PPO и выбранный PPO-чекпойнт

| Профиль | До PPO | Выбранный чекпойнт PPO |
|---|---:|---:|
| joint_control | 0/16 | 0/16 |
| joint_bc800 | 0/16 | 0/16 |
| joint_replay100 | 0/16 | 0/16 |
| joint_replay500 | 0/16 | 0/16 |

8 локальных состояний повторяются для двух seed. Это успешные действия из учебного упражнения, не победы в матчах. Чекпойнт выбирается по полному validation, не по этой таблице.

## Fireball и обе линии

| Профиль | Fireball за матч | Пустые | По ≥3 юнитам | Целей за матч | Потери левой / правой башни |
|---|---:|---:|---:|---:|---:|
| joint_control | 0.44 | 0.24 | 0.07 | 0.48 | 0.326 / 0.339 |
| joint_bc800 | 4.24 | 1.22 | 0.73 | 6.83 | 0.531 / 0.390 |
| joint_replay100 | 6.09 | 0.92 | 1.55 | 13.31 | 0.541 / 0.519 |
| joint_replay500 | 6.33 | 1.30 | 1.35 | 12.35 | 0.553 / 0.528 |

Потери — средняя доля исходного HP своей башни. Это не частота атак по линии. Все карты по каждой линии отдельно сохранены в JSON mean_card_lane_plays.

## Каждый локальный навык отдельно

| Профиль | Навык | До PPO | Выбранный PPO |
|---|---|---:|---:|
| joint_control | air_cannon_efficient | 0/4 | 0/4 |
| joint_control | air_defense | 0/0 | 0/0 |
| joint_control | air_electro_spirit_recovery | 0/4 | 0/4 |
| joint_control | air_fireball_recovery | 0/4 | 0/4 |
| joint_control | air_reserve_deploy | 0/0 | 0/0 |
| joint_control | air_reserve_wait | 0/0 | 0/0 |
| joint_control | air_zap_recovery | 0/4 | 0/4 |
| joint_control | fireball_group | 0/0 | 0/0 |
| joint_control | fireball_restraint | 0/0 | 0/0 |
| joint_bc800 | air_cannon_efficient | 0/4 | 0/4 |
| joint_bc800 | air_defense | 0/0 | 0/0 |
| joint_bc800 | air_electro_spirit_recovery | 0/4 | 0/4 |
| joint_bc800 | air_fireball_recovery | 0/4 | 0/4 |
| joint_bc800 | air_reserve_deploy | 0/0 | 0/0 |
| joint_bc800 | air_reserve_wait | 0/0 | 0/0 |
| joint_bc800 | air_zap_recovery | 0/4 | 0/4 |
| joint_bc800 | fireball_group | 0/0 | 0/0 |
| joint_bc800 | fireball_restraint | 0/0 | 0/0 |
| joint_replay100 | air_cannon_efficient | 0/4 | 0/4 |
| joint_replay100 | air_defense | 0/0 | 0/0 |
| joint_replay100 | air_electro_spirit_recovery | 0/4 | 0/4 |
| joint_replay100 | air_fireball_recovery | 0/4 | 0/4 |
| joint_replay100 | air_reserve_deploy | 0/0 | 0/0 |
| joint_replay100 | air_reserve_wait | 0/0 | 0/0 |
| joint_replay100 | air_zap_recovery | 0/4 | 0/4 |
| joint_replay100 | fireball_group | 0/0 | 0/0 |
| joint_replay100 | fireball_restraint | 0/0 | 0/0 |
| joint_replay500 | air_cannon_efficient | 0/4 | 0/4 |
| joint_replay500 | air_defense | 0/0 | 0/0 |
| joint_replay500 | air_electro_spirit_recovery | 0/4 | 0/4 |
| joint_replay500 | air_fireball_recovery | 0/4 | 0/4 |
| joint_replay500 | air_reserve_deploy | 0/0 | 0/0 |
| joint_replay500 | air_reserve_wait | 0/0 | 0/0 |
| joint_replay500 | air_zap_recovery | 0/4 | 0/4 |
| joint_replay500 | fireball_group | 0/0 | 0/0 |
| joint_replay500 | fireball_restraint | 0/0 | 0/0 |

Удержание Fireball на пустом поле не компенсирует потерю защиты воздуха или группового прицела. Все группы входят в локальный holdout целиком.

Порог для продолжения прошли минимум два профиля.
Это validation для отбора; test и реальные бои не проводились. Частота карты сама по себе не доказывает качество.
