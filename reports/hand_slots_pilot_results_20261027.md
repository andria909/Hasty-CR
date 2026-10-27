# Результаты короткого обучения

Проверено CUDA-шагов: 819200. Два тренировочных seed повторяют72 validation-состояния;144 строки не являются144 независимыми контекстами.

| Профиль | Победы | Воздух | Голем | Мушкетёр за игру | Hog за игру |
|---|---:|---:|---:|---:|---:|
| Исходник | 44/72 | 0/12 | 0/12 | 0.12 | 1.94 |
| joint_control | 89/144 | 1/24 | 0/24 | 0.25 | 1.85 |
| joint_bc800 | 58/144 | 5/24 | 6/24 | 2.83 | 5.06 |
| joint_replay100 | 32/144 | 4/24 | 3/24 | 4.86 | 5.36 |
| joint_replay500 | 24/144 | 5/24 | 6/24 | 4.98 | 4.91 |

## Локальный навык: перед PPO и выбранный PPO-чекпойнт

| Профиль | До PPO | Выбранный чекпойнт PPO |
|---|---:|---:|
| joint_control | 0/24 | 0/24 |
| joint_bc800 | 0/24 | 0/24 |
| joint_replay100 | 0/24 | 0/24 |
| joint_replay500 | 0/24 | 1/24 |

12 локальных состояний повторяются для двух seed. Это успешные действия из учебного упражнения, не победы в матчах. Чекпойнт выбирается по полному validation, не по этой таблице.

## Fireball и обе линии

| Профиль | Fireball за матч | Пустые | По ≥3 юнитам | Целей за матч | Потери левой / правой башни |
|---|---:|---:|---:|---:|---:|
| joint_control | 0.44 | 0.24 | 0.07 | 0.48 | 0.326 / 0.339 |
| joint_bc800 | 5.30 | 0.49 | 1.30 | 10.42 | 0.625 / 0.522 |
| joint_replay100 | 6.62 | 0.38 | 2.18 | 17.01 | 0.635 / 0.582 |
| joint_replay500 | 6.44 | 0.35 | 1.97 | 16.60 | 0.654 / 0.648 |

Потери — средняя доля исходного HP своей башни. Это не частота атак по линии. Все карты по каждой линии отдельно сохранены в JSON mean_card_lane_plays.

## Каждый локальный навык отдельно

| Профиль | Навык | До PPO | Выбранный PPO |
|---|---|---:|---:|
| joint_control | air_cannon_efficient | 0/8 | 0/8 |
| joint_control | air_defense | нет примеров | нет примеров |
| joint_control | air_electro_spirit_recovery | 0/8 | 0/8 |
| joint_control | air_fireball_recovery | 0/4 | 0/4 |
| joint_control | air_reserve_deploy | нет примеров | нет примеров |
| joint_control | air_reserve_wait | нет примеров | нет примеров |
| joint_control | air_zap_recovery | 0/4 | 0/4 |
| joint_control | fireball_group | нет примеров | нет примеров |
| joint_control | fireball_restraint | нет примеров | нет примеров |
| joint_control | hand_slot_permutation | нет примеров | нет примеров |
| joint_bc800 | air_cannon_efficient | 0/8 | 0/8 |
| joint_bc800 | air_defense | нет примеров | нет примеров |
| joint_bc800 | air_electro_spirit_recovery | 0/8 | 0/8 |
| joint_bc800 | air_fireball_recovery | 0/4 | 0/4 |
| joint_bc800 | air_reserve_deploy | нет примеров | нет примеров |
| joint_bc800 | air_reserve_wait | нет примеров | нет примеров |
| joint_bc800 | air_zap_recovery | 0/4 | 0/4 |
| joint_bc800 | fireball_group | нет примеров | нет примеров |
| joint_bc800 | fireball_restraint | нет примеров | нет примеров |
| joint_bc800 | hand_slot_permutation | нет примеров | нет примеров |
| joint_replay100 | air_cannon_efficient | 0/8 | 0/8 |
| joint_replay100 | air_defense | нет примеров | нет примеров |
| joint_replay100 | air_electro_spirit_recovery | 0/8 | 0/8 |
| joint_replay100 | air_fireball_recovery | 0/4 | 0/4 |
| joint_replay100 | air_reserve_deploy | нет примеров | нет примеров |
| joint_replay100 | air_reserve_wait | нет примеров | нет примеров |
| joint_replay100 | air_zap_recovery | 0/4 | 0/4 |
| joint_replay100 | fireball_group | нет примеров | нет примеров |
| joint_replay100 | fireball_restraint | нет примеров | нет примеров |
| joint_replay100 | hand_slot_permutation | нет примеров | нет примеров |
| joint_replay500 | air_cannon_efficient | 0/8 | 0/8 |
| joint_replay500 | air_defense | нет примеров | нет примеров |
| joint_replay500 | air_electro_spirit_recovery | 0/8 | 0/8 |
| joint_replay500 | air_fireball_recovery | 0/4 | 0/4 |
| joint_replay500 | air_reserve_deploy | нет примеров | нет примеров |
| joint_replay500 | air_reserve_wait | нет примеров | нет примеров |
| joint_replay500 | air_zap_recovery | 0/4 | 1/4 |
| joint_replay500 | fireball_group | нет примеров | нет примеров |
| joint_replay500 | fireball_restraint | нет примеров | нет примеров |
| joint_replay500 | hand_slot_permutation | нет примеров | нет примеров |

Удержание Fireball на пустом поле не компенсирует потерю защиты воздуха или группового прицела. Разделение выполнено по целым seed. Навыки без отложенных примеров здесь не проверены.

Минимум два профиля порог не прошли: большое продолжение этого исследования не запускается.
Это validation для отбора; test и реальные бои не проводились. Частота карты сама по себе не доказывает качество.
