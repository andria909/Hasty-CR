# First three Cannon audit timeouts: bounded offline diagnosis

Reproduced the first three `timeout` IDs from `catalog_cannon_audit_v11.jsonl`, with unchanged current `DefenseEnv`, `defender_action(...,'cannon_cycle')`, seed42, legacy geometry, physical_live coordinates, max_ms240000. Only simulator steps; no training, inference, devices, runtime edits, cap changes or case exclusions. Full per-body UID/name/side/HP/position/state/target UID, ten-second history, recent damage, accepted defender actions, scheduled enemy plays and pending-effect queues are in `diagnostics.json`. Reproducer: `PYTHONPATH=. .venv-live/bin/python reports/defense_plan/cannon_timeout_first3_20261022/diagnose_timeouts.py`.

All three reached exactly480 decisions/240000ms, phase `defense`, every scheduled enemy play completed, no illegal actions. They show live combat against repeatedly renewed ground defenders, rather than an untriggered script or a demonstrated pathfinding deadlock. Stationary ranged attackers remain in range of their current live targets; changing target UIDs and damage logs matter more than their unchanged coordinates. This does **not** establish live-game targeting parity or rule out other engine errors.

| Scenario (suffix `/back/xN/right`) | Enemy plays complete | Own tower HP lost L/R/K | Accepted defender actions | Enemy survivors at timeout |
|---|---:|---|---:|---|
| arrows:flying_machine:lava_hound:skeleton_dragons, x3 |4/4|0/0/0|133|Lava Hound UID7 HP2623; Flying Machine profile `dart_barrell` UID14 HP614|
| baby_dragon:balloon:lava_hound, x2 |3/3|0/1280/0|90|Lava Hound UID7 HP2326; Baby Dragon UID20 HP1152|
| baby_dragon:balloon:lava_hound:zap, x2 |4/4|0/1280/0|90|Baby Dragon UID20 HP162; Lava Pup UID44 HP216|

## Concrete activity near the time limit

First case: Lava Hound stays at(14500,12435) while attacking successive Cannon targets UID124→130→136→142→148→154 between190–240s. Its HP falls3217→3019→2920→2821→2722→2623 across those samples. The Flying Machine at(14500,9960) repeatedly acquires fresh Skeleton/ground targets (UID135→141→151→155→159→169), changing WINDUP/ATTACKING state, while retaining614HP. At240s both targetUIDs154/169 are alive. Two ordinary projectiles remain in flight, due240009/240095ms; pending area/spell effects and pending player action are empty. This is not a projectile/effect that never expires.

Second case: Lava Hound attacks fresh CannonUIDs87→93→99→105→111; Baby Dragon moves from(12061,9594) at190s to(12926,12200) at240s, switches targetUIDs and continues damage. At238700ms Hound damages Cannon111 for54; at239600ms Dragon damages Knight116 for160; at240000ms Hound again damages Cannon111 for54. No effects/projectiles are pending at timeout.

Third case: Baby Dragon at(13559,12665) attacks repeatedly replaced defenders while HP falls459→360→261→162 across200–230s. Lava Pup UID44 changes position and targets throughout190–240s. Last observed damage includes Dragon→Knight122 for160 at238950ms and enemy King→Knight122 for128 at239800ms. No effects/projectiles are pending at timeout.

The defender continues spending until the limit (last Cannon235650ms, Zap238650ms in the latter two cases). Cannon cannot attack air; its fixed placement repeatedly draws building-targeting air units. The priority/positions stress the engine and do not guarantee an efficient anti-air response; spells target a fixed opposite-lane cell. Long-lived air bodies under this deliberately weak, endlessly renewed defense are therefore plausible simulator outcomes. They must remain reported as timeouts, not successes or invalid scripts. No basis here to increase the cap or drop these scenarios. Acceptance still needs separate quality criteria and verified real-game mechanics; the current first case also carries an approximate Arrows mechanics flag.
