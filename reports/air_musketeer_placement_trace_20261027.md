# Musketeer placement diagnostic

Three paired TRAIN seeds per intervention. Sampling interval250ms. All final metrics match previous untraced games. No live games or runtime physics changes.

| Intervention | Units | Zero damage | Total damage, all targets | Mean first damage delay, seconds |
|---|---:|---:|---:|---:|
| none | 2 | 0 | 1166 | 4.50 |
| reserve | 10 | 1 | 7077 | 1.92 |
| reserve_and_deploy | 30 | 0 | 25610 | 1.67 |

All30 forced Musketeers dealt damage, but all three games were lost. Increasing card frequency alone did not solve the defense. First-damage delay excludes units with no damage.

Forced-mode target snapshots:383air,551ground. These are duration samples, not attacks or attributed damage; they do not establish a targeting bug. Physics remains unchanged.

If current reward studies fail, test a separate positional-defense curriculum: isolated achievable air threats, then supported pushes, while retaining full-game transfer validation. Select by game outcomes and tower survival, not raw card frequency.

[Raw trace](defense_plan/air_musketeer_placement_trace_20261027.json).
