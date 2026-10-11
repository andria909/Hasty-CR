# Live validation: v11 with strategic action masks

Date: 2026-10-11

## Setup

- Checkpoint: `checkpoints/portable_v11/portable_v11_buildings.pt` (150,016 steps)
- Deck: Cannon, Fireball, Hog Rider, Knight, Electro Spirit, Musketeer, Skeletons, Zap
- Vision: BuildABot fallback (no local YOLO detector checkpoint was available)
- Policy temperature: 0 (greedy)
- Match limit: 1

## Result

- Outcome: win, confirmed by the Clash Royale result screen
- Trophy change: +3
- Duration: 238 seconds
- Final princess-tower fractions: ally `0.00/1.00`, enemy `0.00/0.70`
- Plays: 55
- Hog Rider plays: 6 (11% of all plays)
- Average decision-loop latency observed in the log: about 0.5-0.6 seconds, dominated by screenshot capture

## Strategic-mask observations

- Hog Rider was placed only in the configured bridge pocket (`x=2`, `y=16-17`).
- Hog Rider was not played while the live detector reported pressure on our half.
- Zap was played only when the detector supplied a target cluster or reset target.
- Fireball was played three times at the same upper-left target area. The current log does not record the exact units inside the spell radius, so this one-match result is not enough to claim that spell targeting is solved.

## Training decision

The newly trained `portable_v16_strategic_actual_best.pt` was not promoted. On paired 60-game evaluations per opponent using identical seeds:

- `brain`: v11 53/60, v16 50/60
- `meta`: v11 36/60, v16 38/60

The small meta gain did not compensate for the brain regression and neither difference was statistically convincing. The live runner therefore keeps v11 and applies the new strategic masks around it.

## Next priorities

1. Record the units and confidence that opened each spell target in the live match log.
2. Improve or train the local unit detector; the current fallback produces 25 hand-classifier overrides in one match.
3. Validate over more matches before changing the production checkpoint.
