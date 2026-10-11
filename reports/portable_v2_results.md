# Portable PPO v2 — local training result

Date: 2026-10-11

This run tests whether the new public-state features, event rewards and
factorised spatial policy can improve on the behaviour-clone baseline on a
clean checkout, without downloaded Clash Royale client assets.

## Result

The candidate is statistically better than the locally reproduced baseline
against the nine-deck meta pool.

| Opponent | Baseline wins | PPO wins | Paired better / worse / tied | One-sided exact p | Mean crown-differential gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| rule brain | 4/60 | 10/60 | 8 / 2 / 50 | 0.05469 | +0.583 |
| meta pool | 0/60 | 15/60 | 15 / 0 / 45 | 0.0000305 | +1.917 |

The 95% Wilson win-rate intervals against meta are 0–6.0% for the clone and
15.8–37.2% for PPO. The paired exact sign test is the primary test because both
policies face the same deck and match seed. Against the rule brain the result
is an improvement trend, but narrowly misses the predeclared 5% threshold.

Raw per-game results, deck names, Wilson intervals and the paired tests are in
`reports/portable_v2_paired_eval.json`.

## Reproduction

Behaviour clone:

```bash
python -m sim.clone \
  --episodes 150 --epochs 10 --batch 256 \
  --hold-share 0.10 --hold-loss-weight 0.10 \
  --name portable_bc_balanced_v2 --device cpu
```

PPO:

```bash
python -m sim.train_ppo \
  --steps 50000 --envs 4 --rollout 64 --lr 0.0001 \
  --entropy 0.02 --entropy-final 0.005 \
  --entropy-hold 10000 --entropy-anneal 50000 \
  --eval-every 10000 --eval-episodes 20 \
  --name portable_ppo_v2 --opponent brain --eval-opponent brain \
  --init tmp/rl/portable_bc_balanced_v2.pt \
  --anchor tmp/rl/portable_bc_balanced_v2.pt --anchor-coef 0.05 \
  --value-warmup 5000 --value-warmup-lr 0.001 \
  --elixir 0.05 --device cpu
```

Paired held-out evaluation:

```bash
python -m sim.eval_checkpoints \
  --baseline tmp/rl/portable_bc_balanced_v2.pt \
  --candidate tmp/rl/portable_ppo_v2_best.pt \
  --opponents brain,meta --episodes 60 --seed 920000 --device cpu \
  --output reports/portable_v2_paired_eval.json
```

## What changed

- Public opponent memory: revealed cards, known cycle exclusions, inferred
  elixir interval/recent spend, and known Hog counters. The simulator and live
  bridge share one feature layout, and no hidden hand or exact internal state
  is exposed to training.
- Spell rewards use resolved Fireball/Log hit events. They score effective
  unit-HP damage, kill value and genuine pre-cast prediction, while excluding
  tower chip already covered by the main reward and penalising misses.
- Kiting credit is awarded only when Ice Golem, Ice Spirit or Skeletons really
  retarget a threat; displacement and absorbed damage add credit, and tower
  damage removes it.
- The old roughly 20M-parameter flattening policy was replaced by a 404,050
  parameter factorised spatial network with hand-conditioned placement queries
  and absolute coordinate bases.
- Behaviour-clone loss downweights hold rows and checkpoint selection includes
  play accuracy, preventing an always-hold majority-class model from winning.
- A versioned RoyaleAPI fallback makes 2.6 training possible without the
  untracked extracted client dump. Full client data remains preferred when it
  is present.

## Verification and limits

The focused training/serving/reward suite passes. A broad dependency-reduced
run produced 1062 passed, 18 skipped, 2 xfailed and 359 failed tests. The failed
tests mostly cover champions, evolutions, exotic spells and exact client action
graphs that the public JSON fallback does not contain; they require the full
`tmp/gamedata/csv_logic` extraction. The portable result therefore proves an
improvement inside the locally reproducible simulator, not that it beats the
repository author's reported 94% checkpoint (those weights are not shipped)
and not yet that it wins on a live Clash Royale client.

## Follow-up mixed-opponent run

A further 100,096-step run started from v2, trained on an even brain/meta mix,
and selected on the worse of the two evaluation scores. Its 50k checkpoint was
the best periodic snapshot. A fresh paired test at seed 930000 did **not** show
a reliable improvement over v2:

| Opponent | v2 wins | mixed v3 wins | Paired better / worse / tied | One-sided exact p |
| --- | ---: | ---: | ---: | ---: |
| rule brain | 7/60 | 9/60 | 8 / 6 / 46 | 0.3953 |
| meta pool | 11/60 | 10/60 | 5 / 6 / 49 | 0.7256 |

Accordingly v2 remained the accepted candidate at that stage; v3 is retained
only as a negative experiment. Raw results are in
`reports/portable_v3_paired_eval.json`.

## Conditioned behaviour clone v6

Inspection of the imitation stage found that a static per-slot placement map
could not express different geography for cards occupying the same slot. Card
identity reached the old spatial head only as one constant bias, so Cannon,
Hog and Ice Spirit had to share essentially the same location ranking. The v6
head adds feature-conditioned queries over the spatial tensor plus learned
coefficients over fixed board-coordinate bases. Old v2 checkpoints migrate
with zero-initialised new heads and preserve their logits exactly.

The cloning objective also now contains:

- a separate greedy-compatible play-versus-hold gate loss;
- soft credit for nearby legal placements of the same card;
- checkpoint selection on gate accuracy plus near-placement accuracy;
- cached teacher datasets and meta-opponent state collection.

On 30,193 cached teacher states, play exact accuracy rose from roughly 1% to
18.5%, within-one-cell accuracy reached 41.5%, and play/hold accuracy reached
85.6%. The internal evaluation was 8–8 with a 26–26 crown score.

Most importantly, a fresh paired evaluation at seed 950000 beat the previously
accepted PPO v2 checkpoint:

| Opponent | PPO v2 wins | conditioned clone v6 wins | Paired better / worse / tied | One-sided exact p |
| --- | ---: | ---: | ---: | ---: |
| rule brain | 8/60 | 12/60 | 9 / 5 / 46 | 0.2120 |
| meta pool | 18/60 | 34/60 | 24 / 8 / 28 | **0.00350** |

The meta win rate improved from 30.0% to 56.7%, with a mean crown-differential
gain of +1.083 per match. v6 is therefore the accepted local candidate. The
raw paired rows and Wilson intervals are in
`reports/portable_bc_meta_v6_paired_eval.json`. v5's negative comparison is
retained in `reports/portable_bc_meta_v5_paired_eval.json`.

Reproduce v6 from the cached/collected teacher set with:

```bash
python -m sim.clone \
  --episodes 400 --epochs 12 --batch 256 \
  --hold-share 0.10 --hold-loss-weight 0.10 \
  --placement-radius 1 --act-loss-weight 1.0 --opponent meta \
  --data-cache tmp/rl/portable_bc_meta_v4_data.npz \
  --name portable_bc_meta_v6_conditioned --device cpu
```

### Live promotion state

The accepted weights were copied out of scratch storage to
`checkpoints/portable_v6/portable_bc_meta_v6_conditioned.pt`. Its SHA-256 is
`5e0e7eddada35db179a7f3162edc38735bfd58c6cf2b5281bc29d41076c42fb7`;
the local manifest records the training and paired-evaluation evidence. It is
an accepted **shadow/friendly candidate**, not yet a ladder release.

The first real-client gate is deliberately tap-free. With MuMu and Clash
Royale already open on the Windows host, run:

```powershell
.\run.ps1 -Brain rl -Shadow -Matches 1 `
  -Checkpoint 'checkpoints\portable_v6\portable_bc_meta_v6_conditioned.pt'
```

`-Shadow` implies both `--no-queue` and `--dry-run`: the user starts a friendly
match, the live detector feeds observations into v6, and recommendations are
logged without pressing Battle, selecting a card, or placing it. Only after
the shadow trace passes orientation, hand, latency and illegal-action checks
should the same checkpoint be tried in one bounded friendly match without
`-Shadow`.

## Post-v6 training experiments

v6 remains the champion after three deliberately conservative follow-ups:

- **v7 PPO:** 200,192 steps, 25k critic-only warmup, mixed brain/meta diet,
  self-play league, KL guard and a 0.10 v6 anchor. The best internal snapshot
  lost the fresh seed-960000 comparison: brain 7/60 vs 9/60 and meta 26/60 vs
  28/60. Raw rows are in `reports/portable_v7_paired_eval.json`.
- **v8 continued clone:** added a weights-only `--init` path and continued v6
  for eight epochs at LR 2e-4. Validation act accuracy rose from 85.6% to
  86.7%, exact placement from 18.5% to 19.4%, and near placement from 41.5%
  to 43.0%. A 60-game screen suggested a meta gain (42 vs 35, p=0.0946), but
  an independent 120-game confirmation reversed it (57 vs 64); brain was
  neutral at 14 vs 13. v8 was therefore rejected. Both raw reports are kept.
- **v9 prediction-focused PPO:** fixed tower-only Fireball/Log casts being
  incorrectly counted as misses, raised prediction reward, lowered LR to
  2e-5 and increased the v6 anchor to 0.25. Meta improved during training but
  brain regressed, so the worst-opponent selector retained the critic-warmup
  snapshot. Its action logits are bit-identical to v6; only the value head
  changed, so it is not a new playing-policy champion.

The new reward audit records raw unweighted event totals rather than trying to
infer them from return. Over 100 v6 matches against the meta pool, the current
coefficients contributed per match approximately +1.692 spell quality,
+0.514 kite quality, -0.244 elixir trade, -0.096 spell misses, but only +0.023
spell prediction. Prediction occurred in 13/100 matches. This explains why
the original prediction coefficient was almost invisible and also shows why
simply increasing it can over-specialise the policy. The complete audit is in
`reports/portable_v6_reward_audit.json` and can be reproduced with
`scripts/reward_audit.py`.

### v10 conservative policy soup

Direct PPO checkpoints changed too much at once, so v10 interpolates 90% of
v6 with 10% of the best v7 PPO weights. This small move was positive on both
opponents in both independent evaluations:

- 30-game screen (seed 991000): brain 6 vs 4, meta 16 vs 13;
- 120-game confirmation (seed 997000): brain 26 vs 22, meta 62 vs 57.

The confirmatory paired outcomes combine to 39 better and 30 worse
(`p=0.168`, one-sided), so this is a promising offline candidate rather than a
statistically proven replacement for v6. Crown differential also improved on
both opponents. On the same 100-game reward audit, wins rose 50 to 52,
prediction events rose from 13 to 21 matches, and misses fell from 1.91 to
1.65 per match. Kite quality fell from 3.43 to 3.25, which remains the main
trade-off to watch. The portable candidate and its manifest are under
`checkpoints/portable_v10/`; it is not approved for ladder play.

`spell_prediction` is deliberately treated as a noisy diagnostic proxy, not
an optimisation target or an acceptance metric. It detects the narrow
cast-before-deploy-then-hit pattern, but cannot infer intent, zone denial,
forced placements, or a good prediction that the opponent avoided. Policy
selection therefore remains based on paired match outcomes and crowns across
brain and meta opponents; prediction reward stays bounded and subordinate to
the actual game result.

## v11 building-cycle policy

The policy already had a spatial `enemy_buildings` plane and public one-hot
cards, but the compact counter summary mixed buildings with troops and spells.
v11 appends two honest public features: the number of revealed building
counters and whether cycle tracking says one may be available. It never sees
the opponent's hidden hand. Old checkpoints migrate with zero weights for the
new columns and therefore preserve their logits exactly.

To avoid catastrophic forgetting, PPO updated only those two appended input
columns and the critic. Every established policy parameter remained bit-exact.
Three task-vector soups with the higher-accuracy v8 clone were also screened
first and rejected because their meta or brain result regressed.

Against v10, a 60-game screen improved brain 12 to 17 and meta 23 to 26. Two
independent 120-game-per-opponent confirmations combined to:

- brain: 56 wins vs 44, paired 40 better / 28 worse / 172 tied;
- meta: 132 wins vs 125, paired 17 better / 10 worse / 213 tied.

The causal split is especially useful: among 130 meta matches against decks
with a building, wins improved 36 to 43; among 110 matches without a building,
not one paired outcome changed. A separate direct comparison against v6 gave
33 vs 24 wins on brain and 56 vs 56 on meta. v11 is therefore the accepted
offline champion, but still requires shadow/friendly validation before any
ladder use.

## v14 active-building lane-risk experiment

v14 appends two live public features describing how strongly active enemy
buildings threaten the left and right bridge. The simulator and live
observation bridge compute the same values, and feature-only PPO leaves the
established v11 policy weights untouched.

The final 125,440-step checkpoint beat v11 35–25 in a 60-game direct paired
screen, but the confidence interval still contained an even split. Against
fixed opponents it scored 14/60 versus brain and 29/60 versus meta. That
combination is not robust enough to replace v11. The code, both checkpoints,
manifest, and raw reports are retained for reproducibility, while v11 remains
the live shadow/friendly default.
