# v29: live-case fine-tune (2026-10-14)

Status: **experimental; do not promote to live**. The extra case did not produce a repeatable gain.

The previous live loss `20261013_225020_m009` was added to the `meta` opponent pool as a reconstructed, synthetic deck. This probe initialized from the live-tested v25 checkpoint. It did not use the match's actions as demonstrations, because the saved match has no full observations at each action.

Training: 30,720 PPO steps, six CPU environments, `brain_share=0.35`, learning rate `5e-5`, 3,072 critic-only warmup steps, KL anchor 0.03 to v25, default rewards and action delay. MPS was unavailable on this host. The sampled live case was one of 15 `meta` decks, not a dedicated curriculum.

```bash
.venv-live/bin/python -m sim.train_ppo \
  --opponent meta --brain-share 0.35 --steps 30720 --envs 6 --rollout 128 \
  --init checkpoints/portable_v25/portable_v25_cannon_economy_best.pt \
  --anchor checkpoints/portable_v25/portable_v25_cannon_economy_best.pt \
  --anchor-coef 0.03 --lr 0.00005 --value-warmup 3072 \
  --eval-every 15360 --eval-episodes 16 --eval-opponent brain,meta \
  --device cpu --name portable_v29_livecase_probe
```

Paired greedy evaluation, 80 episodes per opponent and seed block:

| Seed block | Opponent | v25 wins | v29 wins | Paired better / worse | One-sided p |
| --- | --- | ---: | ---: | ---: | ---: |
| 1000108 | brain | 67 | 69 | 11 / 9 | 0.412 |
| 1000108 | meta | 35 | 41 | 14 / 8 | 0.143 |
| 1000000 | brain | 66 | 65 | 11 / 12 | 0.661 |
| 1000000 | meta | 44 | 39 | 7 / 12 | 0.916 |

The first seed block was selected to include 12/80 games against the reconstructed case. On those, v25 won 2/12 and v29 won 5/12 (four changed losses to wins, one changed win to a loss). This is a targeted diagnostic, **not** an unbiased estimate of general ladder performance. The second block was chosen without that enrichment; it contains three case games, where v25 won 1/3 and v29 won 0/3. Neither block gives statistically persuasive evidence of an improvement; the second reverses the apparent gain.

Artifacts: `checkpoints/portable_v29/portable_v29_livecase_probe_best.pt`, `reports/eval_portable_v29_livecase_probe_80.json`, and `reports/eval_portable_v29_livecase_probe_holdout_80.json`. Keep v25 as the live reference. Next experiment should make the case more frequent *only in training*, while selecting on independent non-enriched holdout sets and testing several seeds; a single reconstructed deck may also miss the real cause of the live loss.
