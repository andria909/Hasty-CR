"""Matched TRAIN-only flat versus grouped greedy decoding; never modifies weights.

Grouped greedy chooses the largest aggregate card/WAIT probability first, then
that card's best legal cell. It is a different deployment rule, not joint MAP.
"""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from scripts.kaggle.entity_experiment import sha
from sim import train_attack_ppo as train


def grouped_action(logits, mask):
    legal = logits.masked_fill(~mask, -torch.inf)
    maps = legal[1:2305].reshape(4, 576)
    groups = torch.cat((legal[:1], maps.logsumexp(-1), legal[2305:]))
    group = int(groups.argmax())
    if group == 0:
        return 0
    if group > 4:
        return 2305 + group - 5
    return 1 + (group - 1) * 576 + int(maps[group - 1].argmax())


def options(config):
    keys = ('curriculum', 'continuous_profile', 'validation_selection', 'continuous_focus',
            'geometry', 'coordinate_mode', 'max_ms', 'action_delay_ms', 'deadline_scale',
            'visible_fraction', 'initial_hand_mode', 'tower_support_weight', 'spell_waste_weight',
            'crown_only_weight', 'dormant_king_weight', 'tactical_start_fraction', 'entity_inputs')
    result = {k: config[k] for k in keys if k in config}
    defense, attack = train.reward_objects(config['reward_profile'])
    return dict(result, defense_rewards=defense, attack_rewards=attack)


def run(root, output):
    torch.set_num_threads(1)
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    models = []
    for case in ('cnn', 'entity64', 'entity96'):
        for seed in (512, 513):
            collected = root / case / 'collected'
            selected = json.loads((collected / f'{case}_s{seed}_summary.json').read_text())
            weights = collected / selected['checkpoint']
            if sha(weights) != selected['checkpoint_sha256']:
                raise ValueError('Checkpoint changed')
            blob = torch.load(weights, map_location='cpu', weights_only=False)
            config = blob['config']
            fingerprint = train.code_fingerprint(**{k: config[k] for k in
                inspect.signature(train.code_fingerprint).parameters if k in config})
            if fingerprint != config['source_code_sha256']:
                raise ValueError('Run under original frozen training sources')
            net, _, _ = train.initialize_attack_policy(blob, device='cpu', num_planes=config['num_planes'],
                num_scalars=config['num_scalars'], observation_contract=config['observation_contract'])
            net.eval()
            opts = dict(options(config), entity_inputs=getattr(net, 'entity_inputs', False))
            env = train.make_training_env(task_mode='joint', split='train', seed=20261110, **opts)
            # Twelve deterministic TRAIN contexts, one for each task/family.
            scenarios = sorted(getattr(s, 'scenario_id', s) for s in env.scenarios
                               if getattr(s, 'scenario_id', s).endswith('/push'))
            if len(scenarios) != 12:
                raise ValueError('Expected twelve TRAIN task/family contexts')
            modes = {}
            for mode in ('flat', 'grouped'):
                records, decisions = [], []
                for index, scenario in enumerate(scenarios):
                    obs, info = env.reset(seed=20261110 + index, scenario_id=scenario)
                    total = 0.
                    with torch.inference_mode():
                        for step in range(1242):
                            logits, _ = net(*train.tensor_observations([obs], 'cpu'))
                            mask = train.action_masks([info], 'cpu')[0]
                            legal = logits[0].masked_fill(~mask, -torch.inf)
                            flat, grouped = int(legal.argmax()), grouped_action(logits[0], mask)
                            if int(mask.sum()) > 1:
                                probs = legal.softmax(0)
                                decisions.append(dict(scenario=scenario, step=step, flat=flat, grouped=grouped,
                                    wait_probability=float(probs[0]),
                                    slot_mass=probs[1:2305].reshape(4, 576).sum(-1).tolist(),
                                    max_cell_probability=float(probs[1:2305].max())))
                            action = flat if mode == 'flat' else grouped
                            obs, reward, terminal, truncated, info = env.step(action)
                            total += reward
                            if terminal or truncated:
                                records.append(dict(info['metrics'], episode_return=total))
                                break
                        else:
                            raise ValueError('Episode did not terminate')
                modes[mode] = dict(records=records, decisions=decisions,
                    wins=sum(r['outcome'] == 'win' for r in records), games=len(records),
                    mean_overflow=float(np.mean([r['elixir_overflow'] for r in records])),
                    mean_empty_spell_elixir=float(np.mean([r['empty_spell_elixir'] for r in records])),
                    choice_steps=len(decisions),
                    flat_wait_group_plays=sum(r['flat'] == 0 and r['grouped'] != 0 for r in decisions),
                    low_probability_flat_wait=sum(r['flat'] == 0 and r['wait_probability'] < .1 for r in decisions))
                print(json.dumps(dict(case=case, seed=seed, mode=mode,
                    **{k: v for k, v in modes[mode].items() if k not in ('records', 'decisions')})), flush=True)
                (output / f'{case}_s{seed}_{mode}.json').write_text(json.dumps(modes[mode], indent=2) + '\n')
            models.append(dict(case=case, seed=seed, checkpoint=str(weights), checkpoint_sha256=sha(weights),
                update=blob['update'], modes={m: {k:v for k,v in r.items() if k not in ('records','decisions')}
                                             for m,r in modes.items()}))
    result = dict(schema='entity-decoding-train-v1', source_code_sha256=fingerprint,
        driver_sha256=sha(__file__), split='train', seed=20261110, models=models,
        caveat='Twelve shared TRAIN contexts, selected weights only. Not held-out quality or live wins. No retraining or promotion.')
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.root, args.output)
