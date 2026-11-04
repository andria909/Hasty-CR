"""Short WAIT interventions in already verified TRAIN snapshots.

The saved baseline is reused only after checking its source, physics, weights
and snapshot hashes. One initial pause is followed by the unchanged actor.
This estimates the effect of that pause, not an optimal saving policy.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
from statistics import mean

import torch

from scripts.attack.diagnose_cheap_cycle import decision
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.continuous_push import CONTRACT
from sim.train_attack_ppo import initialize_attack_policy

WAITS_MS = (250, 1000, 2000)


def continue_with_wait(payload, net, wait_ms):
    env = pickle.loads(payload)
    start = env.match.elapsed_ms
    before = env._metrics()
    commands = len(env.actions)
    waits = 0
    first_action = None
    while not env.done:
        waiting = env.match.elapsed_ms - start < wait_ms
        action = 0 if waiting else decision(net, env)
        waits += int(waiting)
        if action and first_action is None:
            first_action = dict(at_ms=env.match.elapsed_ms,
                                card=env.match.players[1].hand[(action - 1)//576], action=action)
        env.step(action)
    after = env._metrics()
    if after['illegal_actions'] != before['illegal_actions']:
        raise ValueError('Intervention produced an illegal action')
    return dict(wait_ms=wait_ms, forced_wait_steps=waits, terminal=True, metrics=after,
                first_action_after_wait=first_action,
                own_loss_delta=after['own_fraction_loss'] - before['own_fraction_loss'],
                enemy_damage_delta=after['enemy_fraction_damage'] - before['enemy_fraction_damage'],
                accepted_cards=dict(Counter(c['card'] for c in env.actions[commands:])),
                spent_delta=after['elixir_spent'] - before['elixir_spent'], illegal_delta=0)


def summarize(records):
    groups = []
    for wait in (0, *WAITS_MS):
        pairs = [(r['baseline'], r['baseline'] if wait == 0 else
                  next(b for b in r['branches'] if b['wait_ms'] == wait)) for r in records]
        gained = sum(a['metrics']['outcome'] != 'win' and b['metrics']['outcome'] == 'win' for a, b in pairs)
        lost = sum(a['metrics']['outcome'] == 'win' and b['metrics']['outcome'] != 'win' for a, b in pairs)
        groups.append(dict(wait_ms=wait, games=len(pairs),
                           wins=sum(b['metrics']['outcome'] == 'win' for _, b in pairs),
                           gained_wins=gained, lost_wins=lost,
                           mean_own_loss_delta=mean(b['own_loss_delta'] for _, b in pairs),
                           mean_enemy_damage_delta=mean(b['enemy_damage_delta'] for _, b in pairs),
                           mean_spent_delta=mean(b['spent_delta'] for _, b in pairs),
                           mean_musketeer_plays=mean(b['accepted_cards'].get('musketeer', 0) for _, b in pairs)))
    return groups


def run(source, output):
    torch.set_num_threads(1)
    source, output = Path(source), Path(output)
    baseline_path = source/'result.json'
    baseline_sha = file_sha(baseline_path)
    baseline = json.loads(baseline_path.read_text())
    original_source = Path(__file__).with_name('diagnose_cheap_cycle.py')
    if (baseline.get('complete') is not True or len(baseline['records']) != 12 or
            baseline['protocol']['physics_pins'] != physics_pins() or
            baseline['protocol']['source_sha256'] != file_sha(original_source) or
            file_sha(source/'driver.py') != file_sha(original_source)):
        raise ValueError('Complete same-source/physics baseline required')
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema='short-wait-train-intervention-v1', source=str(baseline_path),
                    source_sha256=baseline_sha, driver_sha256=file_sha(__file__),
                    physics_pins=physics_pins(), waits_ms=WAITS_MS, live_games=False,
                    scope='Same twelve selected TRAIN states; one short initial pause, then unchanged greedy actor',
                    independent_test=False, training_modified=False, policy_promoted=False)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (output/'driver.py').write_bytes(Path(__file__).read_bytes())
    records, networks = [], {}
    for row in baseline['records']:
        weights = Path(row['weights'])
        snapshot = source/row['snapshot']
        if (not snapshot.resolve().is_relative_to(source.resolve()) or
                file_sha(snapshot) != row['snapshot_sha256'] or
                file_sha(weights) != row['checkpoint_sha256'] or
                file_sha(weights.parent/'config.json') != row['config_sha256']):
            raise ValueError('Frozen weights/config/snapshot changed')
        control = next(b for b in row['branches'] if b['mode'] == 'baseline')
        if control['metrics'] != row['original_metrics'] or control['illegal_delta']:
            raise ValueError('Baseline was not an exact legal terminal replay')
        if str(weights) not in networks:
            net, _, _ = initialize_attack_policy(torch.load(weights, map_location='cpu', weights_only=False),
                device='cpu', num_planes=20, num_scalars=309, observation_contract=CONTRACT)
            net.eval()
            networks[str(weights)] = net
        net = networks[str(weights)]
        payload = snapshot.read_bytes()
        env = pickle.loads(payload)
        if env.match.elapsed_ms != row['state']['at_ms'] or decision(net, env) != row['state']['action']:
            raise ValueError('Snapshot decision or time changed')
        result = {k: row[k] for k in ('replica','scenario','episode_seed','state','checkpoint_sha256','snapshot_sha256')}
        result.update(baseline=control, branches=[continue_with_wait(payload, net, wait) for wait in WAITS_MS])
        records.append(result)
        (output/f"s{row['replica']}_{row['scenario'].split('::')[1].split('/')[0]}.json").write_text(
            json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(completed=len(records), total=12, scenario=row['scenario'], replica=row['replica'],
            outcomes=[b['metrics']['outcome'] for b in result['branches']])), flush=True)
    if file_sha(baseline_path) != baseline_sha or physics_pins() != protocol['physics_pins']:
        raise ValueError('Inputs changed during diagnosis')
    result = dict(protocol=protocol, complete=True, new_full_continuations=36,
                  reused_baseline_games=12, groups=summarize(records), records=records)
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.source, args.output)
