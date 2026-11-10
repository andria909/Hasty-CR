"""Technical full-match execution of CPU preflight weights, never model selection."""
import argparse
import json
from pathlib import Path

import torch

from scripts.attack import evaluate_continuous_pool as pool
from scripts.attack.evaluate_full_candidates import test_game, public_duel_metrics
from scripts.attack.evaluation_execution import execution_for, check_row_execution, check_duel_execution
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck


def run(study, blueprint, output):
    torch.set_num_threads(1)
    study, blueprint, output = map(Path, (study, blueprint, output))
    spec = json.loads(blueprint.read_text()); execution = execution_for(spec)
    seeds = [2026111001, 2026111002, 2026111003]
    heldout = {s for test in spec['test_sets'] for s in test['seeds']} | set(spec['tournament']['seeds'])
    if heldout.intersection(seeds):
        raise ValueError('Technical seeds overlap precommitted quality evaluation')
    proof = json.loads((study / 'preflight.json').read_text())
    entries = []
    for row in proof['proof']:
        path = study / 'preflight' / (row['model'] + '_whole') / 'update_000002.pt'
        if pool.sha(path) != row['checkpoint_sha256'] or row['steps'] != 32:
            raise ValueError('Expected exact technical preflight checkpoint')
        entries.append(dict(id=row['model'], file=str(path.resolve()), sha256=pool.sha(path),
            scalars=309, mode=pool.JOINT_MODE, evaluation_action_mode='grouped_argmax'))
    baseline = next(m for m in spec['baselines'] if m['id'] == 'v25')
    baseline = dict(baseline, file=str(Path(baseline['checkpoint']).resolve()),
                    sha256=baseline['checkpoint_sha256'])
    all_entries = entries + [baseline]
    nets = {m['id']: pool.load(m, '/', 'cpu') for m in all_entries}
    output.mkdir(parents=True, exist_ok=False)
    paths = list(Path('sim').glob('*.py')) + [Path(__file__),
        Path('scripts/attack/evaluate_continuous_pool.py'),
        Path('scripts/attack/evaluate_full_candidates.py'), Path('scripts/attack/evaluation_execution.py')]
    sources = {str(p): pool.sha(p) for p in paths}
    cards, spells, *_ = load_defense_assets(); cards.update(resolve_deck(cards, list(DECK_26)))
    rows, legacy_checks = [], 0
    original = pool.choose

    def audited_choose(network, match, side, mode, mask, device, reserved_milli=0):
        nonlocal legacy_checks
        action = original(network, match, side, mode, mask, device, reserved_milli)
        if network is nets['v25'] and mask[1:].any():
            obs = pool.policy_observation(match, side, mode)
            obs['scalars'] = obs['scalars'].copy()
            obs['scalars'][0] = max(0, obs['scalars'][0] - reserved_milli / 10000)
            with torch.inference_mode():
                logits, _ = network(torch.as_tensor(obs['planes'])[None], torch.as_tensor(obs['scalars'])[None])
                old_action = int(logits[0].masked_fill(~torch.as_tensor(mask), -torch.inf).argmax())
            if action != old_action:
                raise ValueError('Legacy v25 deterministic action changed')
            legacy_checks += 1
        return action

    def save(kind, model, row):
        rows.append(dict(kind=kind, model=model, row=row))
        (output / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(dict(completed=len(rows), target=12, kind=kind, model=model)), flush=True)

    pool.choose = audited_choose
    try:
        for entry in entries:
            net = nets[entry['id']]
            for test, seed in zip(spec['test_sets'], seeds):
                row = test_game(net, entry, test, 'defense', 'air', seed, 'cpu', execution=execution)
                check_row_execution(row, execution)
                save('technical_test_path', entry['id'], row)
            for seat in (1, -1):
                row = pool.duel((net, entry['mode']), (nets['v25'], baseline['mode']), 'cpu', seeds[-1],
                    seat, cards, spells, execution=execution, on_complete=public_duel_metrics)
                check_duel_execution(row, execution)
                if not row['executed_commands'] or any(row['rejected_commands'].values()):
                    raise ValueError('Expected real accepted commands without rejection')
                save('technical_duel', entry['id'], row)
    finally:
        pool.choose = original
    if len(rows) != 12 or not legacy_checks or any(pool.sha(p) != digest for p, digest in sources.items()):
        raise ValueError('Technical budget or sources changed')
    for entry in all_entries:
        if pool.sha(entry['file']) != entry['sha256']:
            raise ValueError('Weights changed')
    result = dict(schema='entity-full-evaluation-readiness-v1', execution=execution,
        technical_games=len(rows), development_seeds=seeds, blueprint_sha256=pool.sha(blueprint),
        preflight_sha256=pool.sha(study / 'preflight.json'), models=all_entries,
        rows_path=str(output / 'rows.json'), rows_sha256=pool.sha(output / 'rows.json'),
        legacy_v25_decisions_exact=legacy_checks, sources=sources,
        independent_quality_test_run=False, quality_accepted=False, live_games_started=False,
        caveat='32-step CPU technical weights. Outcomes are not candidate win rates. Held-out quality seeds unused.')
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', required=True); p.add_argument('--blueprint', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.study, a.blueprint, a.output)
