"""Convert audited safe timing plans to demonstrations of a declared heuristic.

The teacher minimizes paid defensive cost, then delays the request within tested
safe windows. WAIT is demonstrated as part of a successful continuation, never
claimed to dominate every immediate action or to be a full-match optimum.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
from scripts.attack.build_bank_counterfactuals import opening
from scripts.attack.build_late_air_sequences import replay_match
from scripts.attack.sequence_physics import physics_pins
from scripts.attack.sequence_execution import EXECUTION
from scripts.kaggle.local_artifact_transfer import file_sha


def choose(row):
    if not row.get('cannon_first_audited'):
        raise ValueError('Missing Cannon-first audit')
    safe = [r for r in row['trials'] if r['valid'] and r['clear_without_damage']
            and r['commands'] and r['delay_ms'] is not None]
    if not safe:
        return None
    # Prefer cheap defense first; a more expensive late defense is not a win.
    return min(safe, key=lambda r: (r['elixir_spent'], -r['delay_ms'],
                                    json.dumps(r['candidate'], sort_keys=True)))


def replay(row, plan):
    env = opening(row['seed'], row['lane'], row['bank'], row['enemy_y'])
    hp = {k: t.hitpoints for k, t in env.match.towers[1].items()}
    states = replay_match(env.match, hp, plan, env.cards, env.spells)
    if any(r['pending_count'] for r in states):
        raise ValueError('Single-request bank teacher must not hide pending actions')
    if sum(r['action'] != 0 for r in states) != 1:
        raise ValueError('Exactly one final defensive request required')
    return states


def verify_search(root):
    root = Path(root)
    prep = json.loads((root/'preparation.json').read_text())
    receipt = json.loads((root/'collected/collection_receipt.json').read_text())
    result = json.loads((root/'collected/result.json').read_text())
    if not (prep.get('timing') and receipt.get('complete') and
            receipt['kernel'] == prep['kernel'] and result['schema'] == 'bank-timing-search-v1'
            and result['contexts'] == 36 and not result['smoke']):
        raise ValueError('Complete collected timing search required')
    raw = root/'collected/contexts.jsonl'
    for name in ('contexts.jsonl', 'result.json'):
        if file_sha(root/'collected'/name) != receipt['files']['bank_cannon_search/'+name]['sha256']:
            raise ValueError('Collected timing receipt differs')
    if file_sha(raw) != result['raw_sha256']:
        raise ValueError('Timing contexts changed')
    if result['source_sha256'] != file_sha(root/'expanded_probe.py'):
        raise ValueError('Timing probe differs from executed source')
    if file_sha(root/'job/train.py') != prep['bootstrap_sha256']:
        raise ValueError('Timing bootstrap changed')
    config = ast.literal_eval(ast.parse((root/'job/train.py').read_text()).body[-1].value.args[0])
    if file_sha('scripts/attack/build_bank_counterfactuals.py') != config['pins']['bank_builder.py']:
        raise ValueError('Opening builder differs from searched fixture')
    for name, digest in config['extensions'].items():
        # The search imports the match/observation implementation, never the
        # PPO optimizer/CLI. Reward-factory integration can change separately.
        if name.startswith('sim/') and name != 'sim/train_attack_ppo.py' and file_sha(name) != digest:
            raise ValueError('Timing simulator differs: '+name)
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    identities = {(r['seed'], r['lane'], r['bank'], r['enemy_y']) for r in rows}
    expected = {(seed, lane, bank, y) for seed in (352, 353, 354)
                for lane in ('left', 'right') for bank in (2000, 3000, 6000) for y in (9, 15)}
    if len(rows) != 36 or identities != expected:
        raise ValueError('Missing, repeated or unexpected contexts')
    return rows, prep


def build(root, output):
    rows, prep = verify_search(root)
    arrays = {k: [] for k in ('planes', 'scalars', 'masks', 'success_masks')}
    records, contexts = [], []
    for index, row in enumerate(rows):
        plan = choose(row)
        contexts.append(dict(context=index, seed=row['seed'], lane=row['lane'], bank=row['bank'],
                             enemy_y=row['enemy_y'], accepted=plan is not None, plan=plan))
        if plan is None:
            continue
        for state in replay(row, plan):
            i = len(records)
            for key in arrays:
                arrays[key].append(state[key])
            records.append(dict(seed=row['seed'], context=index, accepted=True, array_index=i,
                                skill='safe_bank_sequence', lane=row['lane'],
                                **{k: state[k] for k in ('decision_ms', 'action', 'pending_count', 'reserved_elixir')}))
    if {r['seed'] for r in records} != {352, 353, 354}:
        raise ValueError('Three represented whole-seed groups required')
    # Identical inputs across heldout groups would inflate local accuracy.
    seen = {}
    import hashlib
    for r in records:
        i = r['array_index']
        digest = hashlib.sha256(b''.join(arrays[k][i].tobytes()
                               for k in ('planes', 'scalars', 'masks'))).hexdigest()
        if digest in seen and seen[digest] != r['seed']:
            raise ValueError('Identical inputs cross seed groups')
        seen[digest] = r['seed']; r['input_sha256'] = digest
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'examples.npz', **{k: np.stack(v) for k, v in arrays.items()})
    manifest = dict(schema='bank-safe-sequence-labels-v1', split='train', records=records,
                    contexts=contexts, examples=len(records), seeds=[352, 353, 354],
                    command_delay_ms=1350, execution=EXECUTION, physics_pins=physics_pins(),
                    cannon_first_audited=True, source_kernel=prep['kernel'],
                    search_sha256=file_sha(Path(root)/'collected/contexts.jsonl'),
                    builder_sha256=file_sha(__file__), data_sha256=file_sha(output/'examples.npz'),
                    teacher_order=['zero additional tower damage and all threats cleared',
                                   'minimum paid elixir', 'latest tested safe request'],
                    quality_accepted=False, full_game_training_ready=False,
                    limitations=['Only single paid Balloon; fixed hand; no second push',
                                 'WAIT is a successful continuation, not a counterfactual superiority label',
                                 'Slot diversity, urgent controls and mixed skills required before PPO study',
                                 'Local whole-seed holdout is not a game winrate'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--search-root', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); m = build(a.search_root, a.output)
    print(json.dumps(dict(examples=m['examples'], sequences=sum(r['accepted'] for r in m['contexts']))))
