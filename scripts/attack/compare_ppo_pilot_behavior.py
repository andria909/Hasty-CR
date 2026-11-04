"""Descriptive paired comparisons; no automatic promotion or test-set claims."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from scripts.kaggle.local_artifact_transfer import file_sha

IDENTITY = ('scenario_id', 'episode_seed', 'match_seed', 'task', 'family_id')
FAMILIES = ('air', 'bridge', 'giant', 'golem', 'hog', 'swarm')
METRICS = ('own_fraction_loss', 'empty_spell_elixir', 'elixir_overflow',
           'fireball_unit_hits', 'cluster_fireballs', 'own_tower_damage')


def index_records(records):
    if len(records) != 72:
        raise ValueError('Exactly 72 paired validation contexts required')
    indexed = {}
    for row in records:
        key = tuple(row[k] for k in IDENTITY)
        if key in indexed or row['split'] != 'validation' or row['outcome'] not in ('win', 'draw', 'loss'):
            raise ValueError('Duplicate context, wrong split or unknown outcome')
        if any(type(row[k]) is not int for k in ('episode_seed', 'match_seed')):
            raise ValueError('Explicit context seeds required')
        if not math.isfinite(row['elapsed_ms']) or row['elapsed_ms'] <= 0:
            raise ValueError('Positive match duration required')
        if any(type(v) is not int or v < 0 for v in row['accepted_card_counts'].values()):
            raise ValueError('Actual card counts required')
        for metric in METRICS:
            if not isinstance(row[metric], (int, float)) or not math.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError('Invalid behavioral measurement')
        indexed[key] = row
    expected = {(family, task): 6 for family in FAMILIES for task in ('attack', 'defense')}
    if Counter((r['family_id'], r['task']) for r in records) != expected:
        raise ValueError('Balanced family/task coverage required')
    return indexed


def compare_records(before, after):
    before, after = index_records(before), index_records(after)
    if before.keys() != after.keys():
        raise ValueError('Compared policies did not play identical contexts')
    pairs = [(before[key], after[key]) for key in sorted(before)]
    def summarize(rows):
        n = len(rows)
        gained = sum(a['outcome'] != 'win' and b['outcome'] == 'win' for a, b in rows)
        lost = sum(a['outcome'] == 'win' and b['outcome'] != 'win' for a, b in rows)
        result = dict(games=n, before_wins=sum(a['outcome'] == 'win' for a, b in rows),
                      after_wins=sum(b['outcome'] == 'win' for a, b in rows),
                      gained_wins=gained, lost_wins=lost, wins_delta=gained-lost,
                      metrics={metric:dict(before=sum(a[metric] for a,b in rows)/n,
                                          after=sum(b[metric] for a,b in rows)/n,
                                          delta=sum(b[metric]-a[metric] for a,b in rows)/n)
                               for metric in METRICS})
        def rates(side):
            records = [pair[side] for pair in rows]
            minutes = sum(r['elapsed_ms'] for r in records)/60000
            return dict(duration_seconds=minutes*60/n,
                        fireball_per_game=sum(r['accepted_card_counts'].get('fireball',0) for r in records)/n,
                        unit_hits_per_minute=sum(r['fireball_unit_hits'] for r in records)/minutes,
                        empty_spell_elixir_per_minute=sum(r['empty_spell_elixir'] for r in records)/minutes)
        first, last = rates(0), rates(1)
        result['metrics'].update({k:dict(before=first[k],after=last[k],delta=last[k]-first[k]) for k in first})
        return result
    return dict(all=summarize(pairs),
                families={f:summarize([(a,b) for a,b in pairs if a['family_id'] == f]) for f in FAMILIES},
                tasks={t:summarize([(a,b) for a,b in pairs if a['task'] == t]) for t in ('attack', 'defense')})


def compare_study(root, proof_path):
    root, proof_path = Path(root), Path(proof_path)
    proof = json.loads(proof_path.read_text())
    prep = json.loads((root/'preparation.json').read_text())
    protocol = json.loads((root/'dataset/protocol.json').read_text())
    variant = protocol['study_variant']
    if variant == 'credit_assignment_v1':
        from scripts.kaggle.prepare_credit_assignment_pilot import validate_protocol
        accepted = proof.get('credit_assignment_audit',{}).get('verified') is True
    elif variant == 'tower_credit_v1':
        from scripts.kaggle.prepare_tower_credit_pilot import validate_protocol
        accepted = proof.get('tower_credit_audit',{}).get('verified') is True
    elif variant == 'spell_interaction_v1':
        from scripts.kaggle.prepare_spell_interaction_pilot import validate_protocol
        accepted = proof.get('spell_interaction_audit',{}).get('verified') is True
    elif variant == 'air_exposure_v1':
        from scripts.kaggle.prepare_air_exposure_pilot import validate_protocol
        accepted = proof.get('air_exposure_audit',{}).get('verified') is True
    elif variant == 'group_entropy_v1':
        from scripts.kaggle.prepare_group_entropy_pilot import validate_protocol
        accepted = proof.get('group_entropy_audit',{}).get('verified') is True
    elif variant == 'calibrated_spell_cost_v1':
        from scripts.kaggle.prepare_calibrated_spell_pilot import validate_protocol
        accepted = proof.get('calibrated_spell_cost_verified') is True
    elif variant == 'policy_adaptation_v1':
        from scripts.kaggle.prepare_policy_adaptation_pilot import validate_protocol
        accepted = proof.get('policy_adaptation_audit', {}).get('verified') is True
    else:
        raise ValueError('Unsupported paired PPO study')
    validate_protocol(protocol)
    if not accepted or proof['pilot_steps_verified'] != (3276800 if variant in ('air_exposure_v1','spell_interaction_v1','tower_credit_v1','credit_assignment_v1')else 819200) or proof['kernel'] != prep['kernel']:
        raise ValueError('Completed independent CUDA verification required')
    collected = root/'collected'
    receipt = json.loads((collected/'collection_receipt.json').read_text())
    if not receipt['complete'] or receipt['kernel'] != prep['kernel']:
        raise ValueError('Complete matching collection required')
    def read(name):
        if file_sha(collected/name) != receipt['files'][name]['sha256']:
            raise ValueError('Collected measurements changed')
        return json.loads((collected/name).read_text())
    models = {m['id']:m for m in proof['pilot_models']}
    expected = {f"pilot_{case['id']}_s{seed}" for case in protocol['cases'] for seed in protocol['pilot_seeds']}
    if set(models) != expected:
        raise ValueError('All eight verified policies required')
    summaries = {}
    for name, model in models.items():
        summary = read(name+'_summary.json')
        if (summary['id'] != name or file_sha(collected/summary['checkpoint']) != model['checkpoint_sha256']
                or file_sha(model['checkpoint']) != model['checkpoint_sha256']):
            raise ValueError('Selected checkpoint differs from verified evidence')
        summaries[name] = summary
    rows = []
    for case in protocol['cases']:
        for seed in protocol['pilot_seeds']:
            name = f"pilot_{case['id']}_s{seed}"
            if variant in ('air_exposure_v1','spell_interaction_v1','tower_credit_v1','credit_assignment_v1'):
                control = protocol['paired_controls'].get(case['id'])
                reference = f'pilot_{control}_s{seed}' if control else case['initial_id']
                baseline = summaries[reference] if control else read(reference+'_reference.json')
            elif variant in ('calibrated_spell_cost_v1','group_entropy_v1'):
                reference = f'pilot_control_s{seed}'
                baseline = summaries[reference]
            else:
                reference = case['initial_id']
                baseline = read(reference+'_reference.json')
            rows.append(dict(id=name, case=case['id'], seed=seed, reference=reference,
                             selected_update=summaries[name]['selected_update'],
                             comparison=compare_records(baseline['validation']['records'],
                                                        summaries[name]['validation']['records'])))
    groups = []
    for case in protocol['cases']:
        records = [r for r in rows if r['case'] == case['id']]
        deltas = [r['comparison']['all']['wins_delta'] for r in records]
        groups.append(dict(case=case['id'], wins_delta_by_seed=deltas,
                           mixed_seed_direction=min(deltas) < 0 < max(deltas),
                           wins_non_decreasing_both=all(x >= 0 for x in deltas),
                           sum_wins_delta=sum(deltas)))
    return dict(schema='paired-ppo-behavior-comparison-v1', kernel=proof['kernel'],
                proof_sha256=file_sha(proof_path), collection_sha256=file_sha(collected/'collection_receipt.json'),
                protocol_sha256=file_sha(root/'dataset/protocol.json'), rows=rows, groups=groups,
                quality_accepted=False, automatic_full_training=False,
                scope='Descriptive paired development validation; selected checkpoints reuse these contexts; no independent significance claim',
                unique_contexts=72, independent_training_seeds=len(protocol['pilot_seeds']))


def render(result):
    lines = ['# Парное сравнение PPO', '',
             'Одни и те же 72 сценария. Каждый seed показан отдельно. Контроль указан в JSON: '
             'парный PPO-профиль с тем же seed или соответствующие исходные веса. '
             'Это использованный при выборе чекпойнтов validation, не независимый test.', '',
             '| Вариант | Seed | Контроль → кандидат | Новые / потерянные победы | Δ HP потерь | Δ пустого эликсира |',
             '|---|---:|---:|---:|---:|---:|']
    for row in result['rows']:
        c = row['comparison']['all']; m = c['metrics']
        lines.append(f"| {row['case']} | {row['seed']} | {c['before_wins']} → {c['after_wins']} / 72 | "
                     f"+{c['gained_wins']} / −{c['lost_wins']} | {m['own_fraction_loss']['delta']:+.3f} | "
                     f"{m['empty_spell_elixir']['delta']:+.2f} |")
    lines += ['', 'Отрицательные изменения потерь HP и пустого расхода лучше. HP — сумма долей трёх своих башен (0–3). '
              'Приобретённая победа — сценарий, который контроль не выиграл, а кандидат выиграл; потерянная — наоборот.', '',
              '| Вариант | Seed | Δ побед в атаке | Δ побед в защите | Δ попаданий Fireball по юнитам / матч |',
              '|---|---:|---:|---:|---:|']
    for row in result['rows']:
        c = row['comparison']
        lines.append(f"| {row['case']} | {row['seed']} | {c['tasks']['attack']['wins_delta']:+d} | "
                     f"{c['tasks']['defense']['wins_delta']:+d} | {c['all']['metrics']['fireball_unit_hits']['delta']:+.2f} |")
    lines += ['', 'В JSON также сохранены все шесть семейств, урон своих башен, перелив и попадания по скоплениям. '
              'Снижение пустого расхода без сохранения побед и защиты не служит основанием для большого обучения. '
              'Разные знаки улучшения по seed явно помечаются; они не доказывают устойчивый выигрыш.', '']
    lines += ['| Вариант | Seed | Fireball за матч: до → после | Целей Fireball / минуту: до → после | Длительность, с: до → после |',
              '|---|---:|---:|---:|---:|']
    for row in result['rows']:
        m = row['comparison']['all']['metrics']
        values = [f"{m[k]['before']:.2f} → {m[k]['after']:.2f}" for k in
                  ('fireball_per_game','unit_hits_per_minute','duration_seconds')]
        lines.append(f"| {row['case']} | {row['seed']} | {' | '.join(values)} |")
    lines += ['', 'Нормировка по минутам учитывает разную длительность матчей; число попаданий не равно полезности каждого из них.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('root', 'proof', 'output', 'markdown'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    result = compare_study(args.root, args.proof)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    Path(args.markdown).write_text(render(result))
