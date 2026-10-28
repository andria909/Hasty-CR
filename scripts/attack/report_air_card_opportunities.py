"""Join pinned TRAIN counterfactuals with actual learned-policy episodes."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.attack.search_late_air_defense import rank
from scripts.kaggle.local_artifact_transfer import file_sha


def key(record):
    return tuple(sorted(record.items()))


def report(root, policies, output):
    root, output = Path(root), Path(output)
    result = json.loads((root/'result.json').read_text())
    if file_sha(root/'trials.jsonl') != result['trials_sha256']:
        raise ValueError('Counterfactual trials changed')
    trials = [json.loads(line) for line in (root/'trials.jsonl').read_text().splitlines()]
    if len(trials) != result['trials']:
        raise ValueError('Trial count mismatch')
    for group in result['groups']:
        rows = [r for r in trials if r['context'] == group['context']]
        if len(rows) != group['attempted'] or sum(not r['valid'] for r in rows) != group['invalid']:
            raise ValueError('Context accounting mismatch')
        for card in ('cannon', 'musketeer'):
            legal = [r for r in rows if r['valid'] and r['first'] and r['first']['card'] == card]
            if (min(legal, key=rank) if legal else None) != group['best'][card]:
                raise ValueError('Best counterfactual differs from actual trials')
    groups = {key(g['opening']):g for g in result['groups'] if g['best']['musketeer']}
    models, sources = [], {}
    for policy in map(Path, policies):
        evidence = json.loads((policy/'result.json').read_text())
        if (evidence['protocol']['execution'] != result['protocol']['execution'] or
                evidence['protocol']['physics_pins'] != result['protocol']['physics_pins'] or
                file_sha(policy/'episodes.jsonl') != evidence['episodes_sha256']):
            raise ValueError('Policy execution, physics or episodes differ')
        rows = [json.loads(line) for line in (policy/'episodes.jsonl').read_text().splitlines()]
        sources[str(policy)] = dict(result_sha256=file_sha(policy/'result.json'),
            episodes_sha256=evidence['episodes_sha256'])
        for model in evidence['protocol']['models']:
            selected = [r for r in rows if r['model'] == model['id'] and key(r['opening']) in groups]
            if len(selected) != len(groups) or len({key(r['opening']) for r in selected}) != len(groups):
                raise ValueError('Missing or duplicate policy context')
            no_princess = [r for r in selected if r['opening']['own_mask'] == 0]
            cards = Counter(a['card'] for r in selected for a in r['actions'])
            models.append(dict(model=model, contexts=len(selected),
                zero_damage=sum(r['clear_without_damage'] for r in selected),
                mean_damage=sum(r['total_damage'] for r in selected)/len(selected),
                mean_elixir=sum(r['elixir_spent'] for r in selected)/len(selected),
                no_princess_contexts=len(no_princess),
                no_princess_zero_damage=sum(r['clear_without_damage'] for r in no_princess),
                actual_card_counts=dict(cards)))
    summary = dict(schema='air-card-opportunity-comparison-v1',
        search_result_sha256=file_sha(root/'result.json'), sources=sources,
        contexts=len(groups), trials=len(trials),
        musketeer_zero_damage=sum(g['best']['musketeer']['clear_without_damage'] for g in groups.values()),
        cannon_zero_damage=sum(g['best']['cannon']['clear_without_damage'] for g in groups.values()),
        models=models, groups=list(groups.values()), quality_accepted=False,
        limitations=result['limitations'])
    output.write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--policies', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    report(args.root, args.policies, args.output)
