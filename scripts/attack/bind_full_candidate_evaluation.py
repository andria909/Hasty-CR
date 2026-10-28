"""Freeze completed full-budget weights before independent test inference.

Re-audit training artifacts, retain both models in each supplied full pair,
and never filter on final family guard or test results. Does not submit a job.
"""
import argparse
import json
from pathlib import Path
import shutil

from scripts.attack.evaluate_full_candidates import schedule, sha, validate_protocol
from scripts.attack.verify_global_full_pair import verify as verify_full

BLUEPRINT = Path('data/ppo_tuning/normal_range_evaluation_20261028/protocol.json')


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def bind(prepared_roots, output, *, blueprint_path=BLUEPRINT):
    roots = [Path(p).resolve() for p in prepared_roots]
    output = Path(output)
    if len(roots) not in (1, 2) or len(set(roots)) != len(roots):
        raise ValueError('One or two distinct completed full pairs required')
    if output.exists():
        raise ValueError('Evaluation binding output already exists; preserve it')
    blueprint_path = Path(blueprint_path)
    blueprint_digest = sha(blueprint_path)
    blueprint = json.loads(blueprint_path.read_text())
    if blueprint.get('schema') not in ('precommitted-full-candidate-evaluation-v1','precommitted-full-candidate-evaluation-v2','precommitted-full-candidate-evaluation-v3'):
        raise ValueError('Precommitted evaluation blueprint required')
    # Complete every expensive audit before creating the package. An unfinished
    # run or bad digest must not leave a seemingly usable bound protocol.
    pairs = []
    for root in roots:
        proof = verify_full(root / 'collected', root)
        if (proof.get('schema') != 'verified-global-selected-full-pair-v1'
                or proof.get('full_steps_verified') != 8192000 or len(proof['models']) != 2
                or proof.get('quality_accepted') is not False
                or proof.get('test_used_for_selection') is not False):
            raise ValueError('Successful full-pair provenance required')
        pairs.append((root, proof))
    pairs.sort(key=lambda item: item[1]['kernel'])
    if len({p['kernel'] for _, p in pairs}) != len(pairs):
        raise ValueError('Duplicate training kernel')
    seen = set()
    for _, proof in pairs:
        for model in proof['models']:
            if blueprint['schema'].endswith('-v3'):
                from scripts.attack.evaluation_execution import execution_for,require_training_execution
                require_training_execution(model,execution_for(blueprint))
            identity = (proof['kernel'], model['id'])
            if identity in seen or sha(Path(model['checkpoint'])) != model['checkpoint_sha256']:
                raise ValueError('Duplicate or changed candidate checkpoint')
            if not isinstance(model['family_guard'], bool):
                raise ValueError('Candidate family guard must be explicit')
            seen.add(identity)
    for model in blueprint['baselines']:
        if sha(Path(model['checkpoint'])) != model['checkpoint_sha256']:
            raise ValueError('Precommitted baseline changed')
    output.mkdir(parents=True, exist_ok=False)
    weights = output / 'weights'; weights.mkdir()
    shutil.copyfile(blueprint_path, weights / 'evaluation_blueprint.json')
    if sha(weights / 'evaluation_blueprint.json') != blueprint_digest:
        raise ValueError('Blueprint changed during binding')
    models = []
    for index, (root, proof) in enumerate(pairs):
        proof_path = weights / f'full_pair_{index}_verified.json'
        write_json(proof_path, proof)
        for ordinal, source in enumerate(proof['models']):
            name = f'pair{index}_candidate{ordinal}'
            filename = name + '.pt'
            shutil.copyfile(source['checkpoint'], weights / filename)
            models.append(dict(id=name, file=filename, role='candidate', sha256=source['checkpoint_sha256'],
                full_steps=4096000, mode='joint_enemy_presence_v1', scalars=309,
                full_proof_file=proof_path.name, full_proof_sha256=sha(proof_path),
                origin_model_id=source['id'], origin_kernel=proof['kernel'], origin_prepared=str(root),
                family_guard=source['family_guard'], validation_selection_score=source['selection_score']))
            if 'training_execution' in source:models[-1]['training_execution']=source['training_execution']
    for source in blueprint['baselines']:
        filename = source['id'] + '.pt'
        shutil.copyfile(source['checkpoint'], weights / filename)
        models.append(dict(id=source['id'], file=filename, role='baseline', sha256=source['checkpoint_sha256'],
                           mode=source['mode'], scalars=source['scalars']))
    protocol = dict(schema='bound-full-candidate-evaluation-v1', blueprint_sha256=blueprint_digest,
                    models=models, test_used_for_selection=False, quality_accepted=False,
                    live_checkpoint_changed=False)
    if blueprint['schema'].endswith(('-v2','-v3')):
        from scripts.attack.evaluation_execution import execution_for
        protocol.update(schema='bound-full-candidate-evaluation-'+blueprint['schema'].rsplit('-',1)[1],execution=execution_for(blueprint))
    validated = validate_protocol(protocol, weights)
    jobs = schedule(protocol, validated)
    # The protocol is the final commit marker. Failed copies cannot be submitted
    # as a completed binding; no existing files are removed or overwritten.
    write_json(output / 'schedule.json', jobs)
    write_json(output / 'protocol.json', protocol)
    return dict(protocol=str(output / 'protocol.json'), weights=str(weights),
                candidates=sum(m['role'] == 'candidate' for m in models),
                failed_family_guard=[m['id'] for m in models if m.get('family_guard') is False],
                total_games=sum(j['games'] for j in jobs), submitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--blueprint',default=str(BLUEPRINT))
    args = parser.parse_args()
    print(json.dumps(bind(args.prepared, args.output,blueprint_path=args.blueprint), indent=2))
