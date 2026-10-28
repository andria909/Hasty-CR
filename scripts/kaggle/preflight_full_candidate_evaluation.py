"""Exercise the exact decoded evaluation package before enabling GPU metadata."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys


def child(spec_path):
    import torch
    from scripts.attack.evaluate_continuous_pool import load, duel, sha
    from scripts.attack.evaluate_full_candidates import validate_protocol, test_game, public_duel_metrics
    from scripts.attack.full_evaluation_recovery import source_identity
    from scripts.attack.verify_full_candidate_evaluation import check_cards
    from sim.defense_assets import load_defense_assets
    from sim.env import DECK_26
    from sim.runner import resolve_deck
    spec = json.loads(Path(spec_path).read_text()); inputs = Path(spec['inputs'])
    source_digest = source_identity(inputs / 'source/manifest.json', Path.cwd())
    protocol = json.loads((inputs / 'protocol.json').read_text())
    blueprint = validate_protocol(protocol, inputs / 'weights')
    from scripts.attack.evaluation_execution import execution_for,check_row_execution
    execution=execution_for(protocol)
    torch.set_num_threads(1)
    networks = {}; checked = []
    # These development seeds are deliberately outside both independent tests.
    seeds = (801001, 801002)
    heldout = {s for t in blueprint['test_sets'] for s in t['seeds']} | set(blueprint['tournament']['seeds'])
    if heldout.intersection(seeds) or 801003 in heldout:
        raise ValueError('Technical preflight must not use independent evaluation seeds')
    for entry in protocol['models']:
        network = load(entry, inputs / 'weights', 'cpu'); networks[entry['id']] = network
        for test, seed in zip(blueprint['test_sets'], seeds):
            row = test_game(network, entry, test, 'defense', 'air', seed, 'cpu',execution=execution)
            check_cards(row)
            if protocol['schema'].endswith(('-v2','-v3')):check_row_execution(row,execution)
        checked.append(dict(id=entry['id'], sha256=entry['sha256'], technical_games=2))
    entries = {m['id']: m for m in protocol['models']}
    cards, spells, *_ = load_defense_assets(); cards.update(resolve_deck(cards, list(DECK_26)))
    row = duel((networks['balanced'], entries['balanced']['mode']), (networks['v25'], entries['v25']['mode']),
               'cpu', 801003, -1, cards, spells, on_complete=public_duel_metrics,execution=execution)
    for metrics in row['public_metrics'].values(): check_cards(metrics)
    # Exercise the previously failing pre-serialization side-counter validation.
    from scripts.attack.verify_full_candidate_evaluation import validate_part
    from collections import Counter
    technical_blueprint=dict(blueprint,tournament=dict(blueprint['tournament'],seeds=[801003],seats=[-1]))
    technical_job=dict(kind='duel',pair=['balanced','v25'],games=1)
    validate_part(dict(kind='duel',pair=technical_job['pair'],games=1,rows=[row],counts=dict(Counter([row['outcome']]))),technical_job,protocol,technical_blueprint)
    restored_games=None
    if (inputs/'resume').exists():
        from scripts.attack.full_evaluation_recovery import identity,restore_parts
        from scripts.attack.evaluate_full_candidates import schedule
        check=Path(spec['report']).parent/'recovery_check';check.mkdir()
        parts=restore_parts(inputs/'resume',check,identity(inputs/'protocol.json',source_digest,schedule(protocol,blueprint)),protocol,blueprint)
        restored_games=sum(len(p['rows'])for p in parts.values())

    report = dict(schema='full-evaluation-exact-package-preflight-v1', models=checked,
                  technical_games=2 * len(checked) + 1, source_manifest_sha256=source_digest,
                  protocol_sha256=sha(inputs / 'protocol.json'), actual_cuda=False,
                  independent_test_run=False, quality_accepted=False, restored_games=restored_games)
    if protocol['schema'].endswith(('-v2','-v3')):report['execution']=execution
    with Path(spec['report']).open('x') as stream: json.dump(report, stream, indent=2); stream.write('\n')


def check_package(prepared):
    from scripts.kaggle.local_artifact_transfer import file_sha
    prepared = Path(prepared)
    prep = json.loads((prepared / 'preparation.json').read_text())
    code = (prepared / 'job/train.py').read_text()
    if file_sha(prepared / 'job/train.py') != prep['bootstrap_sha256']:
        raise ValueError('Evaluation bootstrap changed')
    if ast.literal_eval(ast.parse(code).body[-1].value.args[0]) != prep['config']:
        raise ValueError('Evaluation launch config differs')
    if file_sha(prepared / 'upload_dataset/evaluation_inputs.b64') != prep['config']['transport_sha256']:
        raise ValueError('Evaluation upload changed')
    return prep, code


def audit_candidates(inputs):
    from scripts.attack.evaluate_full_candidates import validate_protocol
    from scripts.attack.verify_global_full_pair import verify
    inputs = Path(inputs); protocol = json.loads((inputs / 'protocol.json').read_text())
    validate_protocol(protocol, inputs / 'weights')
    proofs = {}
    for entry in protocol['models']:
        if entry['role'] != 'candidate': continue
        root = Path(entry['origin_prepared'])
        if root not in proofs: proofs[root] = verify(root / 'collected', root)
        frozen = json.loads((inputs / 'weights' / entry['full_proof_file']).read_text())
        if proofs[root] != frozen or entry['origin_kernel'] != frozen['kernel']:
            raise ValueError('Actual full training proof differs from packaged candidate')
    return protocol


def preflight(prepared, output):
    from scripts.kaggle.local_artifact_transfer import file_sha
    prepared, output = Path(prepared).absolute(), Path(output).absolute()
    prep, code = check_package(prepared)
    if (prepared / 'preflight.json').exists(): raise ValueError('Existing preflight must be preserved')
    output.mkdir(parents=True, exist_ok=False)
    # Execute the exact generated unpacker, not a different local substitute.
    namespace = {}; exec(compile(code.rsplit('\nbootstrap(', 1)[0], 'frozen_bootstrap', 'exec'), namespace)
    inputs, project = namespace['unpack'](prep['config'], prepared / 'upload_dataset', output)
    protocol = audit_candidates(inputs)
    spec = dict(inputs=str(inputs), report=str(output / 'child_report.json'))
    spec_path = output / 'spec.json'; spec_path.write_text(json.dumps(spec))
    with (output / 'run.log').open('x') as log:
        subprocess.run([sys.executable, str(Path(__file__).absolute()), '--child', str(spec_path)],
                       cwd=project, env=dict(os.environ, PYTHONPATH=str(project)), stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((output / 'child_report.json').read_text())
    expected = [dict(id=m['id'], sha256=m['sha256'], technical_games=2) for m in protocol['models']]
    if report['models'] != expected or report['technical_games'] != 2 * len(expected) + 1:
        raise ValueError('Exact candidate preflight coverage differs')
    report.update(preparation_sha256=file_sha(prepared / 'preparation.json'), bootstrap_sha256=prep['bootstrap_sha256'],
                  decoded_inputs=str(inputs), actual_full_training_reverified=True)
    with (prepared / 'preflight.json').open('x') as stream: json.dump(report, stream, indent=2); stream.write('\n')
    return report


def enable_submission(prepared):
    from scripts.kaggle.local_artifact_transfer import file_sha
    prepared = Path(prepared); prep, _ = check_package(prepared)
    report = json.loads((prepared / 'preflight.json').read_text())
    if (report.get('schema') != 'full-evaluation-exact-package-preflight-v1'
            or report.get('actual_full_training_reverified') is not True
            or report.get('independent_test_run') is not False or report.get('quality_accepted') is not False
            or report.get('preparation_sha256') != file_sha(prepared / 'preparation.json')
            or report.get('bootstrap_sha256') != prep['bootstrap_sha256']
            or report.get('source_manifest_sha256') != prep['config']['source_manifest_sha256']
            or report.get('protocol_sha256') != prep['config']['files']['protocol.json']):
        raise ValueError('Exact actual-candidate package preflight required')
    protocol = audit_candidates(report['decoded_inputs'])
    if protocol['schema'].endswith(('-v2','-v3')):
        from scripts.attack.evaluation_execution import execution_for
        if report.get('execution')!=execution_for(protocol):raise ValueError('Preflight execution differs')
    expected = [dict(id=m['id'], sha256=m['sha256'], technical_games=2) for m in protocol['models']]
    if report.get('models') != expected or report.get('technical_games') != len(expected) * 2 + 1:
        raise ValueError('Preflight candidate coverage differs')
    path = prepared / 'job/kernel-metadata.json'; metadata = json.loads(path.read_text())
    if metadata['id'] != prep['kernel'] or metadata['dataset_sources'] != [prep['config']['owner'] + '/' + prep['config']['slug']]:
        raise ValueError('Submission target changed')
    if metadata['enable_gpu']: raise ValueError('GPU submission already enabled')
    backup = prepared / 'kernel-metadata.preflight-disabled.json'
    with backup.open('xb') as stream: stream.write(path.read_bytes())
    metadata['enable_gpu'] = True; path.write_text(json.dumps(metadata, indent=2) + '\n')
    ready=dict(schema='full-evaluation-submission-ready-v1', preparation_sha256=file_sha(prepared/'preparation.json'),
        preflight_sha256=file_sha(prepared/'preflight.json'), bootstrap_sha256=file_sha(prepared/'job/train.py'),
        metadata_sha256=file_sha(path), transport_sha256=file_sha(prepared/'upload_dataset/evaluation_inputs.b64'))
    with(prepared/'submission_ready.json').open('x')as stream:json.dump(ready,stream,indent=2);stream.write('\n')
    return dict(kernel=prep['kernel'], enabled=True, submitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--child'); parser.add_argument('--prepared')
    parser.add_argument('--output'); parser.add_argument('--enable-submission', action='store_true')
    args = parser.parse_args()
    if args.child: child(args.child)
    elif args.prepared and args.output:
        print(json.dumps(preflight(args.prepared, args.output), indent=2))
        if args.enable_submission: print(json.dumps(enable_submission(args.prepared)))
    else: parser.error('Provide --child or --prepared with --output')
