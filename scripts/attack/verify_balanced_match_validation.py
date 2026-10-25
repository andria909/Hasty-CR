"""Audit the separate validation-game decision, retaining the original rejection."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.attack.verify_fireball_balanced import verify as original_audit,verify_games
from scripts.attack.verify_fireball_checkpoint import sha
from sim.fireball_balanced import skill_eligibility
from sim.eval_defense_duel import JOINT_MODE


def verify(collected,prepared,training_collected,training_prepared):
    root=Path(collected);prepared=Path(prepared);training_collected=Path(training_collected);training_prepared=Path(training_prepared)
    previous=original_audit(training_collected,training_prepared)
    if previous.get('candidate_available')or previous.get('reason')!='No epoch passed all skill guards':
        raise ValueError('Original rehearsal-proxy rejection must remain recorded')
    prep=json.loads((prepared/'preparation.json').read_text());config=prep['config']
    receipt=json.loads((root/'collection_receipt.json').read_text())
    if receipt.get('remote_status')!='COMPLETE'or receipt.get('complete')is not True or receipt.get('kernel')!=prep['kernel']:
        raise ValueError('Complete terminal game-validation collection required')
    def checked(name):
        path=root/name
        if receipt.get('files',{}).get(name,{}).get('sha256')!=sha(path):raise ValueError('Collected artifact SHA differs')
        return json.loads(path.read_text())
    for name,digest in config['pins'].items():
        if sha(prepared/'dataset'/name)!=digest:raise ValueError('Validation input SHA differs')
    if sha(prepared/'job/validate.py')!=prep['bootstrap_sha256']:raise ValueError('Validation bootstrap changed')
    training=json.loads((training_prepared/'preparation.json').read_text())
    for name,digest in config['source_pins'].items():
        if sha(training_prepared/'dataset'/name)!=digest:raise ValueError('Training source input differs')
    selection=json.loads((prepared/'dataset/selection.json').read_text())
    recovery=json.loads((prepared/'dataset/recovery_protocol.json').read_text());protocol=training['protocol']
    if (recovery.get('original_protocol')!=protocol or recovery.get('original_protocol_sha256')!=training['protocol_sha256']or
            recovery.get('game_guards_changed')is not False or recovery.get('test_used_for_selection')is not False):
        raise ValueError('Recovery validation protocol differs')
    alternative=deepcopy(protocol);alternative['rehearsal_action_agreement_floor']=0.
    baseline=json.loads((training_collected/'baseline_skills.json').read_text())
    rows=[json.loads(line)for line in (training_collected/'metrics.jsonl').read_text().splitlines()]
    qualified=[r for r in rows if 'validation'in r and skill_eligibility(r['validation'],r['negative_validation'],
        r['rehearsal_action_agreement'],baseline['negative'],alternative)]
    chosen=sorted(qualified,key=lambda r:(-r['minimum_skill_rate'],-r['skill_score'],r['epoch']))[0]
    if selection['epoch']!=chosen['epoch']or selection['qualified_epochs']!=[r['epoch']for r in qualified]:
        raise ValueError('Skill-qualified epoch selection differs')
    model=training_collected/f"epoch_{chosen['epoch']}.pt"
    if sha(model)!=config['pins']['epoch.pt']or selection['checkpoint_sha256']!=sha(model):raise ValueError('Epoch weight identity differs')
    if selection['validation']!=chosen['validation']['by_kind']or selection['negative_validation']!=chosen['negative_validation']:
        raise ValueError('Selected skill/cast metrics differ')
    result=checked('result.json')
    if (result.get('schema')!='balanced-match-validation-result-v1'or result.get('status')!='complete'or
            result.get('epoch')!=chosen['epoch']or result.get('source_sha256')!=training['source_sha256']or
            result.get('candidate_sha256')!=sha(model)or result.get('parent_sha256')!=protocol['teacher_sha256']or
            result.get('recovery_protocol_sha256')!=config['pins']['recovery_protocol.json']or
            result.get('training_performed')is not False or result.get('test_used_for_selection')is not False):
        raise ValueError('Game-validation result identity differs')
    parts={name:checked(name+'.json')for name in ('teacher_v25','candidate_v25','candidate_parent')}
    for name in parts:
        if result.get('parts',{}).get(name,{}).get('sha256')!=sha(root/(name+'.json')):raise ValueError('Internal game part SHA differs')
    v25_mode=next(r['observation_mode']for r in json.loads(Path('data/defense/baselines.json').read_text())['checkpoints']if r['id']=='v25')
    reference=verify_games(parts['teacher_v25'],protocol,v25_mode)
    v25=verify_games(parts['candidate_v25'],protocol,v25_mode)
    parent=verify_games(parts['candidate_parent'],protocol,JOINT_MODE)
    scores=dict(teacher_v25=reference,candidate_v25=v25,candidate_parent=parent)
    passed=(all(v is not None for v in scores.values())and parent>=protocol['minimum_score_vs_teacher']and
            v25>=reference-protocol['score_margin_vs_v25_teacher_reference'])
    if result.get('scores')!=scores or result.get('candidate_passes_validation_games')is not passed:
        raise ValueError('Actual game scores or unchanged game guards differ')
    return dict(schema='verified-balanced-match-validation-v1',epoch=chosen['epoch'],checkpoint_sha256=sha(model),
        parent_sha256=protocol['teacher_sha256'],source_sha256=training['source_sha256'],
        validation=selection['validation'],negative_validation=selection['negative_validation'],
        rehearsal_action_agreement=chosen['rehearsal_action_agreement'],rehearsal_agreement_role='diagnostic_only',
        game_summaries={k:v['summary']for k,v in parts.items()},scores=scores,
        candidate_passes_validation_games=passed,independent_test_required=passed,
        original_phase_rejection_preserved=True,quality_accepted=False,live_checkpoint_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','prepared','training-collected','training-prepared','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.output.write_text(json.dumps(verify(a.collected,a.prepared,a.training_collected,a.training_prepared),indent=2)+'\n')
