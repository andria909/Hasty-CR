"""Evaluate skill-qualified v3 weights using games instead of an action-copy proxy.

The original rejected phase remains unchanged. No skill floor or game threshold
is weakened, and this separate validation protocol is fixed before its games.
"""
import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil

from scripts.attack.verify_fireball_balanced import verify
from scripts.kaggle.run_five_round_robin import sha,prepare_source
from sim.fireball_balanced import skill_eligibility


def run_validation(config):
    import hashlib,json,shutil,sys,tarfile
    from pathlib import Path,PurePosixPath
    from concurrent.futures import ThreadPoolExecutor
    def mount(slug):
        root=Path('/kaggle/input');paths=[p for p in (root/slug,root/'datasets'/config['owner']/slug)if p.is_dir()]
        if len(paths)!=1:raise ValueError('Ambiguous validation input mount')
        return paths[0]
    source=mount(config['source_slug']);inputs=mount(config['selection_slug'])
    for name,digest in config['pins'].items():
        if sha(inputs/name)!=digest:raise ValueError('Validation selection input differs')
    for name,digest in config['source_pins'].items():
        if sha(source/name)!=digest:raise ValueError('Original training input differs')
    project=Path('/kaggle/working/balanced_validation_project')
    prepare_source(source,project,config['source_pins']['manifest.json'])
    sys.path.insert(0,str(project))
    import torch
    from sim.train_attack_ppo import initialize_attack_policy
    from scripts.attack.train_fireball_balanced import validation_games,match_score
    from sim.eval_defense_duel import load_network,JOINT_MODE
    from sim.defense_assets import load_defense_assets
    from sim.runner import resolve_deck
    from sim.env import DECK_26
    torch.set_num_threads(2)
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    selection=json.loads((inputs/'selection.json').read_text());protocol=json.loads((inputs/'recovery_protocol.json').read_text())
    original=protocol['original_protocol']
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,DECK_26))
    if spells['fireball'].projectile_speed_mt_per_sec!=12000:raise ValueError('Wrong shared physics')
    candidate_blob=torch.load(inputs/'epoch.pt',map_location='cpu',weights_only=False)
    if candidate_blob['epoch']!=selection['epoch']or candidate_blob['config']['source_sha256']!=config['source_sha256']:
        raise ValueError('Candidate boundary differs')
    parent_blob=torch.load(source/'parent.pt',map_location='cpu',weights_only=False)
    def network(blob,device):
        net,_,_=initialize_attack_policy(blob,device=device,num_scalars=309)
        if not all(torch.isfinite(t).all()for t in net.state_dict().values()):raise ValueError('Nonfinite policy')
        return net.eval().requires_grad_(False)
    output=Path('/kaggle/working/balanced_match_validation');output.mkdir(exist_ok=False)
    v25_mode=next(r['observation_mode']for r in json.loads((project/'data/defense/baselines.json').read_text())['checkpoints']if r['id']=='v25')
    result=dict(schema='balanced-match-validation-result-v1',status='running',epoch=selection['epoch'],
        candidate_sha256=config['pins']['epoch.pt'],parent_sha256=config['source_pins']['parent.pt'],
        source_sha256=config['source_sha256'],recovery_protocol_sha256=config['pins']['recovery_protocol.json'],
        training_performed=False,test_used_for_selection=False,quality_accepted=False,live_checkpoint_changed=False)
    def save(name,value):
        path=output/(name+'.json');path.write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(dict(event='validation_part_complete',part=name)),flush=True)
    print(json.dumps(dict(event='CUDA_READY',gpu=[torch.cuda.get_device_name(i)for i in range(2)],
        source_sha256=config['source_sha256'],candidate_sha256=config['pins']['epoch.pt'])),flush=True)
    def reference_and_v25():
        parent=network(parent_blob,'cuda:0');candidate=network(candidate_blob,'cuda:0')
        v25=load_network(source/'v25.pt','cuda:0',v25_mode)
        reference=validation_games(parent,v25,'cuda:0',v25_mode,cards,spells,original);save('teacher_v25',reference)
        comparison=validation_games(candidate,v25,'cuda:0',v25_mode,cards,spells,original);save('candidate_v25',comparison)
        return reference,comparison
    def against_parent():
        parent=network(parent_blob,'cuda:1');candidate=network(candidate_blob,'cuda:1')
        comparison=validation_games(candidate,parent,'cuda:1',JOINT_MODE,cards,spells,original);save('candidate_parent',comparison)
        return comparison
    try:
        with ThreadPoolExecutor(2)as pool:
            a=pool.submit(reference_and_v25);b=pool.submit(against_parent)
            reference,v25=a.result();parent=b.result()
        reference_score=match_score(reference['summary']);v25_score=match_score(v25['summary']);parent_score=match_score(parent['summary'])
        passed=(all(v is not None for v in (reference_score,v25_score,parent_score))and
            parent_score>=original['minimum_score_vs_teacher']and
            v25_score>=reference_score-original['score_margin_vs_v25_teacher_reference'])
        result.update(status='complete',candidate_passes_validation_games=passed,
            scores=dict(teacher_v25=reference_score,candidate_v25=v25_score,candidate_parent=parent_score),
            independent_test_required=passed,
            parts={name:dict(sha256=sha(output/(name+'.json')))for name in ('teacher_v25','candidate_v25','candidate_parent')})
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__);raise
    finally:(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def prepare(collected,training,output):
    collected=Path(collected);training=Path(training)
    audited=verify(collected,training)
    if audited.get('candidate_available')or audited.get('reason')!='No epoch passed all skill guards':
        raise ValueError('This stage only diagnoses the rehearsal-proxy rejection, not failed game guards')
    source_preparation=json.loads((training/'preparation.json').read_text());protocol=source_preparation['protocol']
    comparison=deepcopy(protocol);comparison['rehearsal_action_agreement_floor']=0.
    baseline=json.loads((collected/'baseline_skills.json').read_text())
    rows=[json.loads(line)for line in (collected/'metrics.jsonl').read_text().splitlines()]
    qualified=[r for r in rows if 'validation'in r and skill_eligibility(r['validation'],r['negative_validation'],
        r['rehearsal_action_agreement'],baseline['negative'],comparison)]
    if not qualified:raise ValueError('No epoch meets every unchanged skill and cast-control threshold')
    row=sorted(qualified,key=lambda r:(-r['minimum_skill_rate'],-r['skill_score'],r['epoch']))[0]
    receipt=json.loads((collected/'collection_receipt.json').read_text());weight=collected/f"epoch_{row['epoch']}.pt"
    if receipt['files'].get(weight.name,{}).get('sha256')!=sha(weight):raise ValueError('Saved epoch SHA differs')
    import torch
    blob=torch.load(weight,map_location='cpu',weights_only=False)
    if (blob.get('epoch')!=row['epoch']or blob.get('config')!=json.loads((collected/'config.json').read_text())or
            any(blob.get(k)!=row[k]for k in ('validation','negative_validation','rehearsal_action_agreement'))or
            not all(torch.isfinite(t).all()for t in blob['state_dict'].values())):
        raise ValueError('Saved epoch boundary differs')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);dataset=output/'dataset';dataset.mkdir()
    shutil.copyfile(weight,dataset/'epoch.pt')
    selection=dict(schema='balanced-validation-recovery-selection-v1',epoch=row['epoch'],checkpoint_sha256=sha(weight),
        source_sha256=source_preparation['source_sha256'],qualified_epochs=[r['epoch']for r in qualified],
        validation=row['validation']['by_kind'],negative_validation=row['negative_validation'],
        rehearsal_action_agreement=row['rehearsal_action_agreement'],test_used_for_selection=False,
        original_phase_result_preserved=True,quality_accepted=False)
    recovery=dict(schema='balanced-validation-recovery-protocol-v1',original_protocol=protocol,
        original_protocol_sha256=source_preparation['protocol_sha256'],source_sha256=source_preparation['source_sha256'],
        selection_rule='unchanged per-skill/cast floors, minimum skill then macro score then earliest epoch',
        rehearsal_agreement_role='diagnostic only; actual paired validation games determine strategy preservation',
        game_guards_changed=False,training_performed=False,test_used_for_selection=False)
    (dataset/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    (dataset/'recovery_protocol.json').write_text(json.dumps(recovery,indent=2)+'\n')
    owner='andriashadrin';slug='hastycr-balanced-match-inputs-20261025'
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    config=dict(owner=owner,source_slug='hastycr-fireball-balanced-inputs-20261025',selection_slug=slug,
        source_sha256=source_preparation['source_sha256'],source_pins=source_preparation['pins'],
        pins={name:sha(dataset/name)for name in ('epoch.pt','selection.json','recovery_protocol.json')})
    job=output/'job';job.mkdir()
    code='import hashlib,json,shutil,sys,tarfile\nfrom pathlib import Path,PurePosixPath\n'+inspect.getsource(sha)+'\n'+inspect.getsource(prepare_source)+'\n'+inspect.getsource(run_validation)+f'\nrun_validation({config!r})\n'
    (job/'validate.py').write_text(code)
    kernel=owner+'/hastycr-balanced-match-validation-20261025'
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title='hastycr balanced match validation 20261025',
        code_file='validate.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,
        enable_internet=False,dataset_sources=[owner+'/'+slug,owner+'/'+config['source_slug']],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    report=dict(schema='balanced-match-validation-preparation-v1',kernel=kernel,config=config,selection=selection,
        bootstrap_sha256=sha(job/'validate.py'),matches=60,submitted=False)
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','training','output'):p.add_argument('--'+name,required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))
