"""Local algorithm rehearsal; not a GPU-controller submission preflight."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.kaggle.local_artifact_transfer import file_sha
from scripts.kaggle.preflight_global_full_pair import smoke_args,set_flag
from scripts.attack.boundary_state import equal_boundary

CASES=(('control','all','all'),('choice_loss','choice','all'),
       ('choice_advantages','all','choice'),('choice_both','choice','choice'))
BOUNDARY_KEYS=('state_dict','optimiser','rng','sampler','environment_states','observations','infos',
               'episode_returns','episode_counters','step','update')


def run(source,output):
    import torch
    from sim.train_attack_ppo import initialize_attack_policy,evaluate,validate_resume
    from sim.continuous_push import CONTRACT
    torch.set_num_threads(1)
    source,output=Path(source).absolute(),Path(output).absolute()
    old_proof=json.loads((source/'child_preflight.json').read_text())
    args=old_proof['original_pilot_command']
    if (not old_proof['exact_resume_16_to_32'] or old_proof['pilot_steps']!=409600 or
            old_proof['actual_focus']!='uniform'):
        raise ValueError('Verified frozen control command required')
    output.mkdir(parents=True,exist_ok=False)
    driver_sha=file_sha(__file__)
    (output/'driver.py').write_bytes(Path(__file__).read_bytes())
    commands=[]
    def execute(command,log):
        command=[sys.executable,*command[1:]]
        commands.append(command)
        with log.open('x')as handle:
            subprocess.run(command,cwd=Path.cwd(),env=dict(os.environ,PYTHONPATH=str(Path.cwd())),
                           stdout=handle,stderr=subprocess.STDOUT,check=True)
    # Same 32-step command and initial state as the frozen pre-change trainer.
    execute(smoke_args(args,output/'legacy_control',32),output/'legacy.log')
    old=torch.load(source/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    legacy=torch.load(output/'legacy_control/update_000002.pt',map_location='cpu',weights_only=False)
    for key in BOUNDARY_KEYS:
        if not equal_boundary(old[key],legacy[key]):raise ValueError('Default changed frozen baseline: '+key)
    print('FROZEN_BASELINE_EXACT',flush=True)
    reports=[]
    for name,reduction,normalization in CASES:
        case=output/name;case.mkdir()
        def command(target,steps,resume=None):
            cmd=smoke_args(args,target,steps,resume)
            for flag,value in (('--envs',2),('--rollout',64),('--updates',steps//128)):
                cmd=set_flag(cmd,flag,value)
            return cmd+['--policy-reduction',reduction,'--advantage-normalization',normalization]
        execute(command(case/'first',128),case/'first.log')
        execute(command(case/'whole',256),case/'whole.log')
        execute(command(case/'resumed',256,case/'first/update_000001.pt'),case/'resumed.log')
        whole=torch.load(case/'whole/update_000002.pt',map_location='cpu',weights_only=False)
        resumed=torch.load(case/'resumed/update_000002.pt',map_location='cpu',weights_only=False)
        for key in BOUNDARY_KEYS:
            if not equal_boundary(whole[key],resumed[key]):raise ValueError('Exact resume changed '+name+': '+key)
        config=whole['config']
        if (config['policy_reduction']!=reduction or config['advantage_normalization']!=normalization
                or whole['step']!=256 or config['continuous_profile']['cannon_center_only']!=1
                or config['action_delay_ms']!=1350):
            raise ValueError('Actual configuration changed')
        for key in ('policy_reduction','advantage_normalization'):
            wrong=dict(config);wrong[key]='choice'if config[key]=='all'else'all'
            try:validate_resume(whole,wrong)
            except ValueError as exc:
                if key not in str(exc):raise
            else:raise ValueError('Cross-objective resume was accepted')
        metrics=[json.loads(line)for line in (case/'whole/metrics.jsonl').read_text().splitlines()]
        for metric in metrics:
            d=metric['policy_decisions']
            if (d['choice_steps']+d['forced_steps']!=128 or d['rollout_steps']!=128 or
                    not 0<d['optimizer_samples']<=384 or d['optimizer_samples']!=16*d['optimizer_batches'] or
                    not 0<d['choice_samples']<=d['optimizer_samples'] or
                    not 0<=d['no_choice_minibatches']<=d['optimizer_batches'] or
                    d['reduction']!=reduction or d['advantage_normalization']!=normalization or
                    not d['actor_updates_enabled']):
                raise ValueError('Actual optimizer decision accounting differs')
            if not metric['optimizer_controls']['kl_stopped'] and (
                    d['optimizer_samples']!=384 or d['choice_samples']!=3*d['choice_steps']):
                raise ValueError('Non-stopped PPO did not complete all three epochs')
        net,_,_=initialize_attack_policy(whole,device='cpu',num_scalars=309,observation_contract=CONTRACT)
        keys=('curriculum','continuous_profile','validation_selection','continuous_focus','geometry',
              'coordinate_mode','max_ms','tower_support_weight','spell_waste_weight','crown_only_weight','dormant_king_weight')
        options={k:config[k]for k in keys};options['action_delay_ms']=config['validation_action_delay_ms']
        validation=evaluate(net,'cpu',episodes=6,seed=900100,task_mode='joint',env_options=options)
        if len(validation['records'])!=12 or any(r['illegal_actions']or r['cannon_outside_center']for r in validation['records']):
            raise ValueError('Full technical games failed')
        (case/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        reports.append(dict(id=name,reduction=reduction,normalization=normalization,exact_resume=True,
            cross_objective_resume_rejected=True,checkpoint_sha256=file_sha(case/'whole/update_000002.pt'),
            source_code_sha256=config['source_code_sha256'],policy_decisions=[r['policy_decisions']for r in metrics],
            optimizer_controls=[r['optimizer_controls']for r in metrics],
            validation_games=12,validation_sha256=file_sha(case/'validation.json')))
        print(json.dumps(dict(case=name,exact_resume=True,technical_games=12)),flush=True)
    if file_sha(__file__)!=driver_sha:raise ValueError('Preflight driver changed')
    result=dict(schema='policy-choice-local-algorithm-preflight-v1',source=str(source),
        source_proof_sha256=file_sha(source/'child_preflight.json'),driver_sha256=driver_sha,
        boundary_comparator_sha256=file_sha(Path(__file__).with_name('boundary_state.py')),
        frozen_default_boundary_exact=True,checked_boundary_keys=BOUNDARY_KEYS,checks=reports,
        commands=commands,validation_games=48,technical_fixture=True,submission_allowed=False,
        actual_cuda=False,quality_accepted=False,
        scope='Actual local PPO128→256 resume; not the still-unprepared GPU controller')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
