"""Real CPU optimizer/resume check for opt-in tactical full-game starts."""
import argparse
import json
from pathlib import Path

import torch

from scripts.kaggle.preflight_global_full_pair import equal_state
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.train_attack_ppo import parser, train, evaluate


def preflight(root, initial):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    profile = root/'profile.json'
    profile.write_text(json.dumps(dict(cannon_center_only=1, action_cooldown_ms=0,
                                      tower_damage_all=1)))
    evidence = []
    for fraction in (0., 1.):
        case = root/('control' if fraction == 0 else 'tactical')
        common = ['--device','cpu','--task-mode','joint','--curriculum','continuous_v1',
            '--continuous-profile',str(profile),'--tower-support-weight','.15','--spell-waste-weight','.1',
            '--tactical-start-fraction',str(fraction),'--action-delay-ms','1350',
            '--geometry','native_anchors_candidate_v1','--max-ms','310000',
            '--envs','1','--rollout','16','--epochs','1','--minibatch-size','16',
            '--seed','34001','--checkpoint-every','1','--eval-every','0','--wallclock-seconds','600']
        for name, steps in (('first',16), ('whole',32), ('resumed',32)):
            source = ['--resume',str(case/'first/update_000001.pt')] if name == 'resumed' else ['--init',str(initial)]
            args = common+source+['--output',str(case/name),'--updates',str(steps//16),
                                   '--target-environment-steps',str(steps)]
            train(parser().parse_args(args))
        a = torch.load(case/'whole/update_000002.pt', map_location='cpu', weights_only=False)
        b = torch.load(case/'resumed/update_000002.pt', map_location='cpu', weights_only=False)
        for key in ('state_dict','optimiser','rng','observations','infos','episode_returns'):
            if not equal_state(a[key], b[key]):
                raise ValueError(f'Exact resume differs: {key}')
        state = a['environment_states'][0]
        if bool(state['tactical_start']) != bool(fraction):
            raise ValueError('Tactical start not actually exercised')
        evidence.append(dict(fraction=fraction, actual_steps=a['step'],
            source_code_sha256=a['config']['source_code_sha256'],
            tactical_start=state['tactical_start'], exact_resume=True,
            checkpoint_sha256=file_sha(case/'whole/update_000002.pt')))
    result = dict(schema='tactical-start-preflight-v1', cases=evidence,
        initial_sha256=file_sha(initial), device='cpu', quality_accepted=False,
        validation_unchanged_test='tests/test_continuous_tactical_starts.py',
        note='Technical 16-to-32-step check only; not a quality training run or GPU submission.')
    from scripts.attack.evaluate_continuous_pool import load
    network=load(dict(file=str(Path(initial).absolute()),sha256=file_sha(initial),scalars=309),Path('/'),'cpu')
    options=dict(curriculum='continuous_v1',action_delay_ms=1350,max_ms=310000,
        continuous_profile=json.loads(profile.read_text()),tower_support_weight=.15,spell_waste_weight=.1,
        validation_selection='worst_family')
    base=evaluate(network,'cpu',episodes=6,seed=834001,env_options=options,task_mode='joint')
    candidate=evaluate(network,'cpu',episodes=6,seed=834001,
        env_options=dict(options,tactical_start_fraction=1.),task_mode='joint')
    for left,right in zip(base['records'],candidate['records']):
        right=dict(right)
        if right.pop('tactical_start') is not None or right.pop('tactical_start_fraction') != 1.:
            raise ValueError('Tactical start leaked into validation')
        if left != right:raise ValueError('Full validation behavior changed')
    if len(base['records']) != 12 or len(candidate['records']) != 12:
        raise ValueError('Incomplete technical validation')
    (root/'validation_control.json').write_text(json.dumps(base,indent=2)+'\n')
    (root/'validation_tactical_option.json').write_text(json.dumps(candidate,indent=2)+'\n')
    result.update(technical_validation_games=24,identical_validation_pairs=12,
        validation_control_sha256=file_sha(root/'validation_control.json'),
        validation_tactical_option_sha256=file_sha(root/'validation_tactical_option.json'))
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--initial', required=True)
    a=p.parse_args()
    preflight(a.root, a.initial)
