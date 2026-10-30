"""Global validation-only profile selection across fully verified pilot cohorts.

Two training seeds must each meet the unchanged worst-family floor. A cohort
may contribute one eligible profile; two profiles are still required globally.
This prepares a ranking only, never submits training or promotes weights.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tarfile

from scripts.attack.verify_continuous_focus import verify


def verify_study(root, default_verifier=None):
    root = Path(root)
    protocol = json.loads((root/'dataset/protocol.json').read_text())
    if protocol.get('study_variant') == 'crown_economy_v1':
        from scripts.attack.verify_crown_economy_pilot import verify_crown
        return verify_crown(root/'collected', root)
    if protocol.get('study_variant') == 'bank_counterfactual_v1':
        from scripts.attack.verify_bank_teacher_pilot import verify as verify_bank
        return verify_bank(root/'collected', root)
    if protocol.get('study_variant') == 'mana_decision_v1':
        from scripts.attack.verify_mana_campaign import verify_mana
        return verify_mana(root/'collected', root)
    return (default_verifier or verify)(root/'collected', root)


def unique_mana_profiles(ranking):
    """Use a deterministic canonical control, never pick its luckiest replica."""
    result = []
    by_profile = {}
    for row in sorted(ranking, key=lambda r: (r['study'], r['trial']['id'])):
        key = json.dumps({k:v for k,v in row['trial'].items() if k != 'id'}, sort_keys=True)
        if key in by_profile:
            by_profile[key]['replicated_controls'].append(dict(
                study=row['study'], score=row['score'], minimum_family_points=row['minimum_family_points']))
        else:
            row = dict(row, replicated_controls=[])
            by_profile[key] = row
            result.append(row)
    return result

CORE_FILES=('sim/match.py','sim/engine.py','sim/entities.py','sim/gamedata.py','sim/public_gamedata.py',
    'sim/spells.py','sim/arena.py','sim/adapter.py','sim/env.py','sim/eval_defense_duel.py',
    'sim/opponents.py','sim/kite_geometry.py','sim/projectile_calibration.py','sim/pathfind.py',
    'sim/continuous_validation.py','scripts/attack/select_continuous_focus.py')
EXTRA_DEFAULTS={'fireball_value_reward','reserve_potential_weight','barbarian_train_fraction','lane_reflection_fraction'}


def attribute(node,name):
    return isinstance(node,ast.Attribute)and isinstance(node.value,ast.Name)and node.value.id=='self'and node.attr==name


def train_guard(node):
    expected=ast.parse("self.split == 'train'",mode='eval').body
    return isinstance(node,ast.BoolOp)and isinstance(node.op,ast.And)and ast.dump(node.values[0])==ast.dump(expected)


def normalized_wrapper(source):
    tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='ContinuousPushEnv')
    class ValidationOnly(ast.NodeTransformer):
        def visit_If(self,node):
            if train_guard(node.test)or attribute(node.test,'reflected'):
                if node.orelse:raise ValueError('Unexpected validation branch')
                return None
            for name in('barbarian_train_fraction','lane_reflection_fraction'):
                expected=ast.parse(f"self.profile['{name}'] > 1",mode='eval').body
                if ast.dump(node.test)==ast.dump(expected)and len(node.body)==1 and isinstance(node.body[0],ast.Raise)and not node.orelse:return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if len(node.targets)==1 and attribute(node.targets[0],'reflected'):
                value=node.value
                if not(isinstance(value,ast.Call)and isinstance(value.func,ast.Name)and value.func.id=='bool'and len(value.args)==1 and train_guard(value.args[0])):
                    raise ValueError('Validation reflection is not proven disabled')
                return None
            if any(attribute(t,n)for t in node.targets for n in('_reserve_shaping_return','_reserve_shaping_discounted_return')):
                if len(node.targets)!=1 or not isinstance(node.value,ast.Constant)or node.value.value!=0:raise ValueError('Unexpected auxiliary state')
                return None
            if len(node.targets)==1 and attribute(node.targets[0],'profile'):
                value=node.value
                if not isinstance(value,ast.Call)or not isinstance(value.func,ast.Name)or value.func.id!='dict'or value.args:raise ValueError('Unexpected profile definition')
                for kw in value.keywords:
                    if kw.arg in EXTRA_DEFAULTS and(not isinstance(kw.value,ast.Constant)or kw.value.value!=0):
                        raise ValueError('New validation default changes behavior')
                value.keywords=[kw for kw in value.keywords if kw.arg not in EXTRA_DEFAULTS]
            return self.generic_visit(node)
    result={}
    for name in('CONTRACT','DECKS'):
        assignment=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets))
        result[name]=ast.literal_eval(assignment.value)
    for name in('__init__','reset','_mask','_observe','_enemy','step','_info'):
        method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name==name))
        method=ValidationOnly().visit(method)
        if name=='step':
            end=next(i for i,n in enumerate(method.body)if isinstance(n,ast.Assign)and any(attribute(t,'terminal_reason')for t in n.targets))
            final=method.body[-1]
            if not isinstance(final,ast.Return):raise ValueError('Unexpected validation transition return')
            method.body=method.body[:end+1]+[final]
        result[name]=ast.dump(method)
    return result


def contract(prepared):
    prepared=Path(prepared);original=json.loads((prepared/'preparation.json').read_text());config=original['config']
    manifest=json.loads((prepared/'dataset/manifest.json').read_text())
    with tarfile.open(prepared/'dataset/extensions.tar.gz')as stream:
        raw=stream.extractfile('sim/continuous_push.py').read()
        trainer=stream.extractfile('sim/train_attack_ppo.py').read()
    if hashlib.sha256(raw).hexdigest()!=config['extensions']['sim/continuous_push.py']:raise ValueError('Wrapper changed')
    if hashlib.sha256(trainer).hexdigest()!=config['extensions']['sim/train_attack_ppo.py']:raise ValueError('Frozen evaluator changed')
    trainer=ast.parse(trainer.decode())
    return dict(source_manifest_sha256=config['source_manifest_sha256'],
        core={name:config['extensions'].get(name,manifest['files'].get(name))for name in CORE_FILES},
        evaluator={name:ast.dump(next(n for n in trainer.body if isinstance(n,ast.FunctionDef)and n.name==name))
            for name in('evaluate','make_training_env','observation_contract')},
        normalized_validation_wrapper=normalized_wrapper(raw.decode()))


def select(prepared_roots,limit=2):
    if not 2<=limit<=4:raise ValueError('Choose2–4 profiles')
    ranking=[];studies=[];common=None;contexts=None;common_runtime=None;kernels=set()
    mana_flags=[];mana_physics=None
    for root in map(Path,prepared_roots):
        proof=verify_study(root)
        if proof.get('promotion_blocked'):
            raise ValueError('Pilot promotion blocked: '+proof['promotion_block_reason'])
        if proof['kernel']in kernels:raise ValueError('Duplicate cohort cannot supply a second profile')
        kernels.add(proof['kernel'])
        if proof['pilot_steps_verified']!=819200:raise ValueError('Complete verified pilot cohort required')
        identity=contract(root)
        if any(value is None for value in identity['core'].values()):raise ValueError('Missing core physics pin')
        if common is None:common=identity
        elif identity!=common:raise ValueError('Different validation physics or wrapper semantics')
        reference=json.loads((root/'collected/parent_reference.json').read_text())
        current=[(r['scenario_id'],r['episode_seed'],r['match_seed'])for r in reference['validation']['records']]
        if contexts is None:contexts=current
        elif current!=contexts:raise ValueError('Different validation contexts')
        protocol=json.loads((root/'dataset/protocol.json').read_text())
        mana_flags.append(protocol.get('study_variant') == 'mana_decision_v1')
        if mana_flags[-1]:
            if mana_physics is None:mana_physics=protocol['physics_pins']
            elif mana_physics!=protocol['physics_pins']:raise ValueError('Mana cohorts use different frozen simulator modules')
        if len(protocol['pilot_seeds'])!=2 or len(set(protocol['pilot_seeds']))!=2:
            raise ValueError('Two independently trained seeds required')
        for model in proof['pilot_models']:
            config=json.loads((root/'collected'/model['id']/'config.json').read_text())
            runtime={key:config[key]for key in('action_delay_ms','max_ms','task_mode','observation_contract',
                'num_scalars','num_planes','history_frames','coordinate_mode','geometry','geometry_sha256',
                'validation_selection','validation_seed','eval_episodes')}
            runtime.update({key:config['continuous_profile'][key]for key in('action_cooldown_ms','decision_interval_ms')})
            if runtime['num_planes']!=20 or runtime['history_frames']!=1 or runtime['num_scalars']!=309:
                raise ValueError('Global pool requires the current20/309 contract')
            if common_runtime is None:common_runtime=runtime
            elif runtime!=common_runtime:raise ValueError('Different runtime geometry, clocks or execution delays')
        for row in proof['pilot_selection']['ranking']:
            floor=proof['pilot_selection']['floor']
            if floor<.1 or row['eligible']!=(row['minimum_family_points']>=floor):
                raise ValueError('Global selection cannot weaken the per-profile floor')
            item=dict(study=proof['kernel'],prepared=str(root),trial=row['trial'],
                seed_pair=protocol['pilot_seeds'],score=row['score'],minimum_family_points=row['minimum_family_points'],
                eligible=row['eligible'],floor=proof['pilot_selection']['floor'])
            if mana_flags[-1]:
                records=[]
                for seed in protocol['pilot_seeds']:
                    saved=json.loads((root/'collected'/f"pilot_{row['trial']['id']}_s{seed}_summary.json").read_text())
                    records.extend(saved['validation']['records'])
                item['quality_diagnostics']=dict(
                    mean_own_fraction_loss=sum(r['own_fraction_loss']for r in records)/len(records),
                    mean_empty_spell_elixir=sum(r['empty_spell_elixir']for r in records)/len(records))
            ranking.append(item)
        studies.append(dict(kernel=proof['kernel'],prepared=str(root),pilot_steps_verified=819200))
    if not studies:raise ValueError('No verified cohorts')
    if any(mana_flags):
        if not all(mana_flags):raise ValueError('Do not mix economy cohorts with old training physics')
        ranking=unique_mana_profiles(ranking)
    ranking.sort(key=lambda r:(-r['score'],r.get('quality_diagnostics',{}).get('mean_own_fraction_loss',0),
        r.get('quality_diagnostics',{}).get('mean_empty_spell_elixir',0),r['study'],r['trial']['id']))
    eligible=[r for r in ranking if r['eligible']]
    return dict(schema='verified-global-pilot-profile-selection-v1',studies=studies,ranking=ranking,
        selected=eligible[:limit]if len(eligible)>=2 else[],eligible_profiles=len(eligible),gate_met=len(eligible)>=2,
        selection='Same72 validation contexts; original per-profile two-seed/all-family floor retained',
        comparison_contract_sha256=hashlib.sha256(json.dumps(common,sort_keys=True).encode()).hexdigest(),
        common_runtime=common_runtime,
        test_used_for_selection=False,training_submitted=False,quality_accepted=False,live_checkpoint_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',nargs='+',required=True);p.add_argument('--limit',type=int,default=2);p.add_argument('--output',required=True)
    a=p.parse_args();target=Path(a.output)
    if target.exists():raise ValueError('Preserve previous selection reports')
    target.write_text(json.dumps(select(a.prepared,a.limit),indent=2)+'\n')
