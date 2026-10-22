"""Offline, narrowly patched stage2 GRPO recovery source; never uploads or trains."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import textwrap
import ast

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.defense.pack_evaluation import verified_frozen_package

STAGE2_ARCHIVE_SHA256='26d0cf56182d7cd7fca2c4ff9320485a3aa06b9b9d19a4b7a1973034346f16bd'
CONTRACT_FILES=('defense_env.py','defense_scenarios.py','defense_assets.py',
    'public_scenario_spells.py','engine.py','entities.py','match.py','env.py',
    'gamedata.py','spells.py','defense_sampler.py','defense_curriculum.py')


def sha(data):return hashlib.sha256(data).hexdigest()


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('Frozen patch marker is absent or ambiguous: '+old[:100])
    return source.replace(old,new,1)


def patch_environment(source):
    source=replace_once(source,'from sim.entities import Entity\n','from sim.entities import Entity, WINDUP, ATTACKING\n')
    source=replace_once(source,'        self._script_seen_units=set()\n',
        '        self._script_seen_units=set()\n        self._script_engaged_units=set()\n        self._script_trigger_reasons={}\n')
    source=replace_once(source,'                self._script_seen_units.add(name)\n                if unit.pos.y>=11500:',
        '                self._script_seen_units.add(name)\n'
        '                # Engagement may prevent a ranged escort from ever reaching the bridge.\n'
        '                if unit.state in (WINDUP,ATTACKING) or 0<unit.hitpoints<unit.max_hitpoints:\n'
        '                    self._script_engaged_units.add(name)\n'
        '                if unit.pos.y>=11500:')
    old='''                # Arrival is latched: later death must not erase the trigger.
                # If the escort dies on approach, continue the scheduled attack
                # rather than waiting forever for an impossible arrival.
                ready=(name in self._bridge_ready or any(u.pos.y>=11500 for u in units)
                       or (name in self._script_seen_units and not units))
'''
    new='''                reason=('bridge_arrival' if name in self._bridge_ready or any(u.pos.y>=11500 for u in units)
                        else 'escort_defeated' if name in self._script_seen_units and not units
                        else 'escort_engaged' if name in self._script_engaged_units else None)
                ready=reason is not None
                if ready:
                    self._script_trigger_reasons.setdefault(self._script_index,dict(
                        event_index=self._script_index,card=event.card,escort=event.near_bridge_after,
                        reason=reason,at_ms=self.match.elapsed_ms))
'''
    source=replace_once(source,old,new)
    source=replace_once(source,'                    script_plays_completed=self._script_index,script_plays_total=len(self.scenario.plays),\n',
        '                    script_plays_completed=self._script_index,script_plays_total=len(self.scenario.plays),\n'
        '                    script_trigger_reasons=list(self._script_trigger_reasons.values()),\n')
    source=replace_once(source,'        self.__dict__=deepcopy(snapshot)\n        return self._observe(),self._info()\n',
        '        self.__dict__=deepcopy(snapshot)\n'
        '        self._script_engaged_units=set(snapshot.get("_script_engaged_units",()))\n'
        '        self._script_trigger_reasons=deepcopy(snapshot.get("_script_trigger_reasons",{}))\n'
        '        return self._observe(),self._info()\n')
    compile(source,'patched frozen defense_env.py','exec')
    return source


MIGRATION_HELPERS='''
APPROVED_ENVIRONMENT_MIGRATION_SHA256 = __PINNED_SHA__


def load_resume_code_migration(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=APPROVED_ENVIRONMENT_MIGRATION_SHA256:
        raise ValueError('Resume code migration file SHA is not approved for this recovery source')
    data=json.loads(raw)
    if (data.get('schema')!='defense-grpo-environment-migration-v1'
            or data.get('not_exact_resume') is not True
            or data.get('reset_best_validation') is not True
            or not isinstance(data.get('reason'),str) or not data['reason'].strip()):
        raise ValueError('Invalid recovery migration metadata')
    return data


def recovery_resume_differences(previous,current,migration=None):
    differences=[key for key in RESUME_KEYS if previous.get(key)!=current.get(key)]
    if migration is not None:
        if (migration.get('from_environment_code_sha256')!=previous.get('environment_code_sha256')
                or migration.get('to_environment_code_sha256')!=current.get('environment_code_sha256')
                or migration.get('from_environment_code_sha256')==migration.get('to_environment_code_sha256')
                or migration.get('not_exact_resume') is not True
                or migration.get('reset_best_validation') is not True):
            raise ValueError('Recovery migration does not match the exact old/new source pair')
        differences=[key for key in differences if key!='environment_code_sha256']
    return differences
'''


def patch_trainer(source,migration_sha):
    source=replace_once(source,'\ndef restore_training_state(',
        '\n'+MIGRATION_HELPERS.replace('__PINNED_SHA__',repr(migration_sha))+'\n\ndef restore_training_state(')
    source=replace_once(source,"    differences=[key for key in RESUME_KEYS if blob['config'].get(key)!=config.get(key)]\n",
        "    migration=config.get('code_migration') if config.get('code_migration_applied_this_resume') else None\n"
        "    differences=recovery_resume_differences(blob['config'],config,migration)\n")
    marker="    load_policy_state(network,blob['state_dict'])\n"
    position=source.index(marker,source.index('def restore_training_state'))
    required='''    required=('state_dict','reference_state_dict','optimizer','sampler','updates','groups',
              'environment_steps','degenerate_groups','actual_environment_steps','torch_rng',
              'cuda_rng','action_rng','numpy_rng','python_rng','best_validation')
    if any(key not in blob or blob[key] is None for key in required):
        raise ValueError('Recovery checkpoint is missing complete boundary state')
    if (not isinstance(blob['optimizer'],dict) or
            not {'state','param_groups'}<=blob['optimizer'].keys() or
            blob['groups']!=blob['updates']):
        raise ValueError('Recovery checkpoint contains partial optimizer/group state')
    # Validate sampler identity/cursor without mutating the active sampler.
    copy.deepcopy(sampler).load_state_dict(blob['sampler'])
'''
    source=source[:position]+'    if migration is not None:\n'+textwrap.indent(required,'    ')+source[position:]
    source=replace_once(source,"                best_validation=blob['best_validation'])",
        "                best_validation=-float('inf') if migration is not None else blob['best_validation'])")
    source=replace_once(source,"    p.add_argument('--output', type=Path, required=True)",
        "    p.add_argument('--resume-code-migration',type=Path,help='Explicit pinned source migration; not an exact trajectory resume')\n"
        "    p.add_argument('--output', type=Path, required=True)")
    source=replace_once(source,'    args = p.parse_args(argv)\n',
        "    args = p.parse_args(argv)\n    if args.resume_code_migration and not args.resume:\n"
        "        p.error('resume-code-migration requires an explicit resume checkpoint')\n")
    config_marker="    config['environment_code_sha256'] = contract.hexdigest()\n"
    config_patch='''    config['not_exact_resume']=bool(blob.get('config',{}).get('not_exact_resume',False)) if args.resume else False
    config['code_migration_applied_this_resume']=False
    if args.resume:
        for key in ('code_migration','code_migration_sha256','code_migration_reason'):
            if key in blob['config']:
                config[key]=blob['config'][key]
    if args.resume_code_migration:
        migration=load_resume_code_migration(args.resume_code_migration)
        config.update(code_migration=migration,code_migration_sha256=APPROVED_ENVIRONMENT_MIGRATION_SHA256,
                      code_migration_reason=migration['reason'],not_exact_resume=True,
                      code_migration_applied_this_resume=True)
'''
    source=replace_once(source,config_marker,config_marker+config_patch)
    source=replace_once(source,'            best_validation=best_validation,\n',
        '            best_validation=best_validation,not_exact_resume=config.get("not_exact_resume",False),\n'
        '            code_migration_sha256=config.get("code_migration_sha256"),\n'
        '            code_migration_reason=config.get("code_migration_reason"),\n')
    compile(source,'patched frozen train_defense_grpo.py','exec')
    return source


def environment_hash(members):
    trainer=ast.parse(members['sim/train_defense_grpo.py'].decode())
    assignment=next(node for node in ast.walk(trainer)if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name)and target.id=='contract_files'for target in node.targets))
    files=tuple(ast.literal_eval(assignment.value))
    if files!=CONTRACT_FILES:raise ValueError('Frozen environment hash contract is not the expected twelve files')
    result=hashlib.sha256()
    for name in files:
        result.update(name.encode());result.update(members['sim/'+name])
    return result.hexdigest()


def build_recovery_source(source_package,output,*,expected_archive_sha256=STAGE2_ARCHIVE_SHA256):
    output=Path(output)
    if output.exists():raise ValueError('Recovery output must be a new directory')
    members,original,original_bytes,_=verified_frozen_package(source_package)
    if original['archive_sha256']!=expected_archive_sha256:
        raise ValueError('Recovery source is not the pinned stage2 frozen archive')
    before=dict(members)
    old_code_hash=environment_hash(members)
    members['sim/defense_env.py']=patch_environment(members['sim/defense_env.py'].decode()).encode()
    new_code_hash=environment_hash(members)
    migration=dict(schema='defense-grpo-environment-migration-v1',
        from_environment_code_sha256=old_code_hash,to_environment_code_sha256=new_code_hash,
        not_exact_resume=True,reset_best_validation=True,
        reason='Continue the full scheduled push after a ranged escort engages before reaching the bridge; source semantics changed.',
        changed_environment_files={'sim/defense_env.py':dict(from_sha256=sha(before['sim/defense_env.py']),to_sha256=sha(members['sim/defense_env.py']))})
    migration_bytes=(json.dumps(migration,indent=2,sort_keys=True)+'\n').encode()
    members['sim/train_defense_grpo.py']=patch_trainer(members['sim/train_defense_grpo.py'].decode(),sha(migration_bytes)).encode()
    # The trainer itself is deliberately excluded from the frozen environment hash.
    if environment_hash(members)!=new_code_hash:raise ValueError('Trainer patch changed the frozen environment contract')
    members['data/defense/grpo_code_migration.json']=migration_bytes
    test_path=ROOT/'tests/test_defense_escort_triggers.py'
    members['tests/test_defense_escort_triggers.py']=test_path.read_bytes()
    changed={name for name,data in before.items()if members[name]!=data}
    if changed!={'sim/defense_env.py','sim/train_defense_grpo.py'}:
        raise ValueError('Recovery changed files outside its exact two-file allowlist')
    manifest=deepcopy(original)
    manifest.update(source_archive_sha256=original['archive_sha256'],source_manifest_sha256=sha(original_bytes),
        files={name:sha(data)for name,data in sorted(members.items())},
        recovery_changed_files={name:dict(from_sha256=sha(before[name]),to_sha256=sha(members[name]))for name in sorted(changed)},
        recovery_added_files=sorted(set(members)-set(before)),
        unchanged_original_files=len(before)-len(changed),recovery_migration=migration,
        recovery_migration_sha256=sha(migration_bytes),not_exact_resume=True,
        external_weight_files_included=False,quality_passed=None,
        purpose='Offline preparation only; explicit operator migration required; no recovered weights or training run included')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'source.tar.gz').open('xb')as destination:
        with gzip.GzipFile(fileobj=destination,mode='wb',mtime=0,filename='')as compressed:
            with tarfile.open(fileobj=compressed,mode='w')as stream:
                for name,data in sorted(members.items()):
                    entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o644;entry.mtime=0
                    stream.addfile(entry,io.BytesIO(data))
    manifest['archive_sha256']=sha((output/'source.tar.gz').read_bytes())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (output/'grpo_code_migration.json').write_bytes(migration_bytes)
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    result=build_recovery_source(args.source_package,args.output)
    print(json.dumps({key:result[key]for key in ('archive_sha256','unchanged_original_files',
        'recovery_changed_files','recovery_added_files','recovery_migration_sha256','not_exact_resume')},indent=2))


if __name__=='__main__':main()
