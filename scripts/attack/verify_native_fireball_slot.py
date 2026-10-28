"""Additional slot augmentation contract audit on top of the complete existing PPO audit."""
import json
from pathlib import Path
from scripts.attack.verify_continuous_focus import verify


def audit_augmentation(report, trial):
    expected=trial['slot_augmentation']
    if (report.get('slot_augmentation') is not expected or
            report.get('sampling')!='uniform_without_replacement'):
        raise ValueError('Actual native Fireball sampling differs from frozen trial')


def verify_native_fireball(collected, prepared):
    collected,prepared=Path(collected),Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    variant=prep.get('study_variant')
    if variant not in ('native_fireball_slot_v1','fireball_reward_ablation_v1'):raise ValueError('Wrong study variant')
    protocol=json.loads((prepared/'dataset/protocol.json').read_text())
    if variant=='fireball_reward_ablation_v1':
        from scripts.kaggle.prepare_fireball_reward_ablation import validate_protocol
        validate_protocol(protocol)
    # Revalidate all budgets, source/weight hashes, CUDA, split, selection and metrics.
    result=verify(collected,prepared)
    protocol=json.loads((prepared/'dataset/protocol.json').read_text())
    audited=[]
    for trial in protocol['cases']:
        for seed in protocol['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}"
            if trial['warmstart_updates']:
                report=json.loads((collected/(name+'_warmstart')/'result.json').read_text())
                audit_augmentation(report,trial);audited.append(name)
    expected=4 if variant=='fireball_reward_ablation_v1' else 6
    if len(audited)!=expected:raise ValueError('Native Fireball warmstart coverage differs')
    result['slot_augmentation_audit']=dict(runs=audited,verified=True)
    return result
