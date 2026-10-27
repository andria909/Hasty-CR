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
    if prep.get('study_variant')!='native_fireball_slot_v1':raise ValueError('Wrong study variant')
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
    if len(audited)!=6:raise ValueError('Six native Fireball warmstarts required')
    result['slot_augmentation_audit']=dict(runs=audited,verified=True)
    return result
