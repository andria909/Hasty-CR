"""Additional sampler contract audit on top of the complete existing PPO audit."""
import json
from pathlib import Path
from scripts.attack.verify_continuous_focus import verify


def audit_sampler(report, trial):
    expected=trial['sequence_wait_fraction']
    if (report.get('sequence_wait_fraction')!=expected or
            report.get('sampling')!='equal_sequence_mass_bounded_wait_with_replacement'):
        raise ValueError('Actual sequential sampling differs from frozen trial')


def verify_sequence(collected, prepared):
    collected,prepared=Path(collected),Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant')!='late_air_sequential_teacher_v1':raise ValueError('Wrong study variant')
    # Revalidate all budgets, source/weight hashes, CUDA, split, selection and metrics.
    result=verify(collected,prepared)
    protocol=json.loads((prepared/'dataset/protocol.json').read_text())
    audited=[]
    for trial in protocol['cases']:
        for seed in protocol['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}"
            if trial['warmstart_updates']:
                report=json.loads((collected/(name+'_warmstart')/'result.json').read_text())
                audit_sampler(report,trial);audited.append(name)
    if len(audited)!=6:raise ValueError('Six sequential warmstarts required')
    result['sequential_sampler_audit']=dict(runs=audited,verified=True)
    return result
