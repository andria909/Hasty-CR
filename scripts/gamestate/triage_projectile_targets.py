"""Reproduce targeting hypotheses without promoting inferred labels to truth.

The input is the read-only projectile audit's local report. Detailed native
snapshots stay in that local artifact; the output contains only derived cases.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.gamestate.audit_projectile_targets import predict_fresh_target
from scripts.gamestate.audit_sim_alignment import unit_profiles


def triage(evidence,path):
    profiles=unit_profiles();rows=[]
    for event in evidence['evidence']:
        if event.get('fresh_prediction_agrees')is not False:continue
        radius=event.get('target_radius_millitiles')
        distance=event['launch_distance_millitiles'];reach=event['source_range_millitiles']
        band=('target_radius_unknown'if radius is None else
              'within_current_reach'if distance<=reach+radius else
              'within_source_radius_extension'if distance<=reach+radius+event['source_radius_millitiles']else
              'beyond_both_reach_hypotheses')
        def assumed_lock(force_rescan):
            target,_=predict_fresh_target(event['launch_bodies'],event['source_id'],profiles,
                prior_target_id=event['inferred_target_id'],force_rescan=force_rescan)
            return target==event['inferred_target_id']
        rows.append(dict(trace_segment=event['trace_segment'],launch_frame=event['launch_frame'],
            impact_frame=event['impact_frame'],source=event['source_name'],
            inferred_target=event['inferred_target_name'],range_band=band,
            launch_distance_millitiles=distance,source_range_millitiles=reach,
            source_radius_millitiles=event['source_radius_millitiles'],target_radius_millitiles=radius,
            source_radius_hypothesis_agrees=event['source_radius_hypothesis_agrees'],
            previous_inferred_lock_agrees=event.get('retained_prediction_agrees'),
            assumed_current_target_lock_can_be_kept=assumed_lock(False),
            assumed_current_target_kept_on_full_rescan=assumed_lock(True)))
    return dict(schema='native-targeting-mismatch-triage-v2',source_sha256=evidence['source_sha256'],
        input_evidence=str(path),comparable=evidence['fresh_acquisition_comparable'],mismatches=len(rows),
        range_bands=dict(Counter(r['range_band']for r in rows)),
        previous_inferred_target_explanations=evidence['fresh_disagreements_explained_by_retaining_previous_inferred_target'],
        lock_rescan_sensitive_cases=sum(r['assumed_current_target_lock_can_be_kept']and
            not r['assumed_current_target_kept_on_full_rescan']for r in rows),rows=rows,
        labels_verified=False,production_mechanics_changed=False,
        limitations=[
            'Native target IDs are not recorded; inferred labels are not a targeting accuracy benchmark.',
            'Birth snapshot follows actual shot launch and is not atomic; movement and sampling phase affect range.',
            'Assuming the inferred victim was already locked is a hypothesis, not measured prior-target evidence.',
            'Full-rescan sensitivity is a reproducible simulator behavior, not proof of real-game target retention.',
            'Do not tune combat mechanics to maximize agreement with these inferred labels.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=triage(json.loads(args.evidence.read_text()),args.evidence)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ('rows','limitations')}))
