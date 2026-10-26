"""Summarize verified pilot validation without confusing it with independent test."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.attack.verify_continuous_focus import verify


def summarize(root):
    root=Path(root);verified=verify(root/'collected',root)
    protocol=json.loads((root/'dataset/protocol.json').read_text());rows=[]
    history=protocol.get('schema')=='continuous-public-history-pilot-v1'
    board=protocol.get('schema')=='continuous-board-potential-pilot-v1'
    def stats(records):
        counts=Counter(r['outcome']for r in records);cards=Counter()
        for r in records:cards.update(r['accepted_card_counts'])
        return dict(games=len(records),wins=counts['win'],draws=counts['draw'],
            mean_missed_fireballs=sum(r['missed_fireballs']for r in records)/len(records),
            mean_card_plays={c:n/len(records)for c,n in sorted(cards.items())},
            by_family={f:dict(games=sum(r['family_id']==f for r in records),
                wins=sum(r['family_id']==f and r['outcome']=='win'for r in records))
                for f in sorted({r['family_id']for r in records})})
    for trial in protocol['cases']:
        records=[]
        for seed in protocol['pilot_seeds']:
            r=json.loads((root/'collected'/f"pilot_{trial['id']}_s{seed}_summary.json").read_text())
            records.extend(r['validation']['records'])
        row=dict(id=trial['id'],**stats(records))
        if history:row.update(history_frames=trial['history_frames'],focus=trial['focus'])
        if board:row.update(board_potential_weight=trial['board_potential_weight'],focus=trial['focus'])
        rows.append(row)
    reference=json.loads((root/'collected/parent_reference.json').read_text())
    return dict(schema='verified-history-pilot-validation-summary-v1'if history else
        'verified-board-pilot-validation-summary-v1'if board else'verified-clock-pilot-validation-summary-v1',kernel=verified['kernel'],
        pilot_steps_verified=verified['pilot_steps_verified'],rows=rows,parent=stats(reference['validation']['records']),
        eligible_for_full=verified['pilot_selection']['gate_met'],quality_accepted=False,
        scope='72 distinct validation contexts replayed by two independently trained seeds; not144 independent contexts',
        test_used_for_selection=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=summarize(a.root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))
