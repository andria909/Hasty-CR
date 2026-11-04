"""Separate native elixir shortage from pending-command reservations in saved games."""
import argparse
import json
import math
from pathlib import Path

from scripts.kaggle.local_artifact_transfer import file_sha


def aggregate(records):
    totals=dict(decisions=0,native_below4=0,available_below4=0,native_elixir_sum=0.,available_elixir_sum=0.)
    for record in records:
        row=record['decision_economy'];n=row['decisions']
        if (any(type(row[k])is not int for k in ('decisions','native_below4','available_below4')) or
                not 0<=row['native_below4']<=row['available_below4']<=n or n<=0):
            raise ValueError('Native/available decision counts inconsistent')
        native,available=(row[k]for k in ('native_elixir_sum','available_elixir_sum'))
        if not all(math.isfinite(v)for v in (native,available)) or not 0<=available<=native+1e-6 or native>10*n+1e-6:
            raise ValueError('Native/available bank sums inconsistent')
        for k in totals:totals[k]+=row[k]
    n=totals['decisions']
    if not n:raise ValueError('Observed decisions required')
    return dict(games=len(records),totals=totals,
        native_below4_fraction=totals['native_below4']/n,
        available_below4_fraction=totals['available_below4']/n,
        reservation_only_below4_fraction=(totals['available_below4']-totals['native_below4'])/n,
        mean_native_elixir=totals['native_elixir_sum']/n,
        mean_available_elixir=totals['available_elixir_sum']/n,
        mean_reserved_gap=(totals['native_elixir_sum']-totals['available_elixir_sum'])/n)


def audit(curves_path):
    path=Path(curves_path);curves=json.loads(path.read_text());by_update={i:[]for i in range(50,401,50)}
    evidence=[]
    for study in curves['studies']:
        root=Path(study['root']);receipt_path=root/'collected/collection_receipt.json'
        if file_sha(receipt_path)!=study['receipt_sha256']:raise ValueError('Verified receipt changed')
        receipt=json.loads(receipt_path.read_text())
        if not receipt.get('complete')or receipt['remote_status']!='COMPLETE' or receipt['kernel']!=study['kernel']:
            raise ValueError('Completed matching collection required')
        for run in study['runs']:
            name=run['id']+'/metrics.jsonl';metrics=root/'collected'/name
            digest=file_sha(metrics)
            if digest!=run['metrics_sha256']or digest!=receipt['files'][name]['sha256']:raise ValueError('Verified metrics changed')
            rows=[json.loads(line)for line in metrics.read_text().splitlines()]
            if [r['update']for r in rows]!=list(range(1,401)):raise ValueError('Full training trace required')
            for row in rows:
                if row['update']in by_update:
                    records=row['validation']['records']
                    if len(records)!=72 or any(r['split']!='validation'for r in records):raise ValueError('Development validation required')
                    by_update[row['update']]+=records
            evidence.append(dict(kernel=study['kernel'],run=run['id'],metrics_sha256=digest))
    return dict(schema='native-vs-reserved-economy-v1',source=str(path),source_sha256=file_sha(path),
        evidence=evidence,updates=[dict(update=i,**aggregate(rows))for i,rows in by_update.items()],
        live_games=False,training_changed=False,
        interpretation='Available elixir is native bank minus pending commands, clamped at zero. '
        'Reservation-only shortage is the subset where native bank is at least four but available bank is below four.',
        limitations=['Repeated development contexts and policies; no independent quality estimate',
                    'Aggregate counters do not identify whether each expenditure was tactically useful'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--curves',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();target=Path(a.output)
    if target.exists():raise ValueError('Preserve previous audit')
    target.write_text(json.dumps(audit(a.curves),indent=2)+'\n')
