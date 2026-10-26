"""Describe completed TRAIN episodes without selecting weights on test data."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def summary(rows):
    counts=Counter(r['outcome']for r in rows);cards=Counter()
    for row in rows:cards.update(row['accepted_card_counts'])
    return dict(episodes=len(rows),outcomes=dict(counts),win_rate=counts['win']/len(rows)if rows else None,
        mean_elixir_overflow=statistics.mean(r['elixir_overflow']for r in rows)if rows else None,
        mean_accepted_card_plays={c:n/len(rows)for c,n in cards.items()}if rows else {})


def analyze(path,start_update=3000):
    path=Path(path);groups=defaultdict(list);all_rows=[];h=hashlib.sha256();updates=0
    with path.open('rb')as stream:
        for raw in stream:
            h.update(raw);r=json.loads(raw);updates=max(updates,r['update'])
            if r['update']<start_update:continue
            for row in r.get('episodes',[]):
                if row.get('split')!='train':raise ValueError('Diagnostics must use train only')
                all_rows.append(row);groups[row['family_id']].append(row)
    return dict(source=str(path),sha256=h.hexdigest(),updates_seen=updates,start_update=start_update,
        overall=summary(all_rows),by_family={name:summary(rows)for name,rows in sorted(groups.items())},
        scope='Completed train episodes at/after update boundary; not a checkpoint test or live win rate')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--start-update',type=int,default=3000);p.add_argument('--output',required=True)
    a=p.parse_args();r=dict(schema='continuous-training-family-diagnostics-v1',models=[analyze(x,a.start_update)for x in a.paths],
        test_used_for_selection=False,live_validation=False)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({m['source']:m['overall']['win_rate']for m in r['models']}))
