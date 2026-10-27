"""Audit commands and hand-confirmed card uses in the completed balanced10."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(root):
    root=Path(root);metadata=json.loads(Path('reports/defense_plan/live_balanced_20261027.json').read_text())
    if metadata['final_results']['status']!='complete':raise ValueError('Completed finite series required')
    excluded=set(metadata['manual_assisted_matches']);rows=[];sources={}
    for name,series,offset in [('run','series',0),('recovery_run','recovery_series',metadata['completed_before_recovery'])]:
        events=root/name/'events.jsonl';result_file=root/series/'series.json'
        result=json.loads(result_file.read_text());sources[str(events)]=hashlib.sha256(events.read_bytes()).hexdigest()
        sources[str(result_file)]=hashlib.sha256(result_file.read_bytes()).hexdigest()
        allowed={r['number']:r for r in result['matches']if r['number']+offset not in excluded}
        data={number:dict(global_match=number+offset,result=r['result'],commands=Counter(),confirmed=Counter(),
                         confirmed_by_lane=dict(left=Counter(),right=Counter()))for number,r in allowed.items()}
        plays={};seen=set();starts=set()
        for line in events.read_text().splitlines():
            event=json.loads(line);number=event.get('match')
            if number not in data:continue
            row=data[number]
            if event['event']=='MATCH_START':
                if event['checkpoint_sha256']!=metadata['checkpoint_sha256']:raise ValueError('Different live weights')
                starts.add(number)
            if event['event']=='PLAY':
                key=number,event['card'],event['tick']
                if key in plays:raise ValueError('Ambiguous command identity')
                plays[key]=event;row['commands'][event['card']]+=1
            if event['event']=='PLAY_ACK':
                key=number,event['card'],event['source_tick']
                if key in seen or key not in plays:raise ValueError('Duplicate or unmatched native hand confirmation')
                seen.add(key);row['confirmed'][event['card']]+=1
                lane='left'if plays[key]['grid'][0]<9 else'right'
                row['confirmed_by_lane'][lane][event['card']]+=1
        if starts!=set(data):raise ValueError('Missing model-confirmed match start')
        rows.extend(data.values())
    if len(rows)!=9 or len({r['global_match']for r in rows})!=9:raise ValueError('Nine autonomous matches required')
    cards=sorted({c for r in rows for c in r['commands']})
    totals={c:dict(commands=sum(r['commands'][c]for r in rows),confirmed=sum(r['confirmed'][c]for r in rows),
        mean_confirmed_per_match=sum(r['confirmed'][c]for r in rows)/len(rows),
        confirmed_by_lane={lane:sum(r['confirmed_by_lane'][lane][c]for r in rows)for lane in('left','right')})for c in cards}
    return dict(schema='balanced-live-native-hand-usage-v1',autonomous_matches=9,excluded_manual_matches=sorted(excluded),
        checkpoint_sha256=metadata['checkpoint_sha256'],wins=sum(r['result']=='win'for r in rows),cards=totals,matches=rows,sources=sources,
        confirmation='Native hand rotation correlated to original card and source_tick',
        limitations=['Unconfirmed commands are not proven failed placements; terminal pending commands may lack ACK',
                     'Confirmed hand uses count plays, not individual summons or successful damage',
                     'Lane is target-cell x<9 versus x>=9; lane preference alone does not prove a tactical error'],
        quality_accepted=False,live_games_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Preserve existing audit')
    out.write_text(json.dumps(summarize(a.root),indent=2)+'\n')
