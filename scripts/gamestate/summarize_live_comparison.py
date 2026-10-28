"""Summarize only completed, counted games; native ACK is not body appearance."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time


def distribution(values):
    if not values:return dict(n=0)
    values=sorted(values)
    def percentile(p):
        pos=(len(values)-1)*p;lo=int(pos);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(pos-lo)
    return dict(n=len(values),mean=statistics.mean(values),median=percentile(.5),
                p90=percentile(.9),p95=percentile(.95),minimum=values[0],maximum=values[-1])


def summarize(root):
    root=Path(root);config=json.loads((root/'config.json').read_text());models=[]
    for model in config['models']:
        path=root/model['label']/'series/series.json'
        if not path.exists():
            models.append(dict(model=model['label'],finished=0));continue
        series=json.loads(path.read_text());games=[];cache={};counts=Counter();lanes=Counter()
        times={k:[] for k in ('parser_request_to_input_return_ms','inference_ms','input_return_to_ack_ms','source_to_ack_game_ms')}
        for match in series['matches']:
            run=match['run'];number=match['local_match']
            if run not in cache:
                cache[run]=[json.loads(line) for line in (Path(run)/'events.jsonl').read_text().splitlines()]
            events=[e for e in cache[run] if e.get('match')==number]
            starts=[e for e in events if e['event']=='MATCH_START'];ends=[e for e in events if e['event']=='MATCH_END']
            if len(starts)!=1 or len(ends)!=1:raise ValueError('Counted game lacks unique start/end')
            if starts[0]['checkpoint_sha256']!=model['sha256']:raise ValueError('Different model in counted game')
            plays={};acknowledged=set();game_counts=Counter()
            for event in events:
                if event['event']=='PLAY':
                    key=(event['card'],event['tick'])
                    if key in plays:raise ValueError('Ambiguous input key')
                    plays[key]=event
                    for name in ('parser_request_to_input_return_ms','inference_ms'):
                        if name in event['timing']:times[name].append(event['timing'][name])
                elif event['event']=='PLAY_ACK':
                    key=(event['card'],event['source_tick'])
                    if key not in plays or key in acknowledged:raise ValueError('ACK without unique matching input')
                    acknowledged.add(key);game_counts[event['card']]+=1
                    lanes['left' if plays[key]['grid'][0]<9 else 'right']+=1
                    for name in ('input_return_to_ack_ms','source_to_ack_game_ms'):
                        if name in event:times[name].append(event[name])
            counts.update(game_counts)
            games.append(dict(number=match['number'],result=match['result'],
                duration_seconds=(ends[0]['host_ns']-starts[0]['host_ns'])/1e9,
                attempted=len(plays),acknowledged=len(acknowledged),unacknowledged=len(plays)-len(acknowledged),
                card_counts=dict(game_counts)))
        if len(games)!=series['finished']:raise ValueError('Series count mismatch')
        n=len(games)
        models.append(dict(model=model['label'],finished=n,wins=series['wins'],losses=series['losses'],unknown=series['unknown'],
            winrate=series['wins']/n if n and not series['unknown'] else None,games=games,
            card_counts=dict(counts),cards_per_match={k:v/n for k,v in counts.items()} if n else {},
            acknowledged_lanes=dict(lanes),timing_ms={k:distribution(v) for k,v in times.items()}))
    return dict(schema='live-comparison-completed-games-v1',created_host_ns=time.time_ns(),models=models,
        limitations=['Only counted completed autonomous games; unfinished match excluded',
                      'Native hand-rotation acknowledgement does not establish unit appearance or spell hit',
                      'Unacknowledged attempts are not necessarily rejected; no spell-quality claim',
                      'Lane uses policy grid column, independent of later unit movement'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    report=summarize(a.root);Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(model=m['model'],finished=m['finished'],cards_per_match=m.get('cards_per_match')) for m in report['models']]))
