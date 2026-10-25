"""Verify every accepted play before reporting average casts per match."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.attack.evaluate_card_usage import sha,card_summary
from sim.eval_defense_duel import summarize,outcome
from sim.env import DECK_26


def verify_model(report,model,protocol):
    if report['model']!=model or report['opponent']!=protocol['opponent']:raise ValueError('Usage model identity differs')
    rows=report['records'];expected={(seed,seat)for seed in protocol['seeds']for seat in protocol['seats']}
    if len(rows)!=len(expected)or {(r['seed'],r['candidate_side'])for r in rows}!=expected:raise ValueError('Usage schedule differs')
    for row in rows:
        seat=row['candidate_side']
        if row['candidate_mode']!=model['mode']or row['baseline_mode']!=protocol['opponent']['mode']:raise ValueError('Usage input modes differ')
        if row['outcome']!=outcome(row.get('result'),seat,row.get('result')in ('top','bottom','draw')):raise ValueError('Usage outcome differs')
        for event in row['accepted_card_events']:
            if event['card']not in DECK_26 or event['side']not in (-1,1)or not 0<=event['at_ms']<=row['elapsed_ms']:
                raise ValueError('Invalid accepted card event')
        for side,field in ((seat,'candidate_cards'),(-seat,'opponent_cards')):
            computed=dict(Counter(e['card']for e in row['accepted_card_events']if e['side']==side))
            if computed!=row[field]or sum(computed.values())!=row['plays'][str(side)]:raise ValueError('Card counts differ from accepted events')
    cards=card_summary(rows);opponent=card_summary(rows,'opponent_cards');summary=summarize(rows)
    if report['cards']!=cards or report['opponent_cards']!=opponent or report['summary']!=summary or report['matches']!=len(rows):
        raise ValueError('Usage aggregate differs')
    return dict(id=model['id'],checkpoint_sha256=model['sha256'],matches=len(rows),summary=summary,cards=cards)


def verify(collected,prepared,live=None):
    root=Path(collected);prepared=Path(prepared);prep=json.loads((prepared/'preparation.json').read_text())
    receipt=json.loads((root/'collection_receipt.json').read_text())
    if receipt.get('kernel')!=prep['kernel']or receipt.get('remote_status')!='COMPLETE'or receipt.get('complete')is not True:
        raise ValueError('Complete terminal usage collection required')
    for name,digest in prep['config']['pins'].items():
        if sha(prepared/'dataset'/name)!=digest:raise ValueError('Prepared usage input changed')
    if sha(prepared/'job/evaluate.py')!=prep['bootstrap_sha256']:raise ValueError('Usage bootstrap changed')
    def checked(name):
        if receipt['files'].get(name,{}).get('sha256')!=sha(root/name):raise ValueError('Collected usage SHA differs')
        return json.loads((root/name).read_text())
    protocol=json.loads((prepared/'dataset/protocol.json').read_text());result=checked('result.json')
    if (result.get('status')!='complete'or result.get('protocol_sha256')!=prep['config']['pins']['protocol.json']or
            result.get('source_sha256')!=protocol['source_sha256']or result.get('models')!=[m['id']for m in protocol['models']]or
            result.get('training_performed')is not False or result.get('test_used_for_selection')is not False or
            result.get('live_games_started')is not False):raise ValueError('Usage result contract differs')
    expected={m['id']+'.json'for m in protocol['models']}
    if set(result['parts'])!=expected:raise ValueError('Missing or extra usage model')
    models=[]
    for model in protocol['models']:
        name=model['id']+'.json';data=checked(name)
        if result['parts'][name]['sha256']!=sha(root/name):raise ValueError('Internal usage part SHA differs')
        models.append(verify_model(data,model,protocol))
    report=dict(schema='verified-card-usage-comparison-v1',protocol_sha256=result['protocol_sha256'],source_sha256=protocol['source_sha256'],
        measurement='Accepted card casts per completed simulation match, not individual spawned bodies',
        simulation_models=models,opponent=protocol['opponent'],geometry=protocol['geometry'],
        matches_per_model=protocol['games_per_model'],quality_accepted=False,correlation_with_live_measured=False)
    if live:
        report['live_reference']=json.loads(Path(live).read_text())
        report['live_note']='Only the original joint economy_soft has completed live matches in this comparison; no live results are imputed to other models.'
    return report


def markdown(report):
    cards=('hog_rider','musketeer','cannon','knight','skeletons','electro_spirit','fireball','zap')
    lines=['# Среднее число розыгрышей карты за матч','',
        'Считаются принятые розыгрыши: Skeletons — одна карта, а не три тела. Нули включены. '+
        'Симуляция: по 20 одинаковых матчей против v25 со сменой сторон; native geometry, Fireball 12 клеток/с, задержка команды 150 мс. '+
        'Это статистика поведения в симуляторе, не прогноз реального винрейта.','']
    live=report.get('live_reference')
    if live:
        lines+=['Реальная игра: исходная PPO joint economy_soft, '+str(live['completed_matches'])+' завершённых матча; '+
            str(live['wins'])+' побед, '+str(live['losses'])+' поражений. Прерванный пятый матч исключён. Подтверждение — смена своей руки.','',
            '| Среда / модель | Hog | Musketeer | Cannon | Knight | Skeletons | Electro Spirit | Fireball | Zap |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
            '| Реальная · joint economy_soft | '+' | '.join(f"{live['per_card'][c]['mean_per_completed_match']:.2f}"for c in cards)+' |','']
    lines+=['| Симуляционная модель | Матчей | Hog | Musketeer | Cannon | Knight | Skeletons | Electro Spirit | Fireball | Zap |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in report['simulation_models']:
        lines.append('| '+model['id']+' | '+str(model['matches'])+' | '+' | '.join(f"{model['cards'][c]['mean_per_match']:.2f}"for c in cards)+' |')
    lines+=['','Разброс (population SD), минимум, максимум, медиана и расход в минуту сохранены в JSON-отчёте. '+
        'По одной модели и четырём реальным матчам статистическую корреляцию между моделями измерить нельзя.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','prepared','output','markdown'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--live',type=Path)
    args=parser.parse_args();report=verify(args.collected,args.prepared,args.live)
    args.output.write_text(json.dumps(report,indent=2)+'\n');args.markdown.write_text(markdown(report))
