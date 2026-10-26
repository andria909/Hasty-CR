"""Readable600-game results and accepted per-card usage from audited artifacts."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

LABELS={'full_balanced_s44':'Контроль · balanced','full_strong_s44':'Контроль · strong',
        'parent':'Исходная joint economy_soft','v25':'v25'}
CARDS=('hog_rider','musketeer','cannon','knight','skeletons','electro_spirit','fireball','zap')


def render(verified,collected):
    verified=Path(verified);root=Path(collected);r=json.loads(verified.read_text())
    if r.get('schema')!='verified-continuous-frozen-pool-v1'or r.get('total_games')!=600:raise ValueError('Verified600-game report required')
    receipt=json.loads((root/'collection_receipt.json').read_text())
    ids=[m['id']for m in r['models']]
    lines=['# 600 симуляционных матчей: контрольные модели','',
        'Обе новые контрольные модели обучались с общей блокировкой ACK. Здесь все четыре модели '+
        'играют через исправленный исполнитель:225мс команда,2100мс обновление только использованного слота,250мс решения. '+
        'Текущие corrected-slot модели ещё обучаются и в эту таблицу не входят.','',
        '480 игр против случайных легальных пушей:120 одинаковых test-контекстов на модель. '+
        'Ещё120 игр — круговой турнир:6пар×20, смена сторон на10seed. Validation не смешана с test.','',
        '| Модель | Случайные пуши: победы | Турнир: победы | Против v25 | Против исходника |',
        '|---|---:|---:|---:|---:|']
    for name in ids:
        test=r['tests'][name];standing=r['standings'][name]
        def pair(other):
            if other==name:return '—'
            c=r['matrix'][name][other];games=sum(c.values())
            return f"{c.get('win',0)}/{games}"
        lines.append(f"| {LABELS[name]} | {test['counts'].get('win',0)}/120 ({100*test['win_rate']:.1f}%) | "+
            f"{standing['wins']}/60 ({100*standing['win_rate']:.1f}%) | {pair('v25')} | {pair('parent')} |")
    lines+=['','## Среднее число принятых розыгрышей за матч со случайными пушами','',
            '| Модель | Hog | Musketeer | Cannon | Knight | Skeletons | Electro Spirit | Fireball | Zap |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ids:
        values=r['tests'][name]['mean_card_plays']
        lines.append('| '+LABELS[name]+' | '+' | '.join(f'{values.get(c,0):.2f}'for c in CARDS)+' |')
    lines+=['','Skeletons — один розыгрыш карты, не число тел. У всех четырёх среднее переполнение эликсира0; '+
        'это не доказывает разумность траты маны.','',
        '## Победы по архетипам случайного противника','',
        '| Модель | air | golem | bridge | giant | swarm | hog |','|---|---:|---:|---:|---:|---:|---:|']
    for name in ids:
        path=root/(name+'_test.json')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['files'][path.name]['sha256']:raise ValueError('Test artifact changed')
        rows=json.loads(path.read_text())['rows'];groups=defaultdict(Counter)
        for row in rows:groups[row['family_id']][row['outcome']]+=1
        lines.append('| '+LABELS[name]+' | '+' | '.join(f"{groups[f]['win']}/{sum(groups[f].values())}"
                     for f in('air','golem','bridge','giant','swarm','hog'))+' |')
    lines+=['','**Вывод:** результаты зависят от соперника; ни одну модель автоматически не продвигаем. '+
        'Повышение частоты карт само по себе не обеспечивает победы. Следующие коэффициенты выбираются по validation, '+
        'а не по этим test-числам. Полная физика и перенос в реальную игру пока не подтверждены.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verified',required=True);p.add_argument('--collected',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();Path(a.output).write_text(render(a.verified,a.collected))
