"""Publish human-readable independent metrics only after full verification."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import time

from scripts.attack.evaluate_continuous_pool import sha
from scripts.attack.verify_full_candidate_evaluation import verify
from sim.env import DECK_26


def standings(verified, names):
    expected = {frozenset(p) for p in itertools.combinations(names, 2)}
    actual = [frozenset(d['pair']) for d in verified['duels']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Complete unique round robin required')
    totals = {n: Counter() for n in names}; matrix = {n: {} for n in names}
    for duel in verified['duels']:
        a, b = duel['pair']; counts = duel['counts']
        reverse = {k: counts.get(v, 0) for k,v in (('win','loss'),('loss','win'),('draw','draw'))}
        for own, other, values in ((a,b,counts),(b,a,reverse)):
            totals[own].update(values); matrix[own][other] = values
    def score(values):
        games = sum(values.values())
        return (values.get('win',0) + .5 * values.get('draw',0)) / games
    ranked = sorted(names, key=lambda n: (-score(totals[n]), n))
    return dict(totals={n:dict(totals[n]) for n in names}, matrix=matrix, order=ranked,
                points={n:score(totals[n]) for n in names})


def model_labels(protocol, weights):
    names = {}
    for m in protocol['models']:
        if m['role'] == 'baseline': names[m['id']] = m['id']; continue
        proof = json.loads((Path(weights)/m['full_proof_file']).read_text())
        source = next(r for r in proof['models'] if r['id'] == m['origin_model_id'])
        origin = source.get('origin','')
        base = 'Накопление' if 'air-reserve' in origin else 'Пушка + спеллы' if 'air-alternatives' in origin else m['id']
        trial = source.get('trial',{}); replay = trial.get('supervised_replay_weight',0)
        suffix = f'повторение{replay:g}' if replay else f"BC{trial.get('warmstart_updates',0)}"
        names[m['id']] = base + ' · ' + suffix
    if len(set(names.values())) != len(names):
        names = {k:f'{v} ({k})' for k,v in names.items()}
    return names


def render(verified, protocol, labels):
    names = [m['id'] for m in protocol['models']]; table = standings(verified, names)
    lines = ['# Независимые замеры полных PPO', '',
             'Это результаты симулятора, не винрейт в Clash Royale. Два теста имеют разную сложность и показаны отдельно. '
             'Очки = победы + половина ничьих, делённые на число игр. Интервал95% учитывает повторения одного seed. '
             'Лидер таблицы не означает доказанного превосходства в реальной игре.', '']
    for test, models in verified['tests'].items():
        title = 'Основной тест: сценарные пуши' if test == 'scripted_stress' else 'Дополнительный тест: случайный легальный соперник'
        lines += ['## '+title, '', '| Модель | Победы / ничьи / поражения | Очки | Интервал95% |', '|---|---:|---:|---:|']
        for name in names:
            row=models[name]; c=row['summary']['counts']; ci=row['uncertainty']['interval95']
            lines.append(f"| {labels[name]} | {c.get('win',0)} / {c.get('draw',0)} / {c.get('loss',0)} | {row['summary']['points']:.1%} | {ci[0]:.1%}–{ci[1]:.1%} |")
        lines += ['', 'Среднее число подтверждённых выставлений за игру; WAIT не учитывается. Равные частоты карт сами по себе не означают хорошую игру.', '',
                  '| Модель | '+' | '.join(DECK_26)+' |', '|---|'+'---:|'*len(DECK_26)]
        for name in names:
            cards=models[name]['summary']['mean_card_plays']
            lines.append('| '+labels[name]+' | '+' | '.join(f'{cards[c]:.2f}' for c in DECK_26)+' |')
        lines += ['', '| Модель | Потерянный эликсир | Пустые Fireball | Fireball по ≥3 юнитам | Отклонённые команды |', '|---|---:|---:|---:|---:|']
        for name in names:
            s=models[name]['summary']
            lines.append('| '+labels[name]+' | '+' | '.join(f'{s[k]:.2f}' for k in ('mean_elixir_overflow','mean_empty_fireballs','mean_cluster_fireballs','mean_rejected_commands'))+' |')
        lines += ['', '<details><summary>Очки отдельно по задаче и типу пуша</summary>', '',
                  '| Задача / пуш | '+' | '.join(labels[n] for n in names)+' |', '|---|'+'---:|'*len(names)]
        for family in models[names[0]]['by_task_family']:
            lines.append('| '+family+' | '+' | '.join(f"{models[n]['by_task_family'][family]['points']:.1%}" for n in names)+' |')
        lines += ['', '</details>', '']
    lines += ['## Турнир каждого с каждым', '', '| Модель | Победы / ничьи / поражения | Очки | Против v25 |', '|---|---:|---:|---:|---:|']
    for name in table['order']:
        c=table['totals'][name]; opponent=table['matrix'][name].get('v25')
        v25='—' if opponent is None else f"{opponent.get('win',0)} / {opponent.get('draw',0)} / {opponent.get('loss',0)}"
        lines.append(f"| {labels[name]} | {c.get('win',0)} / {c.get('draw',0)} / {c.get('loss',0)} | {table['points'][name]:.1%} | {v25} |")
    lines += ['', 'В последней колонке также победы / ничьи / поражения. Равные очки сохраняют ничью в рейтинге.', '',
              '| Пара | Очки первой модели | Интервал95% |', '|---|---:|---:|']
    for row in verified['duels']:
        a,b=row['pair']; u=row['uncertainty']; lo,hi=u['interval95']
        lines.append(f'| {labels[a]} — {labels[b]} | {u["points"]:.1%} | {lo:.1%}–{hi:.1%} |')
    failed=[labels[m['id']] for m in protocol['models'] if m.get('family_guard') is False]
    lines += ['', 'Не прошли порог validation по отдельным семействам, но сохранены в сравнении: '+(', '.join(failed) if failed else 'нет')+'.', '',
              'Контрольные суммы, обе стороны турнира, расход карт по линиям и потери HP каждой башни находятся в соседнем JSON. '
              'Интервалы не скорректированы на множественные сравнения. Новая модель автоматически не назначается.', '']
    return '\n'.join(lines)


def write_consistent(path, text):
    path=Path(path)
    if path.exists():
        if path.read_text()!=text: raise ValueError('Existing verified report differs; preserve it')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream: stream.write(text)


def finalize(prepared, binding, collected, report):
    prepared,binding,collected,report=map(Path,(prepared,binding,collected,report))
    prep=json.loads((prepared/'preparation.json').read_text())
    receipt=json.loads((collected/'collection_receipt.json').read_text())
    if (receipt.get('kernel')!=prep['kernel'] or receipt.get('remote_status')!='COMPLETE'
            or receipt.get('complete') is not True or receipt.get('transfer_complete') is not True):
        raise ValueError('Complete authoritative evaluation collection required')
    if sha(binding/'protocol.json')!=prep['config']['files']['protocol.json']:
        raise ValueError('Submitted protocol differs')
    manifest=json.loads((collected/'result.json').read_text())
    parts=set(manifest['files'])
    required={'result.json','run_identity.json','source_manifest.json'}|parts|{n+'.receipt.json' for n in parts}
    if set(receipt['files'])!=required:
        raise ValueError('Collection receipt does not cover the exact completed evaluation')
    for name,meta in receipt['files'].items():
        path=collected/name
        if Path(name).name!=name or path.is_symlink() or sha(path)!=meta['sha256']:
            raise ValueError('Collected evaluation artifact changed')
    result=verify(binding/'protocol.json',binding/'weights',collected,
                  expected_source_sha256=prep['config']['source_manifest_sha256'])
    if result['total_games_verified']!=prep['total_games']:
        raise ValueError('Submitted game budget differs')
    protocol=json.loads((binding/'protocol.json').read_text());labels=model_labels(protocol,binding/'weights')
    result.update(model_labels=labels,models=protocol['models'],kernel=prep['kernel'])
    markdown=render(result,protocol,labels)
    write_consistent(report,json.dumps(result,indent=2)+'\n');write_consistent(report.with_suffix('.md'),markdown)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('prepared','binding','collected','report'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    deadline=time.monotonic()+172800
    while True:
        receipt=Path(args.collected)/'collection_receipt.json'
        try: state=json.loads(receipt.read_text()) if receipt.exists() else {}
        except json.JSONDecodeError: state={}
        if state.get('transfer_complete'):
            if state.get('remote_status')!='COMPLETE':
                print(json.dumps(dict(status='terminal_failure_parts_preserved',downloaded_parts=state.get('downloaded_parts',0))));break
            result=finalize(args.prepared,args.binding,args.collected,args.report)
            print(json.dumps(dict(status='independent_metrics_verified',games=result['total_games_verified'])));break
        if not args.watch: raise ValueError('Evaluation collection is not complete')
        if time.monotonic()>=deadline: raise TimeoutError('Local observation expired; remote evaluation unchanged')
        time.sleep(30)
