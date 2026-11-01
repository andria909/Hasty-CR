"""Pair control/treatment full-game validation on identical match contexts."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.attack.select_continuous_focus import validation_score
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha


def summarize(verified,validation,output,markdown):
    verified,validation=Path(verified),Path(validation)
    proof=json.loads(verified.read_text());raw=json.loads(validation.read_text());p=raw['protocol']
    depth=proof.get('schema')=='verified-bank-fit-depth-v1'
    if (proof.get('schema')not in ('verified-bank-correction-pair-v1','verified-bank-fit-depth-v1')or not proof['actual_cuda_verified']or
            proof['supervised_updates_verified']!=(35200 if depth else 6400) or len(proof['runs'])!=8 or
            p['physics_pins']!=physics_pins()or file_sha(validation.parent/'driver.py')!=p['source_sha256']or
            p['split']!='validation'or p['independent_test']or p['live_games']):
        raise ValueError('Verified CUDA pair and current simulator validation required')
    candidates=[r for r in proof['runs']if not depth or r['case']['dataset']=='control']
    if depth and ({r['id']for r in candidates}!={f'mixed{n}_s{s}'for n in (800,8000)for s in (402,403)}or len(candidates)!=4):
        raise ValueError('All four mixed fitting candidates required; isolated policies excluded')
    model_count=5 if depth else 9
    if len(raw['reports'])!=model_count or len(p['models'])!=model_count:raise ValueError('Parent and all compared policies required')
    reports=raw['reports'];parent=reports[0];keys=None;summary=[]
    if parent['model']['sha256']!=proof['initial_sha256']:raise ValueError('Wrong initial-policy reference')
    for index,r in enumerate(reports):
        if r['model']!=p['models'][index]or file_sha(r['model']['checkpoint'])!=r['model']['sha256']:
            raise ValueError('Evaluated checkpoint differs')
        score=validation_score(r)
        observed={(x['episode_seed'],x['match_seed'],x['task'],x['family_id'])for x in r['records']}
        expected={(family,task):6 for family in ('air','bridge','giant','golem','hog','swarm')for task in ('attack','defense')}
        if Counter((x['family_id'],x['task'])for x in r['records'])!=expected:
            raise ValueError('Six attack and six defense games per family required')
        if keys is None:keys=observed
        if observed!=keys or r['score']!=score or r['games']!=72 or r['wins']!=sum(x['outcome']=='win'for x in r['records']):
            raise ValueError('Unpaired or incorrectly scored full games')
        if index==0:continue
        source=candidates[index-1]
        if r['model']['sha256']!=source['checkpoint_sha256']:raise ValueError('Verified candidate order differs')
        summary.append(dict(id=source['id'],case=source['case'],seed=source['seed'],checkpoint_sha256=source['checkpoint_sha256'],
            wins=r['wins'],games=72,score=score,
            empty_spell_elixir_per_game=sum(x['empty_spell_elixir']for x in r['records'])/72,
            overflow_per_game=sum(x['elixir_overflow']for x in r['records'])/72))
    grouped=[]
    for name in sorted({r['case']['id']for r in summary}):
        rows=[r for r in summary if r['case']['id']==name]
        if len(rows)!=2 or {r['seed']for r in rows}!={402,403}:raise ValueError('Both independent training seeds required')
        grouped.append(dict(case=name,wins=sum(r['wins']for r in rows),games=144,
            worst_family=min(r['score']['worst_archetype_points']for r in rows),
            selection_score=sum(r['score']['selection_score']for r in rows)/2,
            all_family_floor_passed=all(r['score']['worst_archetype_points']>=.1 for r in rows)))
    deltas=[]
    for kl in ((None,)if depth else('.01','.1')):
        for seed in (402,403):
            control=next(r for r in summary if r['seed']==seed and (r['case']['updates']==800 if depth else
                r['case']['dataset']=='control'and r['case']['kl_weight']==float(kl)))
            correction=next(r for r in summary if r['seed']==seed and (r['case']['updates']==8000 if depth else
                r['case']['dataset']=='corrections'and r['case']['kl_weight']==float(kl)))
            deltas.append(dict(**({'comparison':'8000_minus_800'}if depth else{'kl':float(kl)}),seed=seed,wins_delta=correction['wins']-control['wins'],
                selection_score_delta=correction['score']['selection_score']-control['score']['selection_score']))
    result=dict(schema='verified-fit-depth-full-games-v1'if depth else'verified-correction-full-games-v1',verified_sha256=file_sha(verified),
        raw_validation_sha256=file_sha(validation),games_verified=model_count*72,unique_match_contexts=72,
        parent_wins=parent['wins'],parent_games=72,runs=summary,groups=grouped,paired_deltas=deltas,
        full_training_accepted=False,scope='Paired simulation validation, not live games or independent test')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Длительность обучения: полные матчи'if depth else'# Парное сравнение исправлений: полные матчи','',
        'Каждая модель играет72одинаковых контекста:6семейств, атака/защита. '
        'В таблице сумма по двум seed обучения;144испытания не означают144независимых сценария. '
        'Задержка1350мс, пушка в центре, текущая физика.','',
        f"Исходные веса: **{parent['wins']}/72** побед.",'',
        '| Вариант | Победы / 144 | Худшее семейство на обоих seed | Порог семейств |',
        '|---|---:|---:|---|']
    for r in grouped:
        lines.append(f"| {r['case']} | {r['wins']} — {r['wins']/144:.1%} | {r['worst_family']:.1%} очков | {'да'if r['all_family_floor_passed']else'нет'} |")
    lines+=['','Парное изменение числа побед относительно контроля:','']
    for r in deltas:
        label='8000 против 800'if depth else f"KL{r['kl']}"
        lines.append(f"- {label}, seed{r['seed']}: {r['wins_delta']:+d}/72.")
    lines+=['','Это validation в симуляторе. Большое обучение и реальные бои автоматически не запускаются.','']
    Path(markdown).write_text('\n'.join(lines));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('verified','validation','output','markdown'):p.add_argument('--'+key,required=True)
    a=p.parse_args();summarize(a.verified,a.validation,a.output,a.markdown)
