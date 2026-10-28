"""Summarize verified pilot validation without confusing it with independent test."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.attack.verify_continuous_focus import verify


def summarize(root,verified=None):
    root=Path(root)
    protocol=json.loads((root/'dataset/protocol.json').read_text());rows=[]
    if verified is None:
        variant=protocol.get('study_variant')
        if variant=='own_pending_inputs_v1':
            from scripts.attack.verify_frozen_pending import verify as verify_study
        elif variant=='late_air_sequential_teacher_v1':
            from scripts.attack.verify_sequence_pilot import verify_sequence as verify_study
        elif variant=='native_fireball_slot_v1':
            from scripts.attack.verify_native_fireball_slot import verify_native_fireball as verify_study
        elif variant=='tower_support_reward_v1':
            from scripts.attack.verify_tower_support_pilot import verify_tower_support as verify_study
        elif variant=='spell_waste_reward_v1':
            from scripts.attack.verify_spell_waste_pilot import verify_spell_waste as verify_study
        else:
            verify_study=verify
        verified=verify_study(root/'collected',root)
    history=protocol.get('schema')=='continuous-public-history-pilot-v1'
    board=protocol.get('schema')=='continuous-board-potential-pilot-v1'
    bank=protocol.get('schema')=='continuous-bank-conservation-pilot-v1'
    retention=protocol.get('schema')=='continuous-air-retention-pilot-v1'
    jointskills=protocol.get('schema')=='continuous-joint-skills-pilot-v1'
    warm=protocol.get('schema')=='continuous-air-warmstart-pilot-v1'or retention or jointskills
    if protocol.get('study_variant') in ('own_pending_inputs_v1','tower_support_reward_v1','spell_waste_reward_v1'):warm=False
    temperature=protocol.get('schema')=='continuous-temperature-pilot-v1'
    tactical=protocol.get('schema')=='continuous-live-tactical-reward-pilot-v1'
    exploration=protocol.get('schema')=='continuous-card-exploration-pilot-v1'
    def stats(records):
        counts=Counter(r['outcome']for r in records);cards=Counter()
        for r in records:cards.update(r['accepted_card_counts'])
        return dict(games=len(records),wins=counts['win'],draws=counts['draw'],
            mean_card_lane_plays=({lane:{card:sum(r['accepted_card_lane_counts'][lane].get(card,0)for r in records)/len(records)
                for card in cards}for lane in('left','right')}if all('accepted_card_lane_counts'in r for r in records)else None),
            mean_own_tower_fraction_loss=({lane:sum(r['own_tower_fraction_loss'][lane]for r in records)/len(records)
                for lane in('left','right','king')}if all('own_tower_fraction_loss'in r for r in records)else None),
            mean_missed_fireballs=(sum(r['missed_fireballs']for r in records)/len(records)
                if all('missed_fireballs'in r for r in records)else None),
            mean_card_plays={c:n/len(records)for c,n in sorted(cards.items())},
            by_family={f:dict(games=sum(r['family_id']==f for r in records),
                wins=sum(r['family_id']==f and r['outcome']=='win'for r in records))
                for f in sorted({r['family_id']for r in records})})
    for trial in protocol['cases']:
        records=[];local=[]
        for seed in protocol['pilot_seeds']:
            r=json.loads((root/'collected'/f"pilot_{trial['id']}_s{seed}_summary.json").read_text())
            records.extend(r['validation']['records'])
            if warm:
                from scripts.attack.evaluate_air_position_labels import score_checkpoint
                before=(root/'collected'/f"pilot_{trial['id']}_s{seed}_warmstart/policy.pt"
                    if trial.get('warmstart_updates',0)else root/'dataset/parent.pt')
                local.append(dict(seed=seed,selected_update=r['selected_update'],
                    before=score_checkpoint(before,root/'dataset'),
                    after=score_checkpoint(root/'collected'/r['checkpoint'],root/'dataset')))

        row=dict(id=trial['id'],**stats(records))
        if protocol.get('study_variant')=='spell_waste_reward_v1':
            known=all('empty_spell_counts' in r for r in records)
            minutes=sum(r['elapsed_ms'] for r in records)/60000
            row.update(spell_waste_weight=trial['spell_waste_weight'],
                mean_empty_spells=({card:sum(r['empty_spell_counts'].get(card,0) for r in records)/len(records)
                    for card in ('fireball','zap')} if known else None),
                mean_empty_spell_elixir=sum(r['empty_spell_elixir'] for r in records)/len(records) if known else None,
                empty_spell_elixir_per_minute=sum(r['empty_spell_elixir'] for r in records)/minutes if known and minutes else None)
        if 'action_delay_ms' in trial:
            row.update(action_delay_ms=trial['action_delay_ms'],validation_action_delay_ms=trial['validation_action_delay_ms'])
        if history:row.update(history_frames=trial['history_frames'],focus=trial['focus'])
        if board:row.update(board_potential_weight=trial['board_potential_weight'],focus=trial['focus'])
        if bank:row.update(overflow_penalty=trial['overflow_penalty'],card_balance_weight=trial['card_balance_weight'])
        if warm:
            row.update(warmstart_updates=trial['warmstart_updates'],warmstart_kl_weight=trial['warmstart_kl_weight'],local_diagnostic_by_seed=local)
            if retention:row['supervised_replay_weight']=trial['supervised_replay_weight']
            for phase in('before','after'):
                scores=[r[phase]['local_heldout']for r in local]
                row['local_'+phase+'_ppo']=dict(successful_choices=sum(r['successful_choices']for r in scores),
                    predictions=sum(r['examples']for r in scores),distinct_states=scores[0]['examples'])
            if jointskills:
                row['local_skill_retention']={skill:{phase:dict(
                    successful_choices=sum(r[phase]['by_skill'][skill]['local_heldout']['successful_choices']for r in local),
                    predictions=sum(r[phase]['by_skill'][skill]['local_heldout']['examples']for r in local),
                    distinct_states=local[0][phase]['by_skill'][skill]['local_heldout']['examples'])
                    for phase in('before','after')}for skill in local[0]['before']['by_skill']}
        if temperature:row.update(policy_temperature=trial['policy_temperature'])
        if exploration:row.update(policy_temperature=trial['policy_temperature'],card_exploration_epsilon=trial['card_exploration_epsilon'])
        if tactical or jointskills or exploration:
            row.update(coefficients={k:trial[k]for k in('fireball_value_reward','reserve_potential_weight','own_hp_penalty','overflow_penalty','missed_fireball_penalty')},
                mean_cluster_fireballs=sum(r['cluster_fireballs']for r in records)/len(records),
                mean_fireball_unit_hits=sum(r['fireball_unit_hits']for r in records)/len(records))
        rows.append(row)
    reference=json.loads((root/'collected/parent_reference.json').read_text())
    initial=verified.get('initial_selection',{}).get('selected','parent')
    initial_reference=None
    if initial!='parent':
        source=json.loads((root/'collected'/f'{initial}_reference.json').read_text())
        initial_reference=dict(id=initial,**stats(source['validation']['records']))
    return dict(schema='verified-pending-inputs-pilot-summary-v1'if protocol.get('study_variant')=='own_pending_inputs_v1'else
        'verified-history-pilot-validation-summary-v1'if history else
        'verified-board-pilot-validation-summary-v1'if board else
        'verified-bank-pilot-validation-summary-v1'if bank else
        'verified-joint-skills-pilot-validation-summary-v1'if jointskills else
        'verified-card-exploration-pilot-validation-summary-v1'if exploration else
        'verified-air-retention-pilot-validation-summary-v1'if retention else
        'verified-air-warmstart-pilot-validation-summary-v1'if warm else
        'verified-temperature-pilot-validation-summary-v1'if temperature else
        'verified-live-tactical-reward-pilot-validation-summary-v1'if tactical else'verified-clock-pilot-validation-summary-v1',kernel=verified['kernel'],
        pilot_steps_verified=verified['pilot_steps_verified'],rows=rows,parent=stats(reference['validation']['records']),
        initial_reference=initial_reference,
        comparison_scope=protocol.get('comparison_scope'),
        eligible_for_full=verified['pilot_selection']['gate_met'],quality_accepted=False,
        scope='72 distinct validation contexts replayed by two independently trained seeds; not144 independent contexts',
        test_used_for_selection=False)


def render(summary):
    lines=['# Результаты короткого обучения','',
        f"Проверено CUDA-шагов: {summary['pilot_steps_verified']}. Два тренировочных seed повторяют72 validation-состояния;144 строки не являются144 независимыми контекстами.",'',
        '| Профиль | Победы | Воздух | Голем | Мушкетёр за игру | Hog за игру |',
        '|---|---:|---:|---:|---:|---:|']
    references=[dict(id='Исходник',**summary['parent'])]
    if summary.get('initial_reference'):references.append(summary['initial_reference'])
    for row in references+summary['rows']:
        air=row['by_family']['air'];golem=row['by_family']['golem'];cards=row['mean_card_plays']
        lines.append(f"| {row['id']} | {row['wins']}/{row['games']} | {air['wins']}/{air['games']} | {golem['wins']}/{golem['games']} | {cards.get('musketeer',0):.2f} | {cards.get('hog_rider',0):.2f} |")
    if summary['rows']and 'mean_empty_spells'in summary['rows'][0]:
        lines+=['','## Пустые заклинания','',
            '| Профиль | Пустые Fireball / матч | Пустые Zap / матч | Эликсир впустую / матч | Эликсир впустую / минуту |',
            '|---|---:|---:|---:|---:|']
        for row in summary['rows']:
            if row['mean_empty_spells'] is None:
                lines.append(f"| {row['id']} | {row['mean_missed_fireballs']:.2f} | не измерялось | не измерялось | не измерялось |")
            else:
                lines.append(f"| {row['id']} | {row['mean_empty_spells']['fireball']:.2f} | {row['mean_empty_spells']['zap']:.2f} | {row['mean_empty_spell_elixir']:.2f} | {row['empty_spell_elixir_per_minute']:.2f} |")
        lines+=['','Старый контроль не записывает Zap: отсутствие измерения не означает нулевые промахи. Для полного сравнения расходов нужен повторный прогон его выбранных весов с диагностическим счётчиком на тех же validation-сценах.']
    if summary['rows']and 'local_after_ppo'in summary['rows'][0]:
        lines+=['','## Локальный навык: перед PPO и выбранный PPO-чекпойнт','',
            '| Профиль | До PPO | Выбранный чекпойнт PPO |','|---|---:|---:|']
        for row in summary['rows']:
            before=row['local_before_ppo'];after=row['local_after_ppo']
            lines.append(f"| {row['id']} | {before['successful_choices']}/{before['predictions']} | {after['successful_choices']}/{after['predictions']} |")
        states=summary['rows'][0]['local_after_ppo']['distinct_states']
        lines+=['',f'{states} локальных состояний повторяются для двух seed. Это успешные действия из учебного упражнения, не победы в матчах. Чекпойнт выбирается по полному validation, не по этой таблице.']
    if summary['rows']and 'mean_cluster_fireballs'in summary['rows'][0]:
        lines+=['','## Fireball и обе линии','',
            '| Профиль | Fireball за матч | Пустые | По ≥3 юнитам | Целей за матч | Потери левой / правой башни |',
            '|---|---:|---:|---:|---:|---:|']
        for row in summary['rows']:
            lost=row['mean_own_tower_fraction_loss']
            lines.append(f"| {row['id']} | {row['mean_card_plays'].get('fireball',0):.2f} | {row['mean_missed_fireballs']:.2f} | {row['mean_cluster_fireballs']:.2f} | {row['mean_fireball_unit_hits']:.2f} | {lost['left']:.3f} / {lost['right']:.3f} |")
        lines+=['','Потери — средняя доля исходного HP своей башни. Это не частота атак по линии. Все карты по каждой линии отдельно сохранены в JSON mean_card_lane_plays.']
    if summary['rows']and 'local_skill_retention'in summary['rows'][0]:
        lines+=['','## Каждый локальный навык отдельно','',
            '| Профиль | Навык | До PPO | Выбранный PPO |','|---|---|---:|---:|']
        for row in summary['rows']:
            for skill,phases in row['local_skill_retention'].items():
                before,after=phases['before'],phases['after']
                show=lambda x: f"{x['successful_choices']}/{x['predictions']}" if x['predictions'] else 'нет примеров'
                lines.append(f"| {row['id']} | {skill} | {show(before)} | {show(after)} |")
        lines+=['','Удержание Fireball на пустом поле не компенсирует потерю защиты воздуха или группового прицела. Разделение выполнено по целым seed. Навыки без отложенных примеров здесь не проверены.']
    lines+=['',('Порог для продолжения прошли минимум два профиля.'if summary['eligible_for_full']else
        'Минимум два профиля порог не прошли: большое продолжение этого исследования не запускается.'),
        'Это validation для отбора; test и реальные бои не проводились. Частота карты сама по себе не доказывает качество.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=summarize(a.root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))
