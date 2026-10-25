"""Finite, paired-seat full-game evaluation of explicit checkpoint baselines.

No training, no scripted defense attacks and no live games. Each pair shares a
seed, swaps candidate seats and retains ordinary Match regulation/overtime.
Observation contracts are explicit per policy; mechanical legal actions and
identical command delay apply to both. Quality thresholds are never asserted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sim import arena
from sim.adapter import grid_to_point
from sim.env import ACTIONS,NUM_PLANES,NUM_SCALARS,DECK_26,_placeable,observe
from sim.match import Match
from sim.selfplay import decode_for
from sim.train_ppo import build_network,checkpoint_architecture,load_policy_state,masked_distribution

MODES=('physical_live','legacy_training','physical_live_cells_v2')
JOINT_MODE='joint_enemy_presence_v1'
CANDIDATE_MODES=MODES+(JOINT_MODE,)


def joint_attack_task(match,side):
    """Fixed public deployment rule, not a learned strategy or future schedule.

    Defend while any living enemy troop/building is visible. Otherwise attack.
    Spell deadlines and training scenario labels do not exist in regular games.
    """
    return not any(getattr(e,'alive',True)and getattr(e,'side',0)==-side
                   and not getattr(e,'is_tower',False)
                   and getattr(e,'object_role','')!='pickup_effect_carrier'
                   for e in match.battle.entities.values())


def policy_observation(match,side,mode):
    if mode not in CANDIDATE_MODES:
        raise ValueError('Explicit known observation mode required')
    if mode=='legacy_training':
        return observe(match,side)
    view=copy(match)
    if mode=='physical_live':
        view.battle=copy(match.battle)
        view.battle.entities={}
        for uid,entity in match.battle.entities.items():
            e=copy(entity);e.pos=arena.Point(entity.pos.x,31000-entity.pos.y)
            view.battle.entities[uid]=e
    # Match the current live parser's unavailable public-history features.
    view.play_log=[]
    if side<0:
        def fractions(target):
            original=match.tower_fractions(target)
            return dict(original,left=original['right'],right=original['left'])
        view.tower_fractions=fractions
    if mode=='physical_live':return observe(view,side)
    result=observe(view,side,coordinate_mode='physical_live_cells_v2')
    if mode==JOINT_MODE:
        result=dict(result,scalars=np.concatenate((result['scalars'],np.array(
            [0.,0.,float(joint_attack_task(match,side))],dtype=np.float32))))
    return result


def mechanical_mask(match,cards,spells,side,pending=False):
    mask=np.zeros(ACTIONS,dtype=bool);mask[0]=True
    if match.finished or pending:
        return mask
    down=tuple(lane for lane in ('left','right')if not match.towers[-side][lane].alive)
    player=match.players[side]
    for slot,card in enumerate(player.hand):
        if not match.can_play(side,card):
            continue
        spell=spells.get(card)
        grid=_placeable(side,down,spell is not None,geometry=match.geometry).copy()
        if spell is not None:
            for y,x in np.argwhere(grid):
                point=grid_to_point(int(x),int(y),side)
                if spell.deploy_own_side_only and ((side>0 and point.y<arena.RIVER_Y)or(side<0 and point.y>=arena.RIVER_Y)):
                    grid[y,x]=False
                if not spell.can_place_on_water and arena.RIVER_TOP<point.y<arena.RIVER_BOTTOM and not arena.on_bridge(point):
                    grid[y,x]=False
        mask[1+slot*576:1+(slot+1)*576]=grid.reshape(-1)
    return mask


def outcome(result,candidate_side,finished):
    if not finished or result not in ('bottom','top','draw'):
        return 'unknown'
    if result=='draw':
        return 'draw'
    winning_side=1 if result=='bottom' else -1
    return 'win' if winning_side==candidate_side else 'loss'


def summarize(rows):
    counts=Counter(row['outcome'] for row in rows)
    known=sum(counts[k] for k in ('win','loss','draw'))
    return dict(matches=len(rows),wins=counts['win'],losses=counts['loss'],draws=counts['draw'],
        unknown=counts['unknown'],win_rate_all_matches=counts['win']/len(rows) if rows else None,
        win_rate_known_matches=counts['win']/known if known else None,
        quality_passed=None,quality_flags=sorted({flag for r in rows for flag in r['quality_flags']}))


def run_match(candidate,baseline,device,candidate_mode,baseline_mode,seed,candidate_side,
              cards,spells,delay_ms=150,max_ms=400000,match_factory=Match):
    match=match_factory(cards=cards,decks=(list(DECK_26),list(DECK_26)),seed=seed,spells=spells)
    networks={candidate_side:candidate,-candidate_side:baseline}
    modes={candidate_side:candidate_mode,-candidate_side:baseline_mode}
    pending={1:None,-1:None}
    plays={1:0,-1:0};rejections={1:0,-1:0}
    # Pair retains this tie ordering so role swapping balances simultaneous input.
    order=(1,-1) if seed%2 else (-1,1)
    next_decision=0
    while not match.finished and match.elapsed_ms<max_ms:
        if match.elapsed_ms>=next_decision:
            for side in order:
                mask=mechanical_mask(match,cards,spells,side,pending[side] is not None)
                observation=policy_observation(match,side,modes[side])
                with torch.no_grad():
                    logits,_=networks[side](torch.as_tensor(observation['planes'],device=device)[None],
                                          torch.as_tensor(observation['scalars'],device=device)[None])
                    support=torch.as_tensor(mask,device=device)[None]
                    if not torch.isfinite(logits[support]).all():
                        raise ValueError('Non-finite legal policy logits')
                    action=int(masked_distribution(logits,support).probs.argmax(-1).item())
                decoded=decode_for(action)
                if decoded is not None:
                    if decoded[0]=='ability':
                        raise ValueError('Ability action outside this explicit eight-card duel contract')
                    slot,x,y=decoded
                    pending[side]=(match.elapsed_ms+delay_ms,match.players[side].hand[slot],x,y)
            next_decision+=500
        for side in order:
            command=pending[side]
            if command is not None and command[0]<=match.elapsed_ms:
                _,card,x,y=command
                ok=match.play_card(side,card,grid_to_point(x,y,side))
                plays[side]+=int(ok);rejections[side]+=int(not ok)
                pending[side]=None
        match.step(min(arena.TICK_MS,max_ms-match.elapsed_ms))
    flags=['simulator_mechanics_not_fully_validated','tick_granular_regular_match_terminal',
           'no_live_game_validation','history_contract_differs' if candidate_mode!=baseline_mode else 'shared_history_contract']
    if match.missing_spawns:flags.append('missing_spawn_profiles')
    if match.battle.unmodelled_projectiles:flags.append('unmodelled_projectiles')
    if not match.finished:flags.append('evaluation_time_cap_unknown_result')
    if candidate_mode==JOINT_MODE:flags.append('joint_task_switch_is_public_heuristic')
    return dict(seed=seed,candidate_side=candidate_side,result=match.result,
        outcome=outcome(match.result,candidate_side,match.finished),elapsed_ms=match.elapsed_ms,
        candidate_mode=candidate_mode,baseline_mode=baseline_mode,
        plays={str(k):v for k,v in plays.items()},rejected_commands={str(k):v for k,v in rejections.items()},
        tower_hp={str(side):{lane:t.hitpoints for lane,t in towers.items()}for side,towers in match.towers.items()},
        missing_spawns=sorted(match.missing_spawns),quality_flags=flags)


def load_network(path,device,mode=None):
    blob=torch.load(path,map_location=device,weights_only=False)
    count=NUM_SCALARS
    if mode==JOINT_MODE:
        from sim.strategy_env import OBSERVATION_CONTRACT
        config=blob.get('config',{})
        if (config.get('task_mode')!='joint'or config.get('observation_contract')!=OBSERVATION_CONTRACT
                or config.get('coordinate_mode')!='physical_live_cells_v2'
                or config.get('geometry')!='legacy_v1'):
            raise ValueError('Joint duel checkpoint contract mismatch')
        count+=3
    incoming=blob['state_dict']['scalar.0.weight'].shape[1]
    allowed=(count,)if mode==JOINT_MODE else (NUM_SCALARS-2,NUM_SCALARS)
    if incoming not in allowed:
        raise ValueError('Duel policy scalar shape differs from explicit mode')
    net=build_network(NUM_PLANES,count,ACTIONS,checkpoint_architecture(blob)).to(device)
    load_policy_state(net,blob['state_dict'])
    return net.eval().requires_grad_(False)


def protocol_rows(report):
    """Adapt raw observed results to the fail-closed frozen acceptance gate.

    Keep the observed outcome separately. Unresolved quality flags produce an
    unknown gate result, never an apparently successful acceptance decision.
    """
    rows=[]
    for baseline in report['baselines']:
        identifier=baseline.get('id',baseline['label'])
        for match in baseline['matches']:
            flags=list(match.get('quality_flags',[]))
            blocking=[flag for flag in flags if flag!='shared_history_contract']
            observed=match['outcome']
            rows.append(dict(baseline_id=identifier,baseline_sha256=baseline['sha256'],
                seed=match['seed'],candidate_side=match['candidate_side'],
                result='unknown' if blocking else observed,observed_result=observed,
                coverage_flags=flags,quality_review_required=bool(blocking),
                candidate_mode=match['candidate_mode'],baseline_mode=match['baseline_mode']))
    return rows


def approved_baselines(manifest,project_root=None,default_mode=None):
    data=json.loads(manifest.read_text())
    if data.get('schema')=='defense-baselines-v1':
        from sim.defense_protocol import load_baseline_allowlist
        project_root=Path(__file__).resolve().parents[1] if project_root is None else Path(project_root)
        allowed=load_baseline_allowlist(manifest,project_root=project_root)
        metadata={row['id']:row for row in data['checkpoints']}
        rows=[dict(row,path=Path(row['path']),label=identifier,approved=True,
                   observation_mode=metadata[identifier].get('observation_mode',default_mode))
              for identifier,row in allowed.items()]
    else:
        rows=data['baselines']
    result=[]
    for row in rows:
        if row.get('approved') is not True:
            continue
        path=Path(row['path'])
        if not path.is_absolute():path=manifest.parent/path
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row['sha256'] or row.get('observation_mode') not in MODES:
            raise ValueError('Approved baseline SHA or explicit observation contract mismatch')
        result.append(dict(row,path=path))
    if not result:raise ValueError('Manifest contains no explicitly approved baselines')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--candidate-mode',choices=CANDIDATE_MODES,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--baseline',type=Path)
    source.add_argument('--baseline-manifest',type=Path)
    p.add_argument('--baseline-mode',choices=MODES)
    p.add_argument('--baseline-sha256')
    p.add_argument('--matches',type=int,default=20,help='Total matches per baseline, even 10..50; seats swapped in pairs')
    p.add_argument('--seed',type=int,default=920000)
    p.add_argument('--device',default='cuda')
    p.add_argument('--delay-ms',type=int,default=150)
    p.add_argument('--max-ms',type=int,default=400000)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if not 10<=args.matches<=50 or args.matches%2 or not 0<=args.delay_ms<=10000 or not 1000<=args.max_ms<=600000:
        p.error('Require even10..50 matches, finite0..10000ms delay and1000..600000ms cap')
    if args.output.exists():p.error('Use a new output file')
    if args.device.startswith('cuda')and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if args.baseline:
        if not args.baseline_mode or not args.baseline_sha256:p.error('Explicit baseline needs --baseline-mode and --baseline-sha256')
        digest=hashlib.sha256(args.baseline.read_bytes()).hexdigest()
        if digest!=args.baseline_sha256:raise ValueError('Explicit baseline SHA mismatch')
        baselines=[dict(path=args.baseline,sha256=digest,observation_mode=args.baseline_mode,label=args.baseline.stem)]
    else:baselines=approved_baselines(args.baseline_manifest,default_mode=args.baseline_mode)
    from sim.defense_assets import load_defense_assets
    from sim.runner import resolve_deck
    cards,spells,_,spell_audit=load_defense_assets();cards.update(resolve_deck(cards,DECK_26))
    device=torch.device(args.device)
    candidate=(load_network(args.candidate,device,args.candidate_mode)if args.candidate_mode==JOINT_MODE
               else load_network(args.candidate,device))
    report=dict(schema='defense-duel-v1',candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        config={k:str(v)if isinstance(v,Path)else v for k,v in vars(args).items()},
        deck=list(DECK_26),spell_asset_audit=spell_audit,
        pairing='same seed, both candidate seats; regular Match; identical delays and mechanical action masks',
        baselines=[],quality_passed=None)
    for baseline in baselines:
        network=load_network(baseline['path'],device)
        rows=[run_match(candidate,network,device,args.candidate_mode,baseline['observation_mode'],
            args.seed+i,seat,cards,spells,args.delay_ms,args.max_ms)
            for i in range(args.matches//2)for seat in (1,-1)]
        report['baselines'].append(dict(id=baseline.get('id',baseline['label']),label=baseline['label'],sha256=baseline['sha256'],
            observation_mode=baseline['observation_mode'],summary=summarize(rows),matches=rows))
    report['head_to_head']=protocol_rows(report)
    report['coverage_complete_for_acceptance']=all(not row['quality_review_required'] and row['result']!='unknown'
        for row in report['head_to_head'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x')as stream:json.dump(report,stream,indent=2,allow_nan=False)


if __name__=='__main__':main()
