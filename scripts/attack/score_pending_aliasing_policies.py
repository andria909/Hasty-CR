"""Locate actual feed-forward decisions in the exhaustive local queue search."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.attack.diagnose_pending_action_aliasing import opening, queued_view
from scripts.attack.search_late_air_defense import rank
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.continuous_push import CONTRACT
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck
from sim.train_attack_ppo import initialize_attack_policy


def probability_diagnostic(logits, mask, hand, context, worlds):
    """Exact distribution sums, not sampled wins or an alternative decoder."""
    masked = logits.double().masked_fill(~mask, -torch.inf)
    logp = masked - torch.logsumexp(masked, dim=0)
    prob = logp.exp()
    if not torch.isfinite(prob).all() or abs(float(prob.sum())-1) > 1e-10:
        raise ValueError('Invalid masked action distribution')
    good = context['common_minimum_damage_actions']
    if not good or any(not bool(mask[a]) for a in good):
        raise ValueError('Shared good actions must be nonempty and legal')
    best_good = max(good, key=lambda a: float(masked[a]))
    cards = {'WAIT': float(prob[0])}
    spatial = {}
    for slot, card in enumerate(hand):
        start, end = 1+slot*576, 1+(slot+1)*576
        cards[card] = float(prob[start:end].sum())
        group_good = [a for a in good if start <= a < end]
        log_total = torch.logsumexp(logp[start:end], dim=0)
        spatial[card] = dict(shared_good_cells=len(group_good),
                            legal_cells=int(mask[start:end].sum()),
                            conditional_good_probability=(float(torch.exp(torch.logsumexp(logp[group_good], dim=0)-log_total))
                                                          if group_good and torch.isfinite(log_total) else 0.))
    expected = []
    for i, world in enumerate(worlds):
        expected.append(sum(float(prob[a])*(row['outcome']['total_damage']-context['minimum_damage'][i])
                            for a, row in world.items()))
    return dict(probability_arithmetic='float64 log-softmax of unchanged float32 model logits',
                card_probabilities=cards, conditional_spatial=spatial,
                shared_minimum_hp_probability=float(prob[good].sum()),
                best_shared_hp_action=best_good,
                best_shared_hp_action_rank=1+int((masked > masked[best_good]).sum()),
                best_shared_hp_logit_gap=float(masked.max()-masked[best_good]),
                expected_one_command_hp_regret=expected,
                expected_is_exact_distribution_sum_not_sampled_games=True)


def evaluate(root, checkpoints, output):
    root = Path(root)
    report = json.loads((root/'result.json').read_text())
    if not report['complete'] or report['protocol']['physics_pins'] != physics_pins():
        raise ValueError('Completed search with current physics required')
    for source, sha in report['protocol']['sources'].items():
        if file_sha(source) != sha:
            raise ValueError('Search source changed')
    torch.set_num_threads(1)
    models = []
    for checkpoint in checkpoints:
        path = Path(checkpoint)
        sha = file_sha(path)
        blob = torch.load(path, map_location='cpu', weights_only=False)
        net, _, _ = initialize_attack_policy(blob, device='cpu', num_planes=20,
                                             num_scalars=309, observation_contract=CONTRACT)
        net.eval()
        models.append((path, sha, net))
    cards, spells, *_ = load_defense_assets()
    cards.update(resolve_deck(cards, list(DECK_26)))
    decisions = []
    for context in report['contexts']:
        raw = root/context['raw_file']
        if file_sha(raw) != context['raw_sha256']:
            raise ValueError('Exhaustive outcomes changed')
        worlds = [{}, {}]
        for line in raw.read_text().splitlines():
            row = json.loads(line)
            worlds[row['world']][row['action']] = row
        match, _ = opening(cards, spells, context['lane'], context['support'])
        envs = [queued_view(match, cards, spells, cell) for cell in context['pending_cells']]
        observations = [e._observe() for e in envs]
        digest = hashlib.sha256(b''.join(observations[0][k].tobytes() for k in ('planes', 'scalars'))).hexdigest()
        if digest != context['observation_sha256']:
            raise ValueError('Search observation did not reproduce')
        for path, sha, net in models:
            with torch.inference_mode():
                logits, _ = net(torch.from_numpy(np.stack([o['planes'] for o in observations])),
                                torch.from_numpy(np.stack([o['scalars'] for o in observations])))
                mask = torch.from_numpy(np.stack([e._mask() for e in envs]))
                if not torch.isfinite(logits[mask]).all():
                    raise ValueError('Nonfinite legal logits')
                if not torch.equal(logits[0], logits[1]):
                    raise ValueError('Same observed states gave different feed-forward logits')
                actions = logits.masked_fill(~mask, -torch.inf).argmax(-1).tolist()
            if actions[0] != actions[1] or actions[0] not in worlds[0]:
                raise ValueError('Model action absent from exhaustive legal search')
            outcomes = [worlds[i][actions[i]]['outcome'] for i in range(2)]
            decisions.append(dict(context=context['context'], checkpoint=str(path), checkpoint_sha256=sha,
                                  action=actions[0], helper=worlds[0][actions[0]]['helper'],
                                  same_logits=True, outcomes=outcomes,
                                  hp_regret=[o['total_damage']-context['minimum_damage'][i]
                                             for i, o in enumerate(outcomes)],
                                  optimal_in_local_search=[tuple(rank(o)) == tuple(context['best_rank'][i])
                                                           for i, o in enumerate(outcomes)],
                                  probability_diagnostic=probability_diagnostic(
                                      logits[0], mask[0], envs[0].match.players[1].hand, context, worlds)))
    if any(file_sha(path) != sha for path, sha, _ in models):
        raise ValueError('Checkpoint changed during inference')
    result = dict(schema='pending-aliasing-policy-diagnostic-v1', search=str(root),
                  search_sha256=file_sha(root/'result.json'), source_sha256=file_sha(__file__),
                  decisions=decisions, quality_accepted=False,
                  scope='One command in four constructed TRAIN situations followed by WAIT; not autonomous games or win rates.')
    with Path(output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    evaluate(args.root, args.checkpoint, args.output)
