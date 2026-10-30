"""TRAIN-only instantaneous gradient alignment; no optimizer or gameplay.

Cosines describe the sampled imitation objective at these exact weights, not
proof of long-term interference or a prediction of game win rate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.attack.sequence_sampling import mixed_skill_probabilities
from scripts.attack.train_air_position_warmstart import successful_set_loss
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.train_attack_ppo import initialize_attack_policy


def audit(data, weights, output):
    torch.set_num_threads(1)
    torch.manual_seed(271)
    data = Path(data)
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest['split'] != 'train' or file_sha(data / 'examples.npz') != manifest['data_sha256']:
        raise ValueError('Pinned TRAIN examples required')
    heldout = max(manifest['seeds'])
    rows = [r for r in manifest['records'] if r['accepted'] and r['seed'] != heldout]
    ids = [r['array_index'] for r in rows]
    probabilities = mixed_skill_probabilities(manifest['records'], ids, .5)
    rng = np.random.default_rng(271)
    batches = {}
    for skill in sorted({r['skill'] for r in rows}):
        p = probabilities * np.array([r['skill'] == skill for r in rows])
        p /= p.sum()
        batches[skill] = rng.choice(ids, size=32, replace=True, p=p).tolist()
    with np.load(data / 'examples.npz', allow_pickle=False) as a:
        tensors = {k: torch.as_tensor(a[k]) for k in a.files}
    results = []
    for path in weights:
        blob = torch.load(path, map_location='cpu', weights_only=False)
        net, _, _ = initialize_attack_policy(blob, device='cpu', num_planes=20,
            num_scalars=309, observation_contract='joint_public_presence_continuous_slots_v2')
        net.eval()
        parameters = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
        gradients = {}
        metrics = {}
        for skill, indices in batches.items():
            logits, _ = net(tensors['planes'][indices], tensors['scalars'][indices])
            loss = successful_set_loss(logits, tensors['masks'][indices], tensors['success_masks'][indices])
            values = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True)
            gradients[skill] = {
                group: torch.cat([g.detach().flatten() for (name, _), g in zip(parameters, values)
                    if g is not None and (group == 'all_actor_parameters' or name.startswith(('conv.', 'scalar.', 'trunk.')))])
                for group in ('all_actor_parameters', 'shared_encoder')}
            metrics[skill] = dict(imitation_loss=float(loss.detach()),
                gradient_norms={g: float(v.norm()) for g, v in gradients[skill].items()})
        alignments = []
        skills = sorted(gradients)
        for i, left in enumerate(skills):
            for right in skills[i + 1:]:
                cosines = {}
                for group in gradients[left]:
                    a, b = gradients[left][group], gradients[right][group]
                    denom = a.norm() * b.norm()
                    cosines[group] = float(torch.dot(a, b) / denom) if denom > 1e-12 else None
                alignments.append(dict(left=left, right=right, cosines=cosines))
        results.append(dict(checkpoint=str(path), checkpoint_sha256=file_sha(path),
            by_skill=metrics, alignments=alignments))
    result = dict(schema='train-skill-gradient-alignment-v1', data_sha256=manifest['data_sha256'],
        source_sha256=file_sha(__file__), heldout_seed_excluded=heldout, batches=batches,
        models=results, optimizer_updates=0, environment_steps=0, quality_accepted=False,
        scope='One deterministic TRAIN batch per skill; instantaneous imitation gradients, without KL/value anchor. Negative cosine is local disagreement, not a causal game-performance result.')
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--weights', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    audit(a.data, a.weights, a.output)
