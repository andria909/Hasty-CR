"""TRAIN-only successful-action replay to retain a skill during PPO."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sim.env import ACTIONS


def successful_set_loss(logits, legal, success):
    if logits.shape != legal.shape or legal.shape != success.shape or logits.ndim != 2:
        raise ValueError('Action tensor shapes differ')
    if legal.dtype != torch.bool or success.dtype != torch.bool or not success.any(-1).all() or (success & ~legal).any():
        raise ValueError('Every example needs successful legal actions')
    if not torch.isfinite(logits[legal]).all():
        raise ValueError('Nonfinite legal logits')
    return (torch.logsumexp(logits.masked_fill(~legal, -torch.inf), -1) -
            torch.logsumexp(logits.masked_fill(~success, -torch.inf), -1)).mean()


class SuccessfulActionReplay:
    def __init__(self, directory, device):
        directory = Path(directory)
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        raw = directory / 'examples.npz'
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        if manifest['split'] != 'train' or digest != manifest['data_sha256']:
            raise ValueError('Pinned TRAIN examples required')
        with np.load(raw, allow_pickle=False) as arrays:
            count = len(arrays['planes'])
            groups = {r['array_index']: r['seed'] for r in manifest['records'] if r['accepted']}
            if (set(groups) != set(range(count)) or len(groups) != sum(r['accepted'] for r in manifest['records'])
                    or set(groups.values()) != set(manifest['seeds']) or len(set(groups.values())) < 3):
                raise ValueError('Complete distinct TRAIN seed groups required')
            heldout = max(groups.values())
            ids = [i for i in range(count) if groups[i] != heldout]
            self.planes = torch.as_tensor(arrays['planes'][ids], device=device, dtype=torch.float32)
            self.scalars = torch.as_tensor(arrays['scalars'][ids], device=device, dtype=torch.float32)
            self.legal = torch.as_tensor(arrays['masks'][ids], device=device, dtype=torch.bool)
            self.success = torch.as_tensor(arrays['success_masks'][ids], device=device, dtype=torch.bool)
        if self.planes.shape[1:] != (20, 32, 18) or self.scalars.shape != (len(ids), 309):
            raise ValueError('Replay public observation contract differs')
        if self.legal.shape != (len(ids), ACTIONS) or self.success.shape != self.legal.shape:
            raise ValueError('Replay action contract differs')
        if (not torch.isfinite(self.planes).all() or not torch.isfinite(self.scalars).all()
                or not self.success.any(-1).all() or (self.success & ~self.legal).any()):
            raise ValueError('Invalid successful legal TRAIN labels')
        self.metadata = dict(data_sha256=digest, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             train_seeds=sorted(set(groups.values()) - {heldout}), heldout_seed=heldout,
                             train_examples=len(ids), heldout_examples=count-len(ids))

    def loss(self, network, batch_size=32):
        ids = torch.randperm(len(self.planes), device=self.planes.device)[:batch_size]
        logits, _ = network(self.planes[ids], self.scalars[ids])
        return successful_set_loss(logits, self.legal[ids], self.success[ids])
