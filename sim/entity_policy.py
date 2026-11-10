"""Small set-attention actor/critic on public entities, never image pixels.

The legacy trainer's ``planes`` transport key carries [128, 20] entity rows
for this explicitly versioned contract. No spatial raster or convolution.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

CONTRACT = 'joint_public_entities_v1'
ARCHITECTURES = {'entity_transformer_d64_l2_v1': (64, 2, 4),
                 'entity_transformer_d96_l2_v1': (96, 2, 4),
                 'entity_transformer_d64_l2_relative_v1': (64, 2, 4)}
MAX_ENTITIES, FEATURES = 128, 20
VOCAB = Path(__file__).resolve().parents[1] / 'data/entity_policy_vocab.json'


@lru_cache(maxsize=1)
def vocabulary():
    return {name: i + 2 for i, name in enumerate(json.loads(VOCAB.read_text()))}


def vocabulary_sha():
    return hashlib.sha256(VOCAB.read_bytes()).hexdigest()


def encode_entities(match, side=1, *, reflected=False):
    """Only body identity, position, HP and public type traits. No enemy mana,
    targets, cooldowns, future spawns or curriculum labels. Two overflow rows
    preserve per-side excess count/HP rather than silently dropping bodies.
    """
    from sim.env import tactical_plane_values
    result = np.zeros((MAX_ENTITIES, FEATURES), np.float32)
    rows = []
    for e in match.battle.entities.values():
        if not getattr(e, 'alive', True):
            continue
        x, y = float(e.pos.x), float(e.pos.y)
        if side < 0:
            x, y = 18000 - x, 32000 - y
        if reflected:
            x = 18000 - x
        ours = float((e.side > 0) == (side > 0))
        tower = bool(getattr(e, 'is_tower', False))
        traits = (0.,) * 6 if tower else tactical_plane_values(
            damage=getattr(e, 'damage', 0), hit_speed_ms=getattr(e, 'hit_speed_ms', 1000),
            range_mt=getattr(e, 'range_mt', 0), speed_mt_per_sec=getattr(e, 'speed_mt_per_sec', 0),
            attacks_air=getattr(e, 'attacks_air', False),
            target_only_buildings=getattr(e, 'target_only_buildings', False),
            splash_radius_mt=getattr(e, 'splash_radius_mt', 0))
        row = [1., float(vocabulary().get(e.name, 1)), x / 18000, y / 32000,
               2 * ours - 1, float(getattr(e, 'hitpoints', 0)) / 10000,
               float(getattr(e, 'flying', False)), float(getattr(e, 'is_building', False)),
               float(tower), float(e.name == 'king_tower'), *traits,
               float(e.name not in vocabulary()), 0., 1., 0.]
        rows.append(row)
    # Canonical observable ordering only matters for bounded overflow selection.
    # Keep towers, then forward threats. No UID dependence or simulator secrets.
    rows.sort(key=lambda r: (-r[8], -r[3] if r[4] < 0 else r[3], tuple(r)))
    for i, row in enumerate(rows[:MAX_ENTITIES - 2]):
        result[i] = row
    for team, index in ((-1, MAX_ENTITIES - 2), (1, MAX_ENTITIES - 1)):
        rest = [r for r in rows[MAX_ENTITIES - 2:] if r[4] == team]
        if rest:
            result[index, [0, 1, 4, 17, 18]] = [1, 1, team, 1, len(rest)]
            result[index, 2:4] = np.mean([r[2:4] for r in rest], axis=0)
            result[index, 5] = sum(r[5] for r in rest)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite public entity observation')
    return result


class EntityObservationEnv:
    """Keep physics, scalar inputs, masks, rewards and exact snapshots unchanged."""
    def __init__(self, env):
        self.env = env

    def __getattr__(self, key):
        return getattr(self.env, key)

    def convert(self, obs):
        return dict(obs, planes=encode_entities(self.env.match, reflected=self.env.reflected))

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self.convert(obs), info

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        return self.convert(obs), reward, terminal, truncated, info

    def snapshot(self):
        return self.env.snapshot()

    def restore(self, state):
        self.env.restore(state)


class EntityTransformer(nn.Module):
    def __init__(self, num_scalars, num_actions, architecture):
        super().__init__()
        d, layers, heads = ARCHITECTURES[architecture]
        self.architecture = architecture
        self.entity_inputs = True
        self.relative_cells = architecture == 'entity_transformer_d64_l2_relative_v1'
        self.identity = nn.Embedding(len(vocabulary()) + 2, d, padding_idx=0)
        self.features = nn.Linear(FEATURES - 2, d)
        self.scalar = nn.Sequential(nn.Linear(num_scalars, d), nn.GELU())
        self.hand = nn.Linear(8, d)
        self.slot = nn.Parameter(torch.zeros(1, 4, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.,
                                          activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        # TransformerEncoder clones identical initial weights; initialise blocks independently.
        for block in self.encoder.layers:
            for name, p in block.named_parameters():
                if p.ndim > 1:
                    nn.init.xavier_uniform_(p)
        self.norm = nn.LayerNorm(d)
        self.query = nn.Linear(d, d)
        self.cell = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.cell_attention = nn.MultiheadAttention(d, heads, dropout=0., batch_first=True)
        self.hold = nn.Linear(d, 1)
        self.card_bias = nn.Linear(d, 1)
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        extra = num_actions - 2305
        if extra < 0:
            raise ValueError('Expected four placement maps plus WAIT')
        self.abilities = nn.Linear(d, extra) if extra else None
        y, x = torch.meshgrid((torch.arange(32) + .5) / 32,
                              (torch.arange(18) + .5) / 18, indexing='ij')
        self.register_buffer('cell_features', torch.stack((x, y, x*y, x*x, y*y, (x-.5).abs()), -1).reshape(576, 6), persistent=False)

    def forward(self, entities, scalars):
        if entities.ndim != 3 or entities.shape[1:] != (MAX_ENTITIES, FEATURES):
            raise ValueError('Expected versioned [batch,128,20] public entity rows')
        valid = entities[:, :, 0] > 0
        # CPU live inference avoids paying attention to 128 padded rows.
        # Training batches retain fixed shapes; padding remains fully masked.
        if not self.training and entities.device.type == 'cpu' and len(entities) == 1:
            keep = valid[0].nonzero().flatten()
            entities, valid = entities[:, keep], valid[:, keep]
        body = self.identity(entities[:, :, 1].long()) + self.features(entities[:, :, 2:])
        # Sanitize padding; its features/IDs do not supply context.
        body = body.masked_fill(~valid[:, :, None], 0.)
        state = self.scalar(scalars)[:, None]
        hand = self.hand(scalars[:, 7:39].reshape(-1, 4, 8)) + self.slot
        prefix = torch.zeros((len(entities), 5), device=entities.device, dtype=torch.bool)
        pad = torch.cat((prefix, ~valid), dim=1)
        h = self.norm(self.encoder(torch.cat((state, hand, body), dim=1), src_key_padding_mask=pad))
        cells = self.cell(self.cell_features)[None].expand(len(entities), -1, -1)
        if self.relative_cells:
            # Per-head Gaussian distance bias in actual arena tiles. Prefixes
            # retain global context; nearby bodies supply local placement cues.
            # No trainable parameters added: initialization matches the control.
            delta = (self.cell_features[None, :, None, :2] - entities[:, None, :, 2:4])
            delta = delta * entities.new_tensor([18., 32.])
            distance2 = delta.square().sum(-1)
            radii = entities.new_tensor([1.5, 3., 6., 12.])
            bias = (-.5 * distance2[:, None] / radii[None, :, None, None].square()).clamp_min(-50.)
            bias = torch.cat((bias.new_zeros(len(entities), 4, 576, 5), bias), -1)
            bias = bias.masked_fill(pad[:, None, None, :], -torch.inf)
            context, _ = self.cell_attention(cells, h, h,
                attn_mask=bias.flatten(0, 1), need_weights=False)
        else:
            context, _ = self.cell_attention(cells, h, h, key_padding_mask=pad, need_weights=False)
        spatial = cells + context
        maps = torch.einsum('bkd,bnd->bkn', self.query(h[:, 1:5]), spatial) / h.shape[-1]**.5
        maps = maps + self.card_bias(h[:, 1:5])
        parts = [self.hold(h[:, 0]), maps.flatten(1)]
        if self.abilities is not None:
            parts.append(self.abilities(h[:, 0]))
        return torch.cat(parts, -1), self.value(h[:, 0]).squeeze(-1)
