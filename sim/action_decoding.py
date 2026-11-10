"""Explicit deterministic deployment rules for a masked flat action policy."""
import torch

MODES = ('flat_argmax', 'grouped_argmax')


def greedy_actions(logits, masks, mode='flat_argmax'):
    if mode not in MODES:
        raise ValueError('Unknown deterministic action mode')
    if logits.ndim != 2 or masks.shape != logits.shape or logits.shape[1] < 2305:
        raise ValueError('Expected batched WAIT, four 576-cell slots and optional abilities')
    if masks.dtype != torch.bool or not bool(masks.any(-1).all()):
        raise ValueError('Every state needs a legal action')
    legal = logits.masked_fill(~masks, -torch.inf)
    if mode == 'flat_argmax':
        return legal.argmax(-1)
    cells = legal[:, 1:2305].reshape(-1, 4, 576)
    groups = torch.cat((legal[:, :1], cells.logsumexp(-1), legal[:, 2305:]), -1)
    group = groups.argmax(-1)
    slot = (group - 1).clamp(0, 3)
    cell = cells[torch.arange(len(logits), device=logits.device), slot].argmax(-1)
    actions = 1 + slot * 576 + cell
    actions = torch.where(group == 0, 0, actions)
    return torch.where(group > 4, 2305 + group - 5, actions)
