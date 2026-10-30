"""Paid crown-only spell accounting for optional future economy experiments.

No combat or default reward changes. A crown-only cast is not necessarily bad;
finishing hits are excluded, and troop contacts retain their possible utility.
Dormancy is observed at request, not a claim about causal activation at impact.
"""
from collections import Counter, defaultdict
from sim.spell_waste import SpellWasteTracker


class SpellEconomyTracker(SpellWasteTracker):
    def record(self, play):
        before = len(self.receipts)
        super().record(play)
        if len(self.receipts) == before:
            return
        enemy_king = self.match.towers[-play['side']]['king']
        self.receipts[-1].update(enemy_king_uid=enemy_king.uid,
                                 enemy_king_dormant_at_request=enemy_king.target_only_buildings)

    def metrics(self, side=1):
        result = super().metrics(side)
        receipts = defaultdict(list)
        for receipt in self.receipts:
            receipts[(receipt['spell'], receipt['caster_side'], receipt['cast_ms'])].append(receipt)
        crown = Counter(); dormant = Counter(); finishes = Counter(); troops = Counter()
        for event in self.match.battle.spell_cast_log:
            name = event['spell']
            if name not in ('fireball', 'zap') or event.get('caster_side') != side:
                continue
            receipt = receipts[(name, side, event['cast_ms'])].pop(0)
            if not event.get('resolved'):
                continue
            hits = event['hits']
            if any(not h['is_tower'] for h in hits):
                troops[name] += 1
                continue
            if not hits:
                continue
            if any(h['killed'] for h in hits):
                finishes[name] += 1
                continue
            crown[name] += receipt['cost']
            if (receipt['enemy_king_dormant_at_request'] and
                    all(h['target_uid'] == receipt['enemy_king_uid'] for h in hits)):
                dormant[name] += receipt['cost']
        result.update(nonlethal_crown_only_elixir=sum(crown.values()),
                      nonlethal_crown_only_elixir_by_card=dict(crown),
                      dormant_at_request_king_only_elixir=sum(dormant.values()),
                      dormant_at_request_king_only_elixir_by_card=dict(dormant),
                      crown_finishing_spell_counts=dict(finishes), troop_contact_spell_counts=dict(troops))
        return result
