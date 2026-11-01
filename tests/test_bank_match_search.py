import pytest
import torch

from scripts.attack.search_bank_match_actions import action_candidates,branch,robust_labels
from scripts.attack.build_bank_counterfactuals import opening
from sim.env import ACTIONS
from sim.continuous_push import PushCase


class WaitActor(torch.nn.Module):
    def forward(self,planes,scalars):
        logits=torch.zeros((len(planes),ACTIONS));logits[:,0]=1
        return logits,torch.zeros(len(planes))


@pytest.mark.parametrize('delay,delivery',[(1350,1350),(2025,2050)])
@pytest.mark.parametrize('reflected',[False,True])
def test_branch_keeps_real_payment_delay_and_enemy_reinforcements(delay,delivery,reflected):
    env=opening(352,'left',10000,15)
    env.scenario=PushCase('defense','air','random','train')
    env.reflected=reflected
    env.next_enemy=0
    start=env.match.elapsed_ms;bank=env.match.players[1].elixir
    first=next(a for a in action_candidates(env)if a['card']=='cannon')
    row=branch(env,WaitActor(),first,delay,10000)
    assert row['first_command_accepted']
    assert row['own_commands']==[dict(at_ms=start+delivery,card='cannon',
        x=17-first['cell'][0]if reflected else first['cell'][0],y=first['cell'][1])]
    assert row['spent_elixir']==3
    assert row['enemy_commands']
    for command in row['enemy_commands']:
        assert command['elixir_before']-command['elixir_after']==env.cards[command['card']].cost*1000
    assert env.match.elapsed_ms==start and env.match.players[1].elixir==bank and not env.pending


def result(action,delay,damage,outcome='loss',accepted=True):
    return dict(action=action,delay_ms=delay,own_damage=damage,terminal=True,
        first_command_accepted=accepted,metrics=dict(outcome=outcome))


def test_label_needs_strict_full_game_improvement_at_both_delays():
    base=[result(0,1350,100),result(0,2025,100)]
    assert robust_labels(base+[result(1,1350,99),result(1,2025,99)])==[1]
    assert robust_labels(base+[result(1,1350,99),result(1,2025,100)])==[]
    assert robust_labels(base+[result(1,1350,99),result(1,2025,101)])==[]
    assert robust_labels(base+[result(1,1350,0,accepted=False),result(1,2025,0)])==[]
    assert robust_labels(base+[result(1,1350,200,'win'),result(1,2025,200,'win')])==[1]
    with pytest.raises(ValueError,match='both delays'):
        robust_labels(base+[result(1,1350,0)])
    with pytest.raises(ValueError,match='Full match'):
        robust_labels(base+[{**result(1,1350,0),'terminal':False},result(1,2025,0)])
