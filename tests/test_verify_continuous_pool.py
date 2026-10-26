import itertools
import json
from pathlib import Path

import pytest

from scripts.attack.verify_continuous_pool import verify,sha


def fixture(tmp_path,*,duplicate=False,bad_average=False):
    root=tmp_path/'collected';prepared=tmp_path/'prepared';root.mkdir();(prepared/'dataset').mkdir(parents=True)
    ids=['a','b','c','d'];models=[dict(id=n)for n in ids];seeds=list(range(10));files={}
    def write(name,value):
        path=root/name;path.write_text(json.dumps(value));files[name]=sha(path)
    for name in ids:
        rows=[dict(scenario_id=f'{t}::{f}/heldout_random',seed=s,split='test',outcome='win',
                   accepted_card_counts={'hog_rider':1},elixir_overflow=0)
              for t in('attack','defense')for f in('hog','giant','air','golem','swarm','bridge')for s in seeds]
        if duplicate:rows[-1]=dict(rows[0])
        write(name+'_test.json',dict(model={'id':name},rows=rows,counts={'win':120},win_rate=1.,
            mean_card_plays={'hog_rider':2. if bad_average else 1.},mean_elixir_overflow=0))
    for a,b in itertools.combinations(ids,2):
        rows=[dict(seed=s,candidate_side=seat,outcome='draw')for s in seeds for seat in(1,-1)]
        write(a+'__'+b+'_duels.json',dict(pair=[a,b],counts={'draw':20},rows=rows))
    p=prepared/'dataset/protocol.json';p.write_text(json.dumps(dict(models=models,test_seeds=seeds,duel_seeds=seeds)))
    (prepared/'preparation.json').write_text(json.dumps(dict(kernel='owner/frozen')))
    write('result.json',dict(status='complete',protocol_sha256=sha(p),files=dict(files)))
    (root/'collection_receipt.json').write_text(json.dumps(dict(complete=True,remote_status='COMPLETE',
        kernel='owner/frozen',files={n:{'sha256':h}for n,h in files.items()})))
    return root,prepared


def test_six_hundred_game_schedule_and_draw_standings(tmp_path):
    report=verify(*fixture(tmp_path))
    assert report['total_games']==600
    assert all(r['games']==60 and r['score']==.5 for r in report['standings'].values())


def test_duplicate_context_cannot_replace_missing_seed(tmp_path):
    with pytest.raises(ValueError,match='context'):verify(*fixture(tmp_path,duplicate=True))


def test_card_average_recomputed_from_accepted_events(tmp_path):
    with pytest.raises(ValueError,match='averages'):verify(*fixture(tmp_path,bad_average=True))
