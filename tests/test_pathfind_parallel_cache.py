from concurrent.futures import ThreadPoolExecutor
from sim import pathfind


def test_parallel_cache_eviction_preserves_every_distance_field():
    tasks=[(i%18,(i//18)%32,((3500+i,6500,1600),))for i in range(560)]
    def field(args):return pathfind.distance_field(*args)
    with pathfind._FIELD_CACHE_LOCK:pathfind._FIELD_CACHE.clear()
    expected=list(map(field,tasks))
    with pathfind._FIELD_CACHE_LOCK:pathfind._FIELD_CACHE.clear()
    with ThreadPoolExecutor(8)as pool:actual=list(pool.map(field,tasks))
    assert actual==expected
    assert len(pathfind._FIELD_CACHE)<=512


def test_out_of_arena_goals_also_respect_cache_bound():
    for i in range(530):assert all(v==-1 for row in pathfind.distance_field(30+i,-1)for v in row)
    assert len(pathfind._FIELD_CACHE)<=512
