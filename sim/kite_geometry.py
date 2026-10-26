"""Close pull candidates; selection/retargeting still belongs to combat engine."""
import math


def close_kite_cell(x,y):
    """Move at most two cells toward centre and one behind the visible threat.

    The former centre-only placement could sit six cells from a lane troop,
    outside its acquisition range. This does not force building-only units
    to target troops and does not assert every deployed unit will be pulled.
    """
    centre=9 if x<9 else 8
    return round(x+max(-2,min(2,centre-x))),round(max(17,min(26,y+1)))
