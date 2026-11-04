"""Compare serialized simulator graphs, including RNG and shared references."""
from dataclasses import fields,is_dataclass
from functools import partial
import random
import types

import numpy as np
import torch


def equal_boundary(a,b):
    left,right={},{}
    def equal(x,y):
        if type(x) is not type(y):return False
        if isinstance(x,torch.Tensor):return torch.equal(x,y)
        if isinstance(x,np.ndarray):return np.array_equal(x,y)
        if isinstance(x,(str,bytes,int,float,bool,type(None),np.generic)):
            return bool(x==y)
        if isinstance(x,(types.FunctionType,types.BuiltinFunctionType,type)):
            return x is y
        if isinstance(x,tuple):return len(x)==len(y)and all(equal(u,v)for u,v in zip(x,y))
        if isinstance(x,(set,frozenset)):return x==y
        # State objects may form cycles (tracker → battle → tracker). Preserve
        # alias relationships too, so two independent RNGs cannot pass as one.
        lx,ry=id(x),id(y)
        if lx in left or ry in right:
            return lx in left and ry in right and left[lx][1] is y and right[ry][1] is x
        # Keep references alive: NumPy RNG state properties create temporary
        # dictionaries whose ids could otherwise be reused later in the walk.
        left[lx]=(x,y);right[ry]=(y,x)
        if isinstance(x,random.Random):return equal(x.getstate(),y.getstate())and equal(vars(x),vars(y))
        if isinstance(x,np.random.Generator):return equal(x.bit_generator.state,y.bit_generator.state)
        if isinstance(x,np.random.RandomState):return equal(x.get_state(),y.get_state())
        if isinstance(x,dict):return x.keys()==y.keys()and all(equal(x[k],y[k])for k in x)
        if isinstance(x,list):return len(x)==len(y)and all(equal(u,v)for u,v in zip(x,y))
        if isinstance(x,types.MethodType):return x.__func__ is y.__func__ and equal(x.__self__,y.__self__)
        if isinstance(x,partial):return equal(x.func,y.func)and equal(x.args,y.args)and equal(x.keywords,y.keywords)
        if hasattr(x,'__dict__'):return equal(vars(x),vars(y))
        if is_dataclass(x):return all(equal(getattr(x,f.name),getattr(y,f.name))for f in fields(x))
        return bool(x==y)
    return equal(a,b)
