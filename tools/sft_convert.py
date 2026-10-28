"""NNX-value compatibility shim around the unmodified pinned RLinf converter."""
import argparse
from pathlib import Path
import numpy as np
from tools.research_core import atomic


def unwrap_values(tree):
    leaves=[]
    def walk(value,path=()):
        if isinstance(value,dict):
            for key,item in value.items():walk(item,path+(key,))
        else:leaves.append((path,value))
    walk(tree)
    assert leaves and all(path[-1]=='value' and isinstance(value,np.ndarray) for path,value in leaves), 'Unexpected checkpoint leaf convention'
    result={};audit=[]
    for path,value in leaves:
        target=result
        for key in path[:-2]:target=target.setdefault(key,{})
        assert path[-2] not in target
        target[path[-2]]=value
        assert target[path[-2]] is value  # No array copy, cast or numerical change.
        audit.append(dict(before='/'.join(path),after='/'.join(path[:-1]),shape=list(value.shape),dtype=str(value.dtype)))
    return result,audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    args=p.parse_args()
    from rlinf.utils.ckpt_convertor.openpi import jax_to_openpi
    original=jax_to_openpi._load_jax_params
    def load(path):
        result,audit=unwrap_values(original(path))
        atomic(args.target/'nnx_leaf_adapter.json',dict(passed=True,leaf_count=len(audit),leaves=audit,
            operation='strip trailing NNX value path component only; all arrays preserve object identity',
            reference='original pinned OpenPI models/model.py restore_params trailing-value removal'))
        return result
    jax_to_openpi._load_jax_params=load
    jax_to_openpi.convert(args.source,args.source/'assets/physical-intelligence/libero/norm_stats.json',
        args.target,args.target/'physical-intelligence/libero/norm_stats.json',pi05=False,action_horizon=50,max_token_len=48)


if __name__=='__main__':main()
