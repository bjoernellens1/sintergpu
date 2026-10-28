#!/usr/bin/env python3
"""Reproduce retained Caspar generated expressions (optional SymForce0.12.0)."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'references/donors/caspar-slang/src'))
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if importlib.metadata.version('symforce')!='0.12.0':
        parser.error('regeneration requires pinned SymForce0.12.0; not required to use released shaders')
    from caspar_slang.emitter import emit_slang
    from codegen.caspar_point_plane import residual_kernel,dense_factor_source
    from caspar_slang.primitives import primitives_source
    from caspar_slang.reprojection import reproject_source
    from caspar_slang.pose_graph import pose_graph_source
    outputs={'point_plane_residual':emit_slang(residual_kernel()),
             'primitives':primitives_source(),'reprojection_factors':reproject_source(),
             'dense_factors':dense_factor_source(),'pose_graph_factors':pose_graph_source()}
    expected=json.loads((ROOT/'provenance/generated-math.json').read_text())['generation']['outputs']
    for output in expected:
        path=ROOT/output['path']
        data=outputs[path.name.removesuffix('.generated.slang')].encode()
        if hashlib.sha256(data).hexdigest()!=output['sha256']:
            raise RuntimeError(f'generator output changed: {path.name}; review before updating source')
        if args.check:
            if path.read_bytes()!=data: raise RuntimeError(f'generated artifact differs: {path.name}')
        else:
            path.write_bytes(data)
    print(f'{len(outputs)} generated expression modules reproduced exactly')


if __name__=='__main__': main()
