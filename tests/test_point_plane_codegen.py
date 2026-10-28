"""The shared factory must retain the independently derived point-plane Jacobian."""
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_factory_import_has_no_optional_runtime_dependency():
    run=subprocess.run([sys.executable,'-c',
        'import sys; from sintergpu.codegen.point_plane import point_plane_expression; '
        'assert not any(n.split(".")[0] in {"numpy","symforce","slangpy","torch"} for n in sys.modules)'],
        capture_output=True,text=True)
    assert run.returncode==0,run.stderr


def test_vector_and_scalar_factory_hand_derived_rotation_first_jacobian():
    sf=pytest.importorskip('symforce.symbolic')
    shared=importlib.import_module('sintergpu.codegen.point_plane')
    # p=(2,3,5),q=(7,11,13),n=(17,19,23): n.(p-q)=-421;
    # derivative for left infinitesimal rotation is p cross n=(-26,39,-13).
    r,j=shared.point_plane_expression(sf,sf.V3(2,3,5),sf.V3(7,11,13),sf.V3(17,19,23))
    assert float(r)==-421
    assert tuple(map(float,j))==(-26,39,-13,17,19,23)
    assert tuple(map(float,shared.point_plane_scalars(sf,2,3,5,7,11,13,17,19,23)))==(-421,-26,39,-13,17,19,23)


def test_owned_vector_and_dense_exports_match_frozen_donor_bytes():
    pytest.importorskip('symforce.symbolic')
    sys.path.insert(0,str(ROOT/'references/donors/caspar-slang/src'))
    from caspar_slang.emitter import emit_slang
    from caspar_slang.gn import residual_kernel as frozen_residual
    from caspar_slang.reprojection import dense_factor_source as frozen_dense
    from tools.codegen.caspar_point_plane import residual_kernel,dense_factor_source
    assert emit_slang(residual_kernel())==emit_slang(frozen_residual())
    assert dense_factor_source()==frozen_dense()


@pytest.mark.parametrize('symbolic_vector', ['point', 'target', 'normal'])
def test_factory_preserves_caller_xi_symbols(symbolic_vector):
    sf=pytest.importorskip('symforce.symbolic')
    shared=importlib.import_module('sintergpu.codegen.point_plane')
    values={'point':(2,3,5),'target':(7,11,13),'normal':(17,19,23)}
    vectors={name:sf.V3(*value) for name,value in values.items()}
    caller_symbols=sf.V6.symbolic('xi')[:3]
    vectors[symbolic_vector]=sf.V3(caller_symbols)
    r,j=shared.point_plane_expression(sf,vectors['point'],vectors['target'],vectors['normal'])
    assert set(caller_symbols)<=r.free_symbols
    substitutions=dict(zip(caller_symbols,values[symbolic_vector]))
    # Independently derived: n.(p-q)=-421; p cross n=(-26,39,-13).
    expected=(-421,-26,39,-13,17,19,23)
    assert tuple(float(v.subs(substitutions)) for v in (r,*j))==expected
    scalar=shared.point_plane_scalars(sf,*vectors['point'],*vectors['target'],*vectors['normal'])
    assert tuple(float(v.subs(substitutions)) for v in scalar)==expected
