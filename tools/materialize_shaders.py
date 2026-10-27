#!/usr/bin/env python3
"""Build independent shader modules from frozen fragments and generated math.

No donor checkout, generator or application runtime is needed. Optional regeneration
of the generated math uses the retained definitions with SymForce0.12.0.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'src/sintergpu'
GENERATED = PKG / 'kernels/generated'


def main():
    recipes = {
        'caspar-slang': {
            'gn': ([], ['gn']),
            'point_plane_residual': (['point_plane_residual'], []),
            'reprojection': (['reprojection_factors'], ['reprojection']),
            'pose_graph': (['pose_graph_factors'], ['pose_graph']),
        },
        'slang-rgbd-slam': {
            'tracker': (['primitives'], ['tracker']),
            'features': (['primitives'], ['features']),
            'tsdf': (['primitives'], ['tsdf']),
            'loop_verification': (['primitives'], ['loop_verification']),
            'dense_ba': (['dense_factors'], ['dense_ba']),
            'dense_tracker_workgroup': (['dense_factors'], ['tracker', 'dense_tracker']),
            'dense_tracker_subgroup': (['dense_factors'], ['tracker', 'dense_tracker']),
        },
    }
    for donor, modules in recipes.items():
        provenance_file = ROOT / 'provenance' / (donor + '.json')
        manifest = json.loads(provenance_file.read_text())
        manifest['files'] = [r for r in manifest['files'] if r.get('generator') != 'tools/materialize_shaders.py']
        for record in manifest['files']:
            if record['destination'].startswith('src/sintergpu/kernels/'):
                record['dispatchable'] = False
        folder = donor.replace('-', '_')
        for name, (prefixes, fragments) in modules.items():
            source_paths, parts = [], []
            defines = {'MATCH_GROUP_SIZE': 64} if name == 'features' else {}
            if name.startswith('dense_tracker_'):
                defines = {'DENSE_GROUP_SIZE': 64, 'DENSE_USE_WAVE': int(name.endswith('subgroup')), 'DENSE_MIN_WAVE_SIZE': 1}
            for key, value in defines.items():
                parts.append(f'#define {key} {value}\n')
            for prefix in prefixes:
                path = GENERATED / (prefix + '.generated.slang')
                parts.append(path.read_text() + '\n')
                source_paths.append(str(path.relative_to(ROOT)))
            for fragment in fragments:
                path = PKG / 'kernels' / folder / (fragment + '.slang')
                parts.append(path.read_text() + '\n')
                source_paths.append(str(path.relative_to(ROOT)))
            text = ''.join(parts)
            destination = PKG / 'kernels' / folder / 'composed' / (name + '.slang')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text)
            manifest['files'].append({'source_path': '+'.join(source_paths), 'destination': str(destination.relative_to(ROOT)),
                'source_sha256': None, 'imported_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'changes': ['prepend retained generated resource-independent math', 'compose shader fragments', 'explicit variant defines'],
                'generator': 'tools/materialize_shaders.py', 'defines': defines, 'inputs': source_paths})
        provenance_file.write_text(json.dumps(manifest, indent=2) + '\n')
    scan = PKG/'kernels/primitives/scan.slang'
    if scan.is_file():
        (scan.parent/'scan_uint.slang').write_text('#define SINTERGPU_SCAN_UINT 1\n'+scan.read_text())
    print('11 independently compilable Caspar/SLAM modules materialized')


if __name__ == '__main__':
    main()
