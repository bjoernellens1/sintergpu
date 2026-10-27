#!/usr/bin/env python3
"""Rebuild the public catalog from imported, revision-bound kernel resources."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'src/sintergpu'
LOCUS_IDS = {
    'yuv_to_rgb': 'image.planar_yuv_to_rgb', 'resize': 'image.resize_nearest',
    'undistort': 'image.undistort', 'invalid_depth': 'depth.validity',
    'bilateral': 'depth.bilateral', 'depth_register': 'depth.register',
    'depth_resolve': 'depth.resolve', 'metrics': 'quality.metrics',
    'reduce': 'quality.reduce', 'backproject': 'geometry.backproject',
    'transform_points': 'geometry.transform_points', 'voxel_insert': 'spatial.voxel_insert',
    'coverage_heatmap': 'geometry.coverage_heatmap', 'coverage_render': 'geometry.coverage_points',
    'normalize': 'image.crop_normalize', 'similarity': 'perception.cosine_similarity',
    'mask_postprocess': 'perception.mask_postprocess',
}
COLOR_IDS = {'nv12Decode':'image.nv12_to_rgb', 'nv12Encode':'image.rgb_to_nv12',
             'yuyvDecode':'image.yuyv_to_rgb', 'remapBilinear':'image.remap_bilinear',
             'resizeBilinear':'image.resize_bilinear', 'fusedYUYVRemap':'image.yuyv_remap_fused'}


def main():
    previous = PKG / 'data/catalog.json'
    operations = []
    for manifest_file in sorted((ROOT / 'provenance').glob('*.json')):
        provenance = json.loads(manifest_file.read_text())
        if not provenance.get('files'):
            continue
        repository = provenance['repository'].removeprefix('https://github.com/').removesuffix('.git')
        donor = repository.split('/')[-1]
        for record in provenance['files']:
            if record.get('dispatchable') is False:
                continue
            path = ROOT / record['destination']
            if not str(path).startswith(str(PKG / 'kernels')) or path.suffix != '.slang':
                continue
            text = path.read_text()
            symbols = {k:int(v) for k,v in re.findall(r'^#define\s+(\w+)\s+(\d+)\s*$', text, re.M)}
            pattern = r'\[numthreads\(([^)]*)\)\]\s*(?:\[[^\]]+\]\s*)*void\s+(\w+)\s*\('
            for threads, entry in re.findall(pattern, text):
                name = LOCUS_IDS[path.stem] if donor == 'locus' else COLOR_IDS[entry] if provenance.get('operation_prefix') == 'image' else provenance.get('operation_prefix', donor.replace('-', '_')) + '.' + path.stem + '.' + entry
                profile = 'locus-float4-abi1' if donor == 'locus' else donor+'-source-compatible'
                behavior = 'Preserve imported source behavior; no cross-profile substitution.'
                documentation = f'https://github.com/bjoernellens1/sintergpu/blob/main/references/donors/{donor}/' + ('gpu/README.md' if donor == 'locus' else 'docs/VALIDATION.md')
                if provenance.get('operation_prefix') == 'image':
                    profile = 'portable-fp32-u8'
                    behavior = provenance['profiles']['portable-fp32']
                    documentation = 'https://github.com/bjoernellens1/sintergpu/blob/main/docs/color.md'
                elif record.get('contract_limits'):
                    profile = 'vkl-forward-partial'
                    behavior = record['contract_limits']
                    documentation = 'https://github.com/bjoernellens1/sintergpu/blob/main/docs/vkl-ports.md'
                elif provenance.get('operation_prefix') == 'primitives':
                    profile = 'segmented-hierarchical-forward'
                    behavior = provenance['differences']
                    documentation = 'https://github.com/bjoernellens1/sintergpu/blob/main/docs/primitives.md'
                operations.append({
                    'id': name, 'path': str(path.relative_to(PKG)),
                    'entrypoint': entry, 'workgroup': [int(v.strip()) if v.strip().isdigit() else symbols.get(v.strip(),v.strip()) for v in threads.split(',')],
                    'variant': 'portable-imported' if not record.get('changes') else 'portable-port',
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'source': {'repository': repository, 'revision': provenance['revision'], 'paths': [record['source_path']], 'license': provenance['license']},
                    'contract': {'profile': profile, 'documentation': documentation, 'numerical_behavior': behavior},
                    'requirements': ['Vulkan compute', 'reflected resource layout', 'caller-owned initialized state and validated extents'],
                    'qualification': {'status': 'experimental', 'records': []},
                    'fixture': path.stem if donor == 'locus' else None,
                })
    # A translated shader may have both a CPU and GPU donor as its provenance.
    # Merge provenance edges only; differing implementations need distinct IDs.
    unique = {}
    for operation in operations:
        previous_op = unique.get(operation['id'])
        if previous_op is None:
            unique[operation['id']] = operation
        else:
            if any(previous_op[k] != operation[k] for k in ('path','entrypoint','sha256','contract')):
                raise ValueError('conflicting operation identity: '+operation['id'])
            previous_op['source']['paths'] = sorted(set(previous_op['source']['paths']+operation['source']['paths']))
    operations = list(unique.values())
    PKG.joinpath('data').mkdir(exist_ok=True)
    previous.write_text(json.dumps({'schema_version': 1, 'operations': sorted(operations, key=lambda o: o['id'])}, indent=2) + '\n')
    PKG.joinpath('data/VERSION').write_bytes(ROOT.joinpath('VERSION').read_bytes())
    modules = {}
    for path in sorted(PKG.joinpath('kernels').rglob('*.slang')):
        module_id = str(path.relative_to(PKG/'kernels').with_suffix('')).replace('/', '.')
        modules[module_id] = {'path': str(path.relative_to(PKG)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    PKG.joinpath('data/modules.json').write_text(json.dumps(modules, indent=2)+'\n')
    print(f'{len(operations)} catalog entries')


if __name__ == '__main__':
    main()
