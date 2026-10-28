"""CPU-only checks for the portable D4/T0 input corpus."""
from collections import Counter
import hashlib
import importlib
import inspect

import numpy as np

from benchmarks.reprojection_fixtures import payloads


def _digest(inputs):
    digest = hashlib.sha256()
    for key, value in sorted(inputs.items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def test_case_counts_and_independent_iteration_storage():
    corpus = list(payloads())
    assert len(corpus) == 113
    assert len({p['case_id'] for p in corpus}) == 63
    assert Counter(p['family'] for p in corpus) == dict(projection=26, raster=21, temporal=43, reference=23)
    counts = Counter(p['case_id'] for p in corpus)
    assert Counter(counts.values()) == {2: 50, 1: 13}
    for first, second in zip(corpus, corpus[1:]):
        if first['case_id'] == second['case_id']:
            assert (first['iteration'], second['iteration']) == (0, 1)
            for key in first['inputs']:
                assert not np.shares_memory(first['inputs'][key], second['inputs'][key])
            changed = 'source_depth' if first['family'] in ('projection', 'raster') else 'target_depth'
            if changed == 'source_depth' or first['inputs'][changed].size > 1:
                assert _digest({changed: first['inputs'][changed]}) != _digest({changed: second['inputs'][changed]})
            for key in first['oracle_kwargs'].keys() & second['oracle_kwargs'].keys():
                if isinstance(first['oracle_kwargs'][key], np.ndarray):
                    assert not np.shares_memory(first['oracle_kwargs'][key], second['oracle_kwargs'][key])


def test_graph_inputs_and_oracle_kwargs():
    for p in payloads(max_workgroups_x=65535):
        required = {key for key, resource in p['graph']['resources'].items() if resource['role'] == 'input'}
        assert p['inputs'].keys() == required
        for key, value in p['inputs'].items():
            resource = p['graph']['resources'][key]
            assert value.dtype == np.dtype(resource['dtype'])
            assert value.size == resource['elements']
            if p['family'] == 'raster' and key == 'target_depth':
                assert key not in p['oracle_kwargs']
                continue
            assert isinstance(p['oracle_kwargs'][key], np.ndarray)
            np.testing.assert_array_equal(value, p['oracle_kwargs'][key])
            assert not np.shares_memory(value, p['oracle_kwargs'][key])
        assert p['oracle_module'] in ('reproject', 'temporal_reproject')
        module = 'reprojection_numeric' if p['oracle_module'] == 'reproject' else 'temporal_reprojection_numeric'
        oracle = getattr(importlib.import_module('sintergpu.references.' + module), p['oracle_function'])
        inspect.signature(oracle).bind(**p['oracle_kwargs'])
        if p['family'] in ('projection', 'raster'):
            assert len(p['oracle_kwargs']['target_hw']) == 2
        elif p['family'] == 'temporal':
            rgb = p['oracle_kwargs']['target_rgb']
            depth = p['inputs']['target_depth']
            assert rgb.shape == (depth.shape[0], 3, *depth.shape[-2:])
            assert rgb.dtype == np.float32
            assert np.count_nonzero(rgb) == 0



# SHA256 of the ORIGINAL GPU test CPU input preparation, in pytest case order.
# Independently obtained by executing preparation-only AST through graph setup,
# then only each original "if iteration" mutation branch. No GPU/device/oracle
# was executed. Original tests/test_vkl_reprojection_gpu.py Git blob:
# 85980b8e85622abd4489fa231eb6d4807d68625b
# Digest includes each sorted input name, dtype, shape, and raw array bytes.
_ORIGINAL_INPUT_SHA256 = """
583c08b86ef9e11fa174213de733ee6f5a805eff1f9477426be12d5977cb2853
7c0c8833532807c6748acaedeba739c175a6bb9397cbba194697f3c235cc88d6
36f4946382b9f2c1c8da562a1e78442b603b6e85c9838f8df7d45a7a74241074
54ead2baadc2b5968ec07860ddffb955f7570b7ce0ac20c4d9fac1b7628103df
520e5472666d351c7af78c94835cbaf2af52ce21334b10f845af78a38b706f7d
26bdb610755471e0382f169552ad7e12ee8f61068287cb2c448e3d91357b9920
75c7c3189910e9fdd2c3e2a87996bc6a8dacdd271293136e2768a9d9fb450bb2
0d77c32936dc8233d298b225c5d6e2a82bc4bc4e9855c751699780a2d2e03655
c26850617904be6dc47cc011745c69a4fd438439a2f6e576eb1405e81eb411e1
1a9a35bc645b738d59bf472196cbd7cd2e82a54c631ee43a09f8fa9806279b07
1e2a3028a69182e364560071ca8234c7a5d15b6600a4ff6685ae71aa13c9d2c2
974578cbcf38acf5bd242dfe8aa393121396e38923f77eac907ea2590ef478d6
f827e476173f771b3f6a84d857bc7333378f689d7276e8358b6b94018237c230
b8fbb600a20090ac5684dd2202774713c94dba3ba882500474c57c8fcf8dc6db
cc307ddb51d62c9e49d5981af91024815f188f07058ce04affb6d3e80a0dd2c8
6193f27548e527eaa21d3d1631103cd43961c3134e6fd1c21a9db1375e387129
e2f18ffd07c92b387fe5cccd0b0c563cea00ed409f8e84868f5282708d62cfe4
b481b258d2c97528dcb92e8873e50c336c2c359dd59383fc451e7accd9b55260
8c35446aa5bce0799d0ed42406e7fcb8dd98b4d0ed7c64cb559c8d4825b05af8
81466ca5669ec5da422ef47bc08225c02d5ac44c2eb5a8a37d1958b013f5e8bd
31d7db86218a9eabfe6b342a52d18fc84ae95f3fad46663e3ffdad742bd7687c
1dc32517a1662f9c20d39faf9bd52949e082641603dedb9263c246d5ba2646f4
e51706af2af8b550deae8d9d7e643f77d77cedccda8d06cfbeccc1f4a9d97cd8
427d62eec4e3d631a5cc76631fd6e62e9b6f1972463d253e2bd1aec0d1f46633
d2877d6a786baf79442736e7d297f8455fe67434415810f7a4d7e1a6dfb50695
55af489ac74251e5dcf0ee3dc055aeb461e7fab98f6da45fc2dac972a0f9dadd
5f07b67cb7eabe3cdcbbf76c1cb09f54030ddb048ae97b7c81f9345599ef3309
e790c321ed0983229935fc9dcf9ffa6b0ef503574b855fcc8cbcadc75138b3ff
a84d48b6ba0f32973c98e1ccd30b38d1daa15a1c3d45d515bffabed6dbcded60
be049c40d032789b87e02fb8e2a0ff90c41104a1b37c44307ad310301f0f532f
ae66fa8647113868e7caf9e57b85209129a40ff8c9d425fc3914e0b40c8769a3
f65c122e42ebeaf3ebcc3f624b741836411afd2168c40266b0e0300d3fc45fda
7e64797db6a6f4691b730c982183b19fc5b196e67cf140b1592215a4fc05f737
b02c9431e7e566e64d8c644cfb8f0144e6099ae770a0b6ca0b60168fafc017d3
022fd2b21c1ed238230ec9223b62caecd697123892c8836e357f63bea683f332
84a34bb59c3f6f4e2f4bacb94970f6d935d3bfac9776be192551c4ff0e1a4dcf
b225d8b982ddb3d1a277017004470a801d2b0e67f1d62f547fe7e28cc2b39549
dc7427dc3a3a6527ecc25b9f2c3336b199190149d385f8c0b3c833572c2f24d3
2befaab8283169f3d03e16b0c156951b8620bbbc13217d0a871332bfea6b7cbb
7a4b4242f57c7f1b2b341f6faf7058cbec1b8ca8ac2e4d060e0d3c6edd176887
99521348cb2f79e595b26240c3e33285959380e0328cecd8715c8542dfaa1be7
19c89ad773f6e5cbb2b50af27683a11bcee5d308a41da33a027255baad33e46f
be8d235bd8a60c154ba190940cab9ba86b760e2fb4ad07f05dfd1ceeb600dd7b
ddc93e809cdada2aeea218c92258f1fb044397e0ceb328cf81eff14fc8509bcc
b8c0618955be3af4fec9a10d0c605bf54f1f412078579cf47daaa3c128053f6d
b8c0618955be3af4fec9a10d0c605bf54f1f412078579cf47daaa3c128053f6d
148658a72724da4d78846e4cb4b8fda4bb05b954e142824e2528dedb8cdd3801
c2e08fe6b9a6854f95302b67f7e7500eb2cb59a1d720978906dd28c8f3d1bd82
2c0d09d3e20927c7fcb70df33400bef87d0c5f0d9a95eec74395ebbff7b678a5
d681db61f85bfd8552d6d9efb66151a99461957fc5ed760a95e2c26a3eeacc8a
29a27421f55400b67c973515959d90edad1dd91f94768605bfb9d5f9960e7a1c
2b27caeb716d5566814be8a32d311856c51e0a5f68509f30d8d1d715ddc3672c
c616e233d8621deb32247ca42c49ff2ae5fdd8ddb984bb179207bc93d2969edd
a35f8481a915d52a44af115c8136de35a3ce68f7ff596f9568a5a402b23ce81a
b777b7a41b93bf732295a33d457d11b5382b624d140c1608e8a9ad21aaece22e
8570292bf9a11715ca3735cb463a0274504fbcd9a1dd9492a5fdb919865b6473
89671e19e31a85ac8a9baa32c3f55ea61ced532b31c688fd235fbb33d38ba15f
4ae2caa345dc097391a957cff706c8b8c29062720dae3a4a5fa6115d985afefa
72f7e154e950cae04696f19719f0044c6a136ab7e1db218279c78d16dcb00f52
87645f6652244816814d9aebee23a105f1cfdc833ff3f065903279b8b1e92b02
db49523c1d4a196f331370f80c9109bce6c6d18e180f8c57e835090ca3d03025
1a9c453578752df16f27d8123edde574647a9d0b35251c6100a96b00f78bbb54
a38c73723bdb5665b943c50a0f0653b57a5a9c27e8409c40f5c62282ad8c4217
7810f7bd1684c4481e2f74ce19465faa9454fe5fbdcac4de3846ae2f18e8d0d2
34f79b9b37d86bc0ed059914906207ebf747bbd106de41908815a52ce9a7b16f
34f79b9b37d86bc0ed059914906207ebf747bbd106de41908815a52ce9a7b16f
169787498d29003af75eb13b5ab95dc556c94a62f18763e1f248d6e3286f0564
70b8c5baa698be94dfb0799250617c0fbbf8550cf8495569deed7b6ce83c5e2d
bddb9eb57aaf976aa8571e9f94d9172558b4a3a9dcef08ab578c3d0b13310695
2fcb657d270b41afcd91e82944689b08596791bb21dd7f465473df0cacc389f1
5a5899416a321fcd920219dc7bc1c38cc1fb1d504dbee800816d3efffbc3ef78
0d0eec72e4b3f2b11860ceb2eeff2b13cf3eddba037b8493f046a48434acdf71
d695bffbc9feff8ad64fb4b1e1c000181b8316765c1b2d79f0d2cfd3ace66b17
db0cd21e2b30752b1a6288b4ac3c5ed5a41671c13388ab6ca55fdb8d60b50821
610ed4ef307dc777a05eac9aa57c5924db879ed927ec27e8b7aa840b9066df6e
30a28e501d2bc33384b1e111bebd67a59cb5844b6d4e4d324b8db504aeb33c00
2ce71d9232986fc3cbf76743d2bc9f0b2ffc071249406378c432e629469b206d
ff364683878c632ef8ee6079351ca54011ed5299eda0564b3edf12b2ffe6a36d
8972ee8ac79f44a5b084d4fb2a4d6b7eb3a6b2cc6cc11a0cffab1b2841b27d5d
fc42716ff6c5deac0f9b328e9160e1ecfcb4cf3a83006c1521be6468550dd514
5a5899416a321fcd920219dc7bc1c38cc1fb1d504dbee800816d3efffbc3ef78
0d0eec72e4b3f2b11860ceb2eeff2b13cf3eddba037b8493f046a48434acdf71
d695bffbc9feff8ad64fb4b1e1c000181b8316765c1b2d79f0d2cfd3ace66b17
db0cd21e2b30752b1a6288b4ac3c5ed5a41671c13388ab6ca55fdb8d60b50821
610ed4ef307dc777a05eac9aa57c5924db879ed927ec27e8b7aa840b9066df6e
30a28e501d2bc33384b1e111bebd67a59cb5844b6d4e4d324b8db504aeb33c00
2ce71d9232986fc3cbf76743d2bc9f0b2ffc071249406378c432e629469b206d
ff364683878c632ef8ee6079351ca54011ed5299eda0564b3edf12b2ffe6a36d
8972ee8ac79f44a5b084d4fb2a4d6b7eb3a6b2cc6cc11a0cffab1b2841b27d5d
fc42716ff6c5deac0f9b328e9160e1ecfcb4cf3a83006c1521be6468550dd514
29a27421f55400b67c973515959d90edad1dd91f94768605bfb9d5f9960e7a1c
2b27caeb716d5566814be8a32d311856c51e0a5f68509f30d8d1d715ddc3672c
c616e233d8621deb32247ca42c49ff2ae5fdd8ddb984bb179207bc93d2969edd
a35f8481a915d52a44af115c8136de35a3ce68f7ff596f9568a5a402b23ce81a
b777b7a41b93bf732295a33d457d11b5382b624d140c1608e8a9ad21aaece22e
8570292bf9a11715ca3735cb463a0274504fbcd9a1dd9492a5fdb919865b6473
89671e19e31a85ac8a9baa32c3f55ea61ced532b31c688fd235fbb33d38ba15f
4ae2caa345dc097391a957cff706c8b8c29062720dae3a4a5fa6115d985afefa
72f7e154e950cae04696f19719f0044c6a136ab7e1db218279c78d16dcb00f52
87645f6652244816814d9aebee23a105f1cfdc833ff3f065903279b8b1e92b02
e18437fe785228946214b5bac7a35a9bc682e1791c590f304b6f4e7e593c7397
c5c740f7aff1868d8c93de0b3566e230cfb8ee798242a84341f0bfc1fe2c1539
9d78c316080f30948fedfea7f2ba89220e9c1fb3d2a88610bd0d6d54fdffebff
944b04a97159fab8e29abb31fc09a6357ec83d16d435f406914c4daf335927cb
91d7429a38157bd07bf892a29a592710e51870cdacc185e57f435981bff0c9a8
54eae22d9d440d8287f709132e09944356afe0ffb5b4b762759f672e1fc722fc
be61a7aca9902c3bfaf9ed953385f1c3ba2d4668253b365500bfae06901ade16
2db9a0ebb0ccd4fb0260d3ec593372f735d62af4597ce69d0a4670a2e6547a36
4cc07aa2f6582c25e28d9ad800e95c400342b415a96730e8925bef4b87fb94f8
e44a3be3d13682c0e002a0c5b86ea9b2570a3d18ab850cfe8680d77d149b6cb5
aaf1939e1295a4022834001b10a6c3513b63eeff35e89942fe7936aa82782a30
60a5d562dec16cbdfe413df6dc2339d6f751ace172439489e0d0619f2c277df3
4d867f1b5192cd8e72e5c28f4f91efc05f17628bef3370a329c8e97a1d3a7547
""".split()


def test_all_input_sha256_match_original_cpu_preparation():
    corpus = list(payloads())
    assert len(_ORIGINAL_INPUT_SHA256) == len(corpus) == 113
    assert [_digest(p['inputs']) for p in corpus] == _ORIGINAL_INPUT_SHA256
