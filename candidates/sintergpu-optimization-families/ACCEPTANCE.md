# Candidate acceptance: retained incumbents

Later point-plane native evidence is in [NATIVE_ACCEPTANCE.md](NATIVE_ACCEPTANCE.md).
It records a bounded passing configuration on both devices and every earlier
native failure. The iteration3 Python results below retain their original scope
and decisions; they are not replaced by the separate native adapter.

Source revision: `bebbbe235836eb944f36861609d4c47ed3995336`. These results
cover the exact seeded workloads and configurations below. They do not qualify
a range of sizes, production tracking, gradients beyond the stated Jacobian,
installed APIs or consumer migration. No family passes every applicable boundary;
**all incumbents remain available and no promotion/default change is requested.**

Each cell is candidate/baseline median ratio for runs1/2/3, followed by p95
ratios in parentheses. Below1 is faster. Decisions use the declared paired-block
bootstrap bounds, including credible-slowdown rejection, not these point estimates.
Fusion additionally requires an upper median ratio <=0.95 at each relevant
pipeline boundary. A resident-only win never qualifies a CPU-export route.

Final numerical validation: 64 tests on each device. Both Slang2026.12 and
2026.19 compile15 profiles each; cross-compiler reflection and baseline/candidate
resource layouts match. Library CPU suite on the recorded v0.1.9 base:
259 passed,290 skipped,142 subtests, one existing pytest fixture deprecation warning.

Raw samples retain CPU-reference wall, GPU kernel/producer/full-pipeline
timestamps, resident host wall and transfer-inclusive wall separately. The JSON
below additionally records raw baseline/candidate medians and p95s in milliseconds,
each bound, source/compiler/device identity and exact workload hashes.

- [Machine-readable per-configuration acceptance](evidence/iteration3/acceptance.json)
- [Numerical tests on Radeon](evidence/iteration3/numerical.log)
- [Numerical tests on selected RTX4070Ti](evidence/iteration3/rtx4070ti-numerical.log)
- [Pinned compiler/reflection compatibility](evidence/iteration3/compilers/compatibility.json)
- [Baseline/candidate resource layouts](evidence/iteration3/baseline-candidate-reflection.json)

Historical iteration1 and iteration2 rejections are retained with their original
method and source identities. No historical result is relabeled as final.

## radeon8060s

AMD Radeon 8060S Graphics (RADV STRIX_HALO); vendor0x1002/device0x1586; SlangPy0.43.1; precise FP mode.

Raw runs (lossless gzip JSON): [run1](evidence/iteration3/radeon8060s-run-1.json.gz), [run2](evidence/iteration3/radeon8060s-run-2.json.gz), [run3](evidence/iteration3/radeon8060s-run-3.json.gz).

### point_plane

| Profile / exact size | GPU ratio median (p95) | Resident wall ratio median (p95) | Transfer ratio median (p95) | GPU+resident gate / all-boundary gate |
| --- | --- | --- | --- | --- |
| aosoa / 1 | 1.000/1.000/0.984 (0.957/1.060/0.953) | 1.001/0.996/1.000 (0.917/1.025/0.777) | 0.987/0.995/0.990 (1.012/1.043/0.819) | retain / retain |
| scalar / 1 | 0.984/1.000/1.000 (0.987/0.959/0.988) | 0.999/1.001/0.990 (1.174/1.040/1.007) | 1.009/0.999/1.011 (1.227/1.023/0.979) | retain / retain |
| aosoa / 63 | 1.000/1.000/0.985 (1.071/1.025/0.988) | 1.000/0.999/1.000 (1.019/1.002/1.006) | 1.012/0.994/0.992 (1.074/1.038/1.034) | retain / retain |
| scalar / 63 | 1.000/1.015/0.985 (1.001/0.976/1.000) | 0.996/0.997/0.985 (1.047/0.733/0.804) | 0.991/1.015/0.970 (1.059/1.037/1.015) | retain / retain |
| aosoa / 1023 | 1.000/1.014/0.986 (1.001/0.997/1.001) | 1.003/1.001/0.993 (1.079/0.975/0.883) | 0.998/1.005/0.998 (0.980/1.055/1.063) | retain / retain |
| scalar / 1023 | 0.993/1.000/1.000 (1.030/1.011/1.034) | 0.996/0.999/1.002 (0.899/1.027/0.933) | 0.998/1.001/1.009 (0.873/0.947/1.003) | retain / retain |
| aosoa / 1024 | 1.000/1.000/0.986 (0.988/1.011/0.968) | 1.000/1.004/1.002 (1.074/1.004/1.016) | 1.007/0.999/1.004 (0.991/1.122/1.134) | retain / retain |
| scalar / 1024 | 1.000/1.000/1.000 (1.057/1.009/0.989) | 1.002/0.999/0.941 (1.003/1.281/0.922) | 1.003/0.993/1.000 (0.943/0.963/0.945) | retain / retain |
| aosoa / 1025 | 0.986/1.000/1.000 (0.990/1.000/0.977) | 0.997/1.001/0.997 (0.994/1.005/0.955) | 1.005/0.992/1.004 (1.038/0.979/1.003) | retain / retain |
| scalar / 1025 | 0.986/0.986/1.000 (0.981/1.000/1.000) | 1.002/1.001/1.004 (1.027/0.984/0.891) | 1.005/1.006/1.002 (1.045/1.165/1.017) | retain / retain |
| aosoa / 4099 | 1.000/1.000/1.000 (1.000/1.001/0.989) | 1.000/0.999/0.997 (1.009/1.027/1.019) | 1.001/1.005/1.002 (1.019/0.961/1.023) | retain / retain |
| scalar / 4099 | 1.000/1.014/1.000 (0.989/0.951/1.108) | 1.002/1.000/0.994 (1.101/1.028/0.994) | 1.001/0.992/1.012 (1.012/1.038/1.017) | retain / retain |
| aosoa / 65539 | 1.004/1.004/1.004 (1.011/1.079/1.206) | 1.011/1.000/0.999 (0.780/1.007/0.961) | 1.003/1.003/0.999 (0.997/0.992/0.997) | retain / retain |
| scalar / 65539 | 0.982/0.985/1.000 (0.808/0.963/0.987) | 0.996/0.990/0.998 (0.987/0.967/1.007) | 0.999/1.001/0.999 (1.014/1.007/1.003) | retain / retain |

### small_cholesky

| Profile / exact size | GPU ratio median (p95) | Resident wall ratio median (p95) | Transfer ratio median (p95) | GPU+resident gate / all-boundary gate |
| --- | --- | --- | --- | --- |
| caspar6 / 6 | 1.043/0.992/1.034 (1.032/1.052/1.019) | 1.006/1.024/1.002 (1.370/1.002/1.343) | 0.991/0.984/1.001 (0.849/1.033/1.085) | retain / retain |
| schur / 6 | 1.017/1.019/1.020 (1.019/1.017/1.016) | 1.005/1.002/1.004 (1.300/1.219/0.925) | 0.997/1.003/1.010 (1.116/0.887/1.019) | retain / retain |
| schur / 12 | 1.035/1.037/1.034 (1.042/1.039/1.041) | 1.016/1.016/1.017 (0.967/0.975/0.957) | 0.996/1.003/1.010 (0.915/1.073/1.236) | retain / retain |
| schur / 18 | 1.052/1.051/1.051 (1.044/1.047/1.049) | 1.023/1.010/1.031 (0.985/0.988/1.026) | 1.012/1.019/1.029 (0.783/1.017/1.049) | retain / retain |

### point_plane_gn

| Profile / exact size | GPU ratio median (p95) | Resident wall ratio median (p95) | Transfer ratio median (p95) | GPU+resident gate / all-boundary gate |
| --- | --- | --- | --- | --- |
| workgroup / 1 | 0.951/0.955/0.912 (0.942/0.934/0.958) | 0.968/0.985/0.983 (0.736/0.724/0.943) | 0.936/0.915/0.926 (0.898/0.981/0.855) | retain / retain |
| subgroup / 1 | 0.932/0.964/0.947 (0.957/0.946/0.943) | 0.989/0.987/0.979 (0.937/0.782/0.722) | 0.954/0.909/0.957 (0.733/0.869/0.958) | retain / retain |
| workgroup / 63 | 0.904/0.937/0.939 (0.927/0.922/0.928) | 0.929/0.985/0.981 (0.850/0.967/0.902) | 0.900/0.916/0.908 (0.847/0.955/0.936) | retain / retain |
| subgroup / 63 | 0.919/0.945/0.951 (0.941/0.921/0.927) | 0.695/0.980/0.954 (0.947/1.001/0.737) | 0.936/0.916/0.910 (1.035/0.965/0.932) | retain / retain |
| workgroup / 64 | 0.897/0.939/0.942 (0.918/0.924/0.932) | 0.982/0.964/0.981 (0.702/0.893/0.770) | 0.934/0.927/0.911 (1.053/1.085/0.836) | retain / retain |
| subgroup / 64 | 0.954/0.938/0.949 (0.933/0.939/0.930) | 0.974/0.981/0.979 (0.779/0.726/0.884) | 0.906/0.924/0.922 (0.973/0.912/0.773) | retain / retain |
| workgroup / 65 | 0.928/0.938/0.933 (0.965/0.968/0.962) | 0.887/0.705/0.707 (0.708/0.860/0.942) | 0.958/0.956/0.922 (0.929/0.842/0.945) | retain / retain |
| subgroup / 65 | 0.937/0.928/0.924 (0.965/0.946/0.944) | 0.712/0.804/0.738 (0.709/0.830/0.970) | 0.959/0.967/0.963 (1.022/0.876/1.026) | retain / retain |
| workgroup / 1025 | 0.928/0.932/0.962 (0.952/0.956/0.963) | 0.758/0.710/0.711 (0.967/0.948/0.948) | 0.992/0.984/0.971 (0.916/0.964/0.982) | retain / retain |
| subgroup / 1025 | 0.935/0.938/0.941 (0.948/0.955/0.957) | 0.733/0.691/0.690 (0.861/1.004/0.951) | 0.965/0.976/0.933 (0.964/0.980/0.956) | retain / retain |
| workgroup / 4099 | 0.945/0.947/0.946 (0.968/0.966/0.962) | 0.715/0.713/0.717 (0.956/0.858/0.857) | 0.962/0.960/0.954 (0.999/0.999/0.948) | retain / retain |
| subgroup / 4099 | 0.951/0.949/0.962 (0.966/0.961/0.963) | 0.697/0.702/0.972 (0.918/0.926/0.871) | 0.971/0.962/0.970 (1.059/0.987/1.025) | retain / retain |
| workgroup / 65539 | 0.879/0.846/0.874 (0.844/0.852/0.844) | 0.751/0.745/0.743 (0.850/0.814/0.864) | 0.998/0.990/1.001 (0.985/0.980/0.996) | PASS / retain |
| subgroup / 65539 | 0.853/0.845/0.839 (0.851/0.840/0.837) | 0.753/0.906/0.906 (0.828/0.878/0.727) | 0.992/0.998/0.996 (0.978/1.011/0.994) | PASS / retain |

## rtx4070ti

NVIDIA GeForce RTX 4070 Ti; vendor0x10de/device0x2782; SlangPy0.43.1; precise FP mode.

Raw runs (lossless gzip JSON): [run1](evidence/iteration3/rtx4070ti-run-1.json.gz), [run2](evidence/iteration3/rtx4070ti-run-2.json.gz), [run3](evidence/iteration3/rtx4070ti-run-3.json.gz).

Canonical host `cps-wkstn-nvidia4070`, GPU UUID
`GPU-96d2dba0-95fd-5c00-e849-2e0610f24e06`, driver595.99.02.
The earlier unreachable RTX3090 is historical and was replaced by explicit user steering.

### point_plane

| Profile / exact size | GPU ratio median (p95) | Resident wall ratio median (p95) | Transfer ratio median (p95) | GPU+resident gate / all-boundary gate |
| --- | --- | --- | --- | --- |
| aosoa / 1 | 1.000/1.000/1.000 (1.000/1.000/1.000) | 0.999/1.001/1.000 (1.040/0.974/1.007) | 1.001/1.000/1.001 (0.998/1.000/1.000) | retain / retain |
| scalar / 1 | 1.000/1.000/1.000 (1.000/1.000/1.000) | 0.997/0.999/1.000 (0.982/1.022/1.013) | 1.000/0.999/0.999 (1.001/1.001/0.997) | retain / retain |
| aosoa / 63 | 1.000/1.000/1.000 (1.000/1.000/1.000) | 0.998/0.996/0.998 (0.846/1.014/0.974) | 0.999/0.999/1.000 (1.001/1.001/1.002) | retain / retain |
| scalar / 63 | 1.000/1.000/1.000 (1.000/1.000/1.000) | 0.997/0.996/0.998 (1.001/1.000/0.998) | 1.001/0.998/0.998 (1.001/0.998/0.999) | retain / retain |
| aosoa / 1023 | 1.000/1.000/1.000 (1.000/1.000/1.000) | 0.999/0.997/0.996 (1.019/1.006/0.988) | 1.001/1.000/0.998 (1.001/0.995/0.997) | retain / retain |
| scalar / 1023 | 1.000/1.000/1.004 (1.000/1.000/1.000) | 1.000/0.998/0.997 (0.975/0.883/0.995) | 0.999/1.000/1.003 (0.999/0.999/1.001) | retain / retain |
| aosoa / 1024 | 1.000/1.000/1.000 (1.000/1.000/1.000) | 1.000/0.999/1.000 (0.981/1.093/1.021) | 1.000/1.000/0.998 (0.998/1.000/0.995) | retain / retain |
| scalar / 1024 | 1.007/1.000/1.000 (1.000/1.000/1.000) | 0.999/0.999/0.999 (1.099/1.012/0.988) | 1.000/1.001/0.999 (1.000/0.998/0.999) | retain / retain |
| aosoa / 1025 | 1.003/0.993/1.000 (1.000/1.000/1.000) | 0.998/1.000/0.999 (0.976/1.010/1.231) | 0.999/1.000/0.999 (0.999/1.000/0.998) | retain / retain |
| scalar / 1025 | 1.000/1.007/1.000 (1.000/1.000/1.000) | 0.999/0.998/1.001 (0.975/1.109/1.026) | 1.000/1.000/0.999 (0.998/1.002/0.998) | retain / retain |
| aosoa / 4099 | 1.000/1.000/1.000 (1.000/1.000/1.000) | 0.996/0.999/0.995 (1.010/0.976/0.804) | 0.999/1.000/0.999 (0.998/0.937/0.998) | retain / retain |
| scalar / 4099 | 1.000/1.000/1.000 (1.006/1.007/1.000) | 0.999/0.997/1.000 (1.024/1.008/1.018) | 1.000/1.000/0.999 (1.000/1.002/0.997) | retain / retain |
| aosoa / 65539 | 1.024/1.024/1.024 (1.027/1.027/1.027) | 1.011/1.011/1.013 (0.984/1.012/1.008) | 0.975/0.965/1.023 (0.990/0.983/1.003) | retain / retain |
| scalar / 65539 | 0.978/0.978/0.985 (0.973/0.976/0.904) | 0.991/0.991/0.990 (0.996/0.944/0.990) | 1.017/1.009/1.004 (1.024/0.984/0.886) | PASS / retain |

### small_cholesky

| Profile / exact size | GPU ratio median (p95) | Resident wall ratio median (p95) | Transfer ratio median (p95) | GPU+resident gate / all-boundary gate |
| --- | --- | --- | --- | --- |
| caspar6 / 6 | 1.097/1.097/1.097 (1.101/1.101/1.101) | 1.048/1.047/1.046 (1.022/1.054/1.035) | 0.916/1.005/1.005 (0.892/0.992/1.000) | retain / retain |
| schur / 6 | 0.985/0.987/0.996 (1.000/1.000/0.986) | 0.993/0.996/0.995 (0.994/1.094/0.995) | 0.999/0.999/0.999 (0.999/0.999/0.999) | retain / retain |
| schur / 12 | 0.986/0.983/0.988 (0.988/0.987/0.983) | 0.987/0.989/0.989 (0.907/0.980/0.992) | 0.998/0.997/0.997 (0.997/0.991/0.998) | PASS / PASS |
| schur / 18 | 0.982/0.981/0.983 (0.982/0.982/0.983) | 0.985/0.986/0.985 (0.969/0.987/0.984) | 0.998/0.999/0.998 (0.995/0.998/0.999) | PASS / PASS |

### point_plane_gn

| Profile / exact size | GPU ratio median (p95) | Resident wall ratio median (p95) | Transfer ratio median (p95) | GPU+resident gate / all-boundary gate |
| --- | --- | --- | --- | --- |
| workgroup / 1 | 0.690/0.690/0.690 (0.703/0.698/0.703) | 0.762/0.762/0.764 (0.723/0.772/0.743) | 0.981/0.981/0.981 (0.975/0.979/0.980) | PASS / retain |
| subgroup / 1 | 0.936/0.936/0.936 (0.936/0.939/0.937) | 0.867/0.866/0.859 (0.913/0.864/0.861) | 0.986/0.986/0.987 (0.986/0.982/0.989) | PASS / retain |
| workgroup / 63 | 0.688/0.688/0.680 (0.693/0.693/0.683) | 0.762/0.761/0.767 (0.759/0.767/0.746) | 0.978/0.979/0.981 (0.977/0.988/0.976) | PASS / retain |
| subgroup / 63 | 0.925/0.928/0.927 (0.924/0.918/0.927) | 0.859/0.856/0.859 (0.873/0.848/0.873) | 0.987/0.986/0.987 (0.991/0.984/0.984) | PASS / retain |
| workgroup / 64 | 0.695/0.702/0.702 (0.708/0.706/0.706) | 0.767/0.770/0.772 (0.913/0.779/0.833) | 0.982/0.982/0.981 (0.982/0.992/0.984) | PASS / retain |
| subgroup / 64 | 0.914/0.917/0.919 (0.927/0.912/0.920) | 0.852/0.853/0.856 (0.802/0.948/0.835) | 0.986/0.986/0.988 (0.991/0.988/0.988) | PASS / retain |
| workgroup / 65 | 0.732/0.744/0.740 (0.741/0.750/0.754) | 0.791/0.805/0.805 (0.812/0.846/0.833) | 0.984/0.986/0.984 (0.995/1.014/1.007) | PASS / retain |
| subgroup / 65 | 0.936/0.937/0.938 (0.936/0.937/0.933) | 0.884/0.883/0.880 (0.867/0.919/0.891) | 0.989/0.990/0.989 (0.998/0.996/0.997) | PASS / retain |
| workgroup / 1025 | 0.678/0.680/0.697 (0.686/0.684/0.701) | 0.769/0.769/0.779 (0.761/0.795/0.759) | 0.981/0.982/0.982 (0.983/0.985/1.033) | PASS / retain |
| subgroup / 1025 | 0.881/0.875/0.875 (0.875/0.881/0.881) | 0.857/0.858/0.856 (0.861/0.899/0.865) | 0.988/0.988/0.988 (0.989/0.989/0.986) | PASS / retain |
| workgroup / 4099 | 0.721/0.726/0.726 (0.726/0.730/0.723) | 0.789/0.796/0.795 (0.798/0.795/0.826) | 0.990/0.992/0.991 (0.991/1.035/0.993) | PASS / retain |
| subgroup / 4099 | 0.890/0.890/0.890 (0.876/0.890/0.891) | 0.870/0.866/0.874 (0.891/0.879/0.890) | 0.993/0.957/0.996 (0.994/0.990/0.996) | PASS / retain |
| workgroup / 65539 | 0.691/0.708/0.694 (0.678/0.691/0.686) | 0.783/0.791/0.795 (0.808/0.817/0.820) | 0.981/0.987/0.988 (0.994/0.979/0.998) | PASS / retain |
| subgroup / 65539 | 0.880/0.878/0.879 (0.898/0.882/0.875) | 0.860/0.876/0.885 (0.891/0.820/0.896) | 0.996/0.995/0.993 (1.050/1.105/0.989) | PASS / retain |

## Interpretation and pending work

The fused producer and full GN step produce real resident GPU wins in measured
large configurations on both targets while preserving exact full outputs. Those
wins do not overcome transfer cost or noise in every supported configuration.
The generic point-plane and solver helpers therefore remain candidate implementations.
The checked solver convenience profile has direct numerical tests but no timing
qualification. Preserve specialized incumbents for every failed/noisy/unmeasured
configuration; do not infer a size threshold or choose a default from this table.

Further work: stabilize/understand host/transfer tails, optimize remaining regressing
solver cases without changing loop semantics, independently qualify any new profile
or workload, and request maintainer review before a separate promotion PR.
