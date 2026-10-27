# VKL temporal best-observation forward profile

`vkl.temporal_best.temporalBest` ports all four forward numerical outputs of
`temporal_best_observation` at frozen VKL commit
`3384023a311651ea4260cf12ddab1567972f89f3`. It selects equivalent evidence;
consumer admission, novelty and mapping policy remain downstream. The unchanged
Torch function retains its original validation and autograd route.

Input qualities are float32, row-major `[B,N]`; target quality is `[B]`.
Equivalence uses scalar uint32 zero/nonzero storage. Optional pairwise quality
replaces neighbor quality. Optional confidence must be finite in [0,1]; when
absent, confidence is equivalence converted to float32. Qualities may contain
NaN or infinity. At least one neighbor is required.

`TemporalBestParams` is four uint32 fields: `batches`, `neighbors`, `useQuality`,
`useConfidence`. `temporal_best_layout` returns capacities, parameters and a
logical `(B,1,1)` dispatch using 256-thread workgroups. It rejects offset-product,
signed source-index and actual caller-device X-dispatch-limit overflow. A zero
batch records no work. `validate_capacities` checks scalar32 counts; descriptor
limits, types, content contracts, alignment and alias hazards remain caller-owned.

Bindings are `equivalence`, `target_quality`, `neighbor_quality`,
`pairwise_quality`, `pairwise_confidence`; outputs are `best_indexOut` (signed
int32), `best_qualityOut`, `replaceabilityOut`, `replacement_confidenceOut`
(float32). Values preserve Torch's int64 source indices within the declared
int32 range. Caller-owned one-element dummy buffers satisfy disabled pairwise
inputs. With pairwise quality enabled, neighbor quality is unused and likewise
accepts a one-element dummy. Inputs and outputs must not alias in ways that race.

The first maximum wins, including the first NaN as in Torch's max reduction.
A nonfinite winning neighbor makes the target win, even when another equivalent
neighbor has a finite quality. Otherwise target quality wins ties. Target index
is zero; neighbor n uses n+1. A NaN target can lose to a finite neighbor and then
produce NaN replaceability. The shader explicitly preserves these NaNs and
`0*Inf` replacement-confidence products. All four outputs are written for every
in-range row, without scratch or atomics.

This initial implementation scans each row serially, O(BN), with O(1) private
state. Adjacent threads read rows N elements apart; large N can limit throughput.
A cooperative reduction would need fresh first-NaN, tie and numerical qualification.
No optimized default is selected. The production entrypoint owns no compilation,
allocation, submission, synchronization or readback.

CPU tests compare the independent NumPy oracle with frozen Torch for N=1,2,7,33,
randomized observations, ties, no equivalents, NaNs/infinities and optional inputs.
GPU fixtures compare all four outputs and poisoned output padding, cross a
256-thread group with a tail, and exercise all four independent option-flag
combinations. Integer outputs are exact; tested float limits are rtol=8e-6 and
atol=5e-7, with explicit NaN equality. Gradients, other devices/compiler runtime
configurations, full consumer behavior and general extreme-value certification
remain pending.

```sh
PYTHONPATH=src python -m pytest tests/test_vkl_temporal_best.py -q
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src python -m pytest tests/test_vkl_temporal_best.py -q
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src python benchmarks/vkl_temporal_best.py \
  --batches 65536 --neighbors 32 --pattern sparse --output temporal-best.json
```

The benchmark preallocates and warms up, retains raw GPU timestamp samples,
and reports host record/submit/wait and full upload/dispatch/readback separately.
One kernel is this operation's complete GPU pipeline. Dense/sparse equivalence,
early/late winners, NaN rows and N scaling are independent configurations.
Source/compiler/device/driver identities and transfer bytes are recorded.
Shared-device measurements provide no controlled speedup, cross-vendor or
optimization-adoption claim.
