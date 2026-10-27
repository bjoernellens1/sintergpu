# Explicit scan, compaction and borrowed Vulkan radix sort

`kernels/primitives/scan.slang` provides parallel forward inclusive/exclusive SUM
and PRODUCT over contiguous nondecreasing uint32 segment IDs. Repeated IDs must
never occur after another segment: `[0,0,2,2]` represents an empty segment1, while
`[0,1,0]` is invalid. Dense `[B,N]` inputs use IDs repeated N times per batch.
Empty input records no scan dispatch. Segment IDs and counts are caller metadata;
no shader infers packed segment boundaries through a CPU readback.

Float32 is default. Compile the identical source with
`SINTERGPU_SCAN_UINT=1` for uint32 exact prefix counts (arithmetic modulo2^32).
The `product` uniform selects multiplication versus addition for every pass.
PRODUCT exclusive shifts the inclusive result with identity1; it never divides by
an input, so zero factors are supported. SUM identity is0. This is a forward-only
port of nerfacc semantics; gradients and normalized variants are not provided.
Tree parenthesization changes floating rounding; tolerance comparisons against the
independent serial NumPy oracle are required. No exact floating serial-order claim.

`scan_layout(count, exclusive=...)` returns element counts for every buffer and an
ordered plan. Caller allocates once and owns all buffers, kernels, device and command
encoder. Scratch IDs are uint32, scalar scratch matches the compiled specialization.
Distinct labels require distinct allocations/ranges, particularly `output` versus
`exclusive_output`. All inputs must fit uint32 indexing, device buffer limits and
maximum one-dimensional dispatch workgroup count. This recipe does not partition a
dispatch exceeding the adapter's `maxComputeWorkGroupCount[0]`; the caller must reject
such input or partition at a higher level, never truncate a dispatch.

Each 256-lane block initializes all shared values/IDs and performs eight parallel
stages. Every stage finishes all reads before shared writes (two group barriers).
Each active output and block-end value/ID is written before its first subsequent
read, including partial blocks. Recursively scan block ends until one block remains;
then descend through `scanCarry`. Carry applies only when a lane belongs to the
preceding block's final segment. The hierarchy has no fixed256/65536 input-length
ceiling. Work is O(n log256), scratch O(n), pass count O(log256 n). Zero input has
zero scratch reads and zero dispatches.

`record_scan` records this plan into an explicitly supplied encoder. The adapter
must enforce compute storage write/read visibility between passes (SlangPy encoder
resource tracking provides this). It performs no resource creation, submit, wait,
queue selection or readback. Consumers may record the returned plan themselves
using their existing resource barriers and runtime. Keep scratch alive through GPU
completion and prevent overlapping invocations from sharing scratch unsafely.

For stable uint32 payload compaction, record `compactPrepare` with at least one
256-lane group even when count0; it initializes selectedCount[0]=0 and flags/IDs.
Record a uint32 SUM inclusive scan of flags (all IDs0), then `compactScatter` for
nonempty input. Selected values retain input order and duplicates. Output capacity
must be at least input count; only the first selectedCount elements are defined.
selectedCount stays device-owned; direct/indirect downstream dispatch is consumer
policy. No hidden host count transfer exists. Prepare and scatter require the same
immutable predicate. Scratch need not be zero-filled by the host.

## Frozen standalone vrdx dependency

`third_party/vrdx` is an exact extraction of all files at
vk_gaussian_splatting commit`d2f603251b1b007c94ee07dd9c5579b7492af6fe`, path
`3rdparty/vrdx`. `provenance/vrdx.json` binds every byte hash and MIT license.
No handwritten replacement sorting algorithm was introduced.

The standalone CMake target is `vk_radix_sort` (static C++17). Root integration must
supply `volk`, Vulkan headers/library and `Slang_SLANGC_EXECUTABLE`; the donor CMake
builds four shader variants from three shaders and preserves their local constants
include. Upstream currently generates shader headers inside its source directory;
use a writable build staging copy if the installed source tree is read-only.

`vrdxCreateSorter` borrows an existing VkPhysicalDevice/VkDevice/VkPipelineCache and
creates descriptor/pipeline state at setup. Caller destroys the sorter after its
recorded GPU uses finish. `vrdxCmdSort`, `vrdxCmdSortKeyValue` and their indirect
variants record into an existing VkCommandBuffer using caller buffers and offsets.
Call `vrdxGetSorterStorageRequirements` or the key/value version for maximum count,
allocate the required usage/size and obey device storage-offset alignment. This API
creates no device, queue, command pool, command buffer or memory allocation and
performs no submit/wait. It records histogram fills and compute/transfer barriers.

The app must enable VK_KHR_push_descriptor, required compute subgroup operations
and compatible subgroup-size support. Shaders use 512-thread workgroups and need
subgroup size>=32; upstream attempts to pin32 using subgroup-size-control properties.
The caller must verify/enable the logical-device subgroupSizeControl feature when
this path is selected and reject unsupported subgroup/workgroup configurations.
Physical property queries alone do not prove feature enablement. Respect upstream
indirect input transfer usage/barrier requirements; leave unsupported count0/indirect
edge handling unpromoted until its native tests are run. Upstream returns void and
largely ignores Vulkan result codes, so creation success/error handling needs an
application integration gate. This extraction is optional native Vulkan source,
not a SlangPy sort kernel and not a qualified sort release.

MIT RuilongLi and AMD notices are retained in `provenance/primitives.json` together
with frozen reference hashes and the PyTorch ScanUtils helper lineage. No upstream
CUDA, HIP, CUB or PyTorch dependency source is copied into the portable kernels.

Native CMake archive installation and relocated CPU-only static linking are covered under both pinned Slang versions; see [packaging](packaging.md#optional-native-c-radix-library). These checks initialize no Vulkan device and do not establish sorting correctness, timing or cross-vendor support.
