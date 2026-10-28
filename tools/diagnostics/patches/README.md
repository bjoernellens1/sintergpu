# Draft allocation query patches: incomplete, unavailable

These partial source patches are review artifacts only. The RHI patch proposes a
separate versioned COM query interface without changing `IBuffer`'s vtable. It is
based on the EliGSiR creator-receipt-patched RHI header (SHA-256
`516a4390911cae8b3ca93e9191ef145c39f232aaef80fa9f4c8c1ed2111f9111`), itself
based on Slang RHI commit `1a9768741246b1f356116d5dfa305933034ce049`. The
SlangPy patch applies to the audited `resource.cpp` SHA-256
`a1d0082a7440f17d4a51c231fcc14ffb66088e8fdb869058d4a21aec735335d4`
from the SlangPy 0.43.1 source set. Modified files carry a SinterGPU notice;
both upstream projects declare `SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception`.
Retain their complete license texts and notices and the existing EliGSiR patch
notices when distributing any patched source or binary. The local-only patch
application tests require the audited source checkout; skips mean no source
application validation and do not constitute CI proof.

Neither patch implements the interface. The Python property always returns
`None`, including for ordinary buffers. In particular it does not query or
cast a private backend object. There is no compiled runtime, no mapping, no
selected-device or provider receipt here and no GPU/CPU performance result.

Before an operational implementation: resolve the pinned COM aggregation and
reference-count implementation, arrange a live owner generation and copied
allocation size only after a successful bind, reject every unsupported path,
connect the retained interface to a SlangPy value copy with close/release
serialization, and bind it to the separately created same-device UUID and
instance dispatch observations. Review externally exposed handles and document
that the controlled fixture prohibits foreign native mutations; this is not a
universal interception guarantee. The interface UUID, ABI layout, exception and
thread rules remain provisional. Compile, independent source review, device and
runtime tests are mandatory before use.
