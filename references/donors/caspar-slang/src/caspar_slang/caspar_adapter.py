from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .ir import AccessKind, Buffer, KernelIR

_MEMORY = re.compile(
    r"^\s*(?P<mode>Read|Write|Add)Idx(?P<dim>[1-4])<[^>]+>\("
    r"(?P<name>[A-Za-z_]\w*)\s*,\s*"
    r"(?P<offset>\d+)\s*\*\s*(?P=name)_num_alloc\s*,\s*"
    r"(?P<index>[^,]+)\s*,\s*(?P<args>[^)]*)\)\s*;\s*$"
)


def _lower_memory_line(line: str) -> list[str] | None:
    """Lower Caspar's vector chunks using its padded AoSoA storage addressing."""
    m = _MEMORY.match(line)
    if not m:
        return None
    dim = int(m["dim"])
    width = 4 if dim == 3 else dim
    args = [x.strip() for x in m["args"].split(",") if x.strip()]
    if len(args) != dim:
        raise ValueError(f"Accessor dimension mismatch in: {line}")
    index = m["index"].strip()
    if not re.fullmatch(r"(?:global_thread_idx|[A-Za-z_]\w*_idx)", index):
        raise NotImplementedError(f"Unsupported accessor index expression: {index}")
    result = []
    for j, arg in enumerate(args):
        address = (
            f"({int(m['offset'])}u * params.{m['name']}_num_alloc + {index} * {width}u + {j}u)"
        )
        if m["mode"] == "Read":
            result.append(f"{arg} = {m['name']}[{address}];")
        else:
            op = "+=" if m["mode"] == "Add" else "="
            result.append(f"{m['name']}[{address}] {op} {arg};")
    return result


def _access_kind(accessor: Any) -> AccessKind:
    cls = accessor.__class__.__name__
    if cls in {"ReadSequential", "ReadIndexed"}:
        return AccessKind.READ
    if cls in {"WriteSequential", "WriteIndexed"}:
        return AccessKind.WRITE
    if cls in {"AddSequential", "AddIndexed"}:
        return AccessKind.READ_WRITE
    raise NotImplementedError(f"{cls} has no portable accessor lowering")


def from_caspar_kernel(kernel: Any) -> KernelIR:
    """Reuse upstream DABSEG/register scheduling and lower sequential/indexed access."""
    if str(kernel.kernel_t) != "float":
        raise NotImplementedError("Only float32 Caspar kernels are supported")
    try:
        from symforce.caspar.code_formulation.dabseg_from_accessors import make_dabseg
        from symforce.caspar.code_formulation.dabseg_sorter import get_lines
    except ImportError as exc:
        raise RuntimeError("Install SymForce/Caspar with `uv sync --extra symforce`") from exc

    seen: dict[str, Buffer] = {}
    prep: dict[str, str] = {}
    indexed_names: set[str] = set()
    for accessor in kernel.accessors:
        kind = _access_kind(accessor)
        if str(accessor.storage_t) != "float":
            raise NotImplementedError("Only float32 Caspar storage is supported")
        if accessor.offset is not None or accessor.stride is not None:
            raise NotImplementedError("Accessor offset/stride is not supported")
        indexed = accessor.__class__.__name__.endswith("Indexed")
        layout = tuple(tuple(chunk) for chunk in accessor.chunk_indices)
        buf = Buffer(
            accessor.name,
            kind,
            components=sum(map(len, layout)),
            layout=layout,
            index_name=f"{accessor.idx_name}_indices" if indexed else None,
            unique_indices=indexed and kind != AccessKind.READ,
        )
        previous = seen.get(buf.name)
        if previous is not None:
            if previous.layout != buf.layout or previous.index_name != buf.index_name:
                raise ValueError(f"Inconsistent accessor layout/index for {buf.name}")
            buf = replace(
                buf,
                access=kind if previous.access == kind else AccessKind.READ_WRITE,
                unique_indices=buf.unique_indices or previous.unique_indices,
            )
        seen[buf.name] = buf
        if indexed:
            indexed_names.add(accessor.idx_name)
            if accessor.prep_lines:
                prep[accessor.idx_name] = accessor.prep_lines.replace("unsigned int", "uint")
            # AddIndexed upstream currently uses name_idx rather than reused idx_name_idx.
            if kind == AccessKind.READ_WRITE and accessor.name != accessor.idx_name:
                prep[accessor.name] = f"uint {accessor.name}_idx = {accessor.idx_name}_idx;"

    # Every shared index stream is declared and initialized exactly once.
    preparation = [
        f"uint {name}_idx = {name}_indices[global_thread_idx];" for name in sorted(indexed_names)
    ]
    preparation.extend(line for name, line in prep.items() if name not in indexed_names)
    buffers = list(seen.values())
    buffers.extend(
        Buffer(f"{name}_indices", AccessKind.READ, num_alloc_param=False, scalar_type="uint")
        for name in sorted(indexed_names)
    )
    dabseg = make_dabseg(kernel.inputs, kernel.outputs)
    lines, n_registers = get_lines(dabseg, dtype=kernel.kernel_t)
    portable_lines = preparation
    for line in lines:
        memory = _lower_memory_line(line)
        if memory is not None:
            portable_lines.extend(memory)
        else:
            if re.search(
                r"\b(?:Read\w*|Write\w*|AddIdx\w*|Load\w*|Flush\w*|SumStore|SumFlushFinal)\s*(?:<|\()",
                line,
            ) or any(
                token in line
                for token in ("__syncthreads", "__shared__", "cuda", "threadIdx", "blockIdx")
            ):
                raise NotImplementedError(f"Unlowered CUDA-specific Caspar line: {line}")
            portable_lines.append(line)
    return KernelIR(
        name=kernel.name,
        buffers=buffers,
        registers=[f"r{i}" for i in range(n_registers)],
        code_lines=portable_lines,
        block_size=min(int(getattr(kernel, "block_size", 256)), 1024),
        metadata={"source": "symforce.caspar", "layout": "aosoa"},
    )
