"""Explicit Vulkan execution through SlangPy's Slang-RHI bindings."""

from __future__ import annotations

import hashlib

import numpy as np

from .emitter import emit_slang
from .ir import AccessKind, KernelIR
from .layout import pack, unpack


class VulkanRuntime:
    def __init__(self, device=None):
        import slangpy as spy

        self.device = device if device is not None else spy.Device(type=spy.DeviceType.vulkan)
        self._kernels = {}

    def compile(self, source: str, entry: str = "computeMain"):
        key = (source, entry)
        if key not in self._kernels:
            name = "caspar_" + hashlib.sha256(source.encode()).hexdigest()[:16]
            module = self.device.load_module_from_source(name, source)
            program = self.device.link_program([module], [module.entry_point(entry)])
            self._kernels[key] = self.device.create_compute_kernel(program)
        return self._kernels[key]

    def buffer(self, array):
        import slangpy as spy

        data = np.ascontiguousarray(array)
        if data.dtype not in (np.dtype("float32"), np.dtype("uint32")):
            raise TypeError("GPU arrays must have dtype float32 or uint32")
        if not data.size:
            raise ValueError("GPU buffers must contain at least one element")
        return self.device.create_buffer(
            element_count=data.size,
            struct_size=4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            data=data,
        )

    def dispatch(
        self,
        ir: KernelIR,
        buffers: dict,
        *,
        problem_size: int,
        allocations: dict[str, int],
        command_encoder=None,
    ):
        """Dispatch device buffers without readback; caller validates GPU index contents."""
        ir.validate()
        if not isinstance(problem_size, int) or not 0 <= problem_size <= 0xFFFFFFFF:
            raise ValueError("problem_size must be a uint32 integer")
        params = {"problem_size": problem_size}
        for buf in ir.buffers:
            if buf.name not in buffers:
                raise ValueError(f"Missing buffer {buf.name}")
            if buf.num_alloc_param:
                n = allocations.get(buf.name)
                if not isinstance(n, int) or not 0 <= n <= 0xFFFFFFFF:
                    raise ValueError(f"Missing or invalid allocation count for {buf.name}")
                if buf.index_name is None and n < problem_size:
                    raise ValueError(f"Allocation {buf.name} is smaller than dispatch")
                params[f"{buf.name}_num_alloc"] = n
        if problem_size:
            self.compile(emit_slang(ir)).dispatch(
                thread_count=[problem_size, 1, 1],
                vars={**buffers, "params": params},
                command_encoder=command_encoder,
            )

    def dispatch_arrays(self, ir: KernelIR, arrays: dict, *, problem_size: int | None = None):
        """Validate host arrays/index streams, upload, run, and return written arrays."""
        ir.validate()
        normalized = {}
        allocations = {}
        for buf in ir.buffers:
            if buf.name not in arrays:
                raise ValueError(f"Missing array {buf.name}")
            array = np.asarray(arrays[buf.name])
            if buf.scalar_type == "uint":
                if array.dtype != np.uint32 or array.ndim != 1:
                    raise ValueError(f"{buf.name} must be a one-dimensional uint32 index array")
            else:
                if array.dtype != np.float32:
                    raise TypeError(f"{buf.name} must have dtype float32")
                if array.ndim == 1 and buf.components == 1:
                    array = array.reshape(-1, 1)
                if array.ndim != 2 or array.shape[1] != buf.components:
                    raise ValueError(f"{buf.name} must have shape (N, {buf.components})")
                allocations[buf.name] = array.shape[0]
            normalized[buf.name] = array
        if problem_size is None:
            counts = [
                normalized[b.index_name].size if b.index_name else allocations[b.name]
                for b in ir.buffers
                if b.scalar_type == "float"
            ]
            problem_size = min(counts)
        if not isinstance(problem_size, int) or problem_size < 0:
            raise ValueError("problem_size must be nonnegative")
        for buf in ir.buffers:
            if buf.scalar_type != "float":
                continue
            if buf.index_name:
                indices = normalized[buf.index_name]
                if indices.size < problem_size:
                    raise ValueError(f"{buf.index_name} has fewer entries than problem_size")
                indices = indices[:problem_size]
                if indices.size and np.max(indices) >= allocations[buf.name]:
                    raise ValueError(
                        f"{buf.index_name} contains out-of-bounds indices for {buf.name}"
                    )
                if buf.unique_indices and np.unique(indices).size != indices.size:
                    raise ValueError(f"{buf.index_name} must have unique destination indices")
            elif allocations[buf.name] < problem_size:
                raise ValueError(f"{buf.name} has fewer rows than problem_size")
        if problem_size == 0:
            return {
                b.name: normalized[b.name].copy() for b in ir.buffers if b.access != AccessKind.READ
            }
        gpu = {
            b.name: self.buffer(
                normalized[b.name]
                if b.scalar_type == "uint"
                else pack(normalized[b.name], b.layout)
            )
            for b in ir.buffers
        }
        self.dispatch(ir, gpu, problem_size=problem_size, allocations=allocations)
        return {
            b.name: unpack(
                gpu[b.name].to_numpy().view(np.float32), allocations[b.name], b.components, b.layout
            )
            for b in ir.buffers
            if b.access != AccessKind.READ
        }
