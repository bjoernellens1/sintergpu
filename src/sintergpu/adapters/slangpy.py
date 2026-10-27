"""Record work on an existing SlangPy device and command encoder."""
import hashlib
from pathlib import Path
from .. import kernel_path
from ..catalog import get_operation


class SlangPyAdapter:
    def __init__(self, device, *, allow_experimental=False):
        if device is None:
            raise ValueError('a caller-owned SlangPy device is required')
        self.device = device
        self._kernels = {}
        self.source_identities = {}
        self.allow_experimental = allow_experimental

    def kernel(self, operation_id):
        if operation_id in self._kernels:
            return self._kernels[operation_id]
        op = get_operation(operation_id)
        if op['qualification']['status'] != 'qualified' and not self.allow_experimental:
            raise RuntimeError('operation requires explicit allow_experimental=True; inspect its qualification record')
        source = kernel_path(operation_id)
        # Include dependencies in identity, not only the entry source.
        dependency_bytes = b''.join(str(p.relative_to(source.parent)).encode()+b'\0'+p.read_bytes()+b'\0' for p in sorted(source.parent.rglob('*.slang')))
        root = source.parent.parent
        common = root / 'common'
        if common.is_dir():
            dependency_bytes += b''.join(str(p.relative_to(root)).encode()+b'\0'+p.read_bytes()+b'\0' for p in sorted(common.rglob('*.slang')))
        digest = hashlib.sha256(source.read_bytes() + dependency_bytes).hexdigest()
        self.source_identities[operation_id] = digest
        module = self.device.load_module(str(source))
        program = self.device.link_program([module], [module.entry_point(op['entrypoint'])])
        self._kernels[operation_id] = self.device.create_compute_kernel(program)
        return self._kernels[operation_id]

    prepare = kernel

    def record(self, operation_id, *, thread_count, bindings, command_encoder):
        if command_encoder is None:
            raise ValueError('caller-owned recording command encoder required')
        if len(thread_count) != 3 or any(not isinstance(v, int) or v < 0 for v in thread_count):
            raise ValueError('thread_count requires three nonnegative integer extents')
        if all(thread_count):
            if operation_id not in self._kernels:
                raise RuntimeError('prepare(operation_id) during setup before recording')
            self._kernels[operation_id].dispatch(thread_count=list(thread_count), vars=bindings,
                                                command_encoder=command_encoder)
