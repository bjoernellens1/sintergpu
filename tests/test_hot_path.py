"""Static ownership guards; synthetic source mutations prove guard sensitivity."""
import ast
from pathlib import Path
import unittest


ADAPTER = Path(__file__).resolve().parents[1] / 'src/sintergpu/adapters/slangpy.py'
FORBIDDEN = {
    'Device', 'create_device', 'create_buffer', 'create_texture', 'create_query_pool',
    'create_command_encoder', 'submit', 'finish', 'wait', 'wait_idle', 'synchronize',
    'to_numpy', 'to_torch', 'readback', 'read_buffer', 'copy_to_numpy', 'copy_to_host',
}


def borrowed_adapter_violations(source):
    """Conservative source guard, not a substitute for GPU/runtime validation."""
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            if name in FORBIDDEN:
                violations.append((node.lineno, name))
    record = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'record')
    # Recording must be a bounded operation independent of image/point count.
    # Extent validation generators are bounded by the three-element public ABI.
    for node in ast.walk(record):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            if name in {'kernel', 'prepare', 'get_operation', 'kernel_path', 'load_module', 'link_program', 'create_compute_kernel', 'rglob', 'read_bytes', 'read_text', 'sha256'}:
                violations.append((node.lineno, 'setup work during recording: ' + name))
        if isinstance(node, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp)):
            violations.append((node.lineno, 'item-sized recording loop or materialization'))
        elif isinstance(node, ast.GeneratorExp):
            if any(not isinstance(gen.iter, ast.Name) or gen.iter.id != 'thread_count' for gen in node.generators):
                violations.append((node.lineno, 'unbounded recording generator'))
    return violations


class BorrowedBoundaryTests(unittest.TestCase):
    def test_adapter_preserves_borrowed_runtime_boundary(self):
        self.assertEqual(borrowed_adapter_violations(ADAPTER.read_text()), [])

    def test_guard_rejects_runtime_ownership_mutations(self):
        source = ADAPTER.read_text()
        marker = '    def record('
        for statement in ('spy.Device()', 'self.device.create_buffer()', 'self.device.submit()',
                          'self.device.wait()', 'bindings["output"].to_numpy()'):
            with self.subTest(statement=statement):
                mutated = source.replace(marker, f'    def forbidden_mutation(self):\n        {statement}\n\n{marker}', 1)
                self.assertTrue(borrowed_adapter_violations(mutated))

    def test_guard_rejects_setup_rescan_during_recording(self):
        source = ADAPTER.read_text()
        target = '        if command_encoder is None:'
        mutated = source.replace(target, '        self.kernel(operation_id)\n' + target, 1)
        self.assertTrue(borrowed_adapter_violations(mutated))

    def test_guard_rejects_per_item_recording_allocation(self):
        source = ADAPTER.read_text()
        mutated = source + '\n'  # Insert inside the real method, never execute the mutation.
        target = '        if command_encoder is None:'
        self.assertIn(target, mutated)
        mutated = mutated.replace(target, '        rows = [dict() for _ in range(thread_count[0])]\n' + target, 1)
        self.assertTrue(borrowed_adapter_violations(mutated))


if __name__ == '__main__':
    unittest.main()
