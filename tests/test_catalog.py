import json
from pathlib import Path
import subprocess
import sys
import unittest


class CatalogTests(unittest.TestCase):
    def test_packaged_kernels_and_provenance(self):
        import sintergpu
        from sintergpu.catalog import list_operations, validate_catalog
        operations = list_operations()
        self.assertGreaterEqual(len(operations), 17)
        self.assertEqual(sintergpu.__version__, '0.1.0')
        self.assertEqual(validate_catalog(), [])
        for operation in operations:
            with self.subTest(operation['id']):
                self.assertTrue(sintergpu.kernel_path(operation['id']).is_file())
                self.assertEqual(len(operation['source']['revision']), 40)
                self.assertEqual(len(operation['sha256']), 64)
                self.assertNotEqual(operation['qualification']['status'], 'qualified')

    def test_unknown_and_traversal_are_rejected(self):
        import sintergpu
        for name in ('missing', '../VERSION', '/etc/passwd'):
            with self.assertRaises(KeyError):
                sintergpu.kernel_path(name)

    def test_device_module_available_without_runtime(self):
        import sintergpu
        self.assertIn('SINTERGPU_REDUCE_256', sintergpu.module_path('common.block_reduce').read_text())
        with self.assertRaises(KeyError):
            sintergpu.module_path('../VERSION')

    def test_catalog_rejects_hash_and_extent_corruption(self):
        from sintergpu.catalog import list_operations, validate_catalog
        import copy
        data = copy.deepcopy(list_operations())
        data[0]['sha256'] = '0' * 64
        self.assertTrue(validate_catalog(data))
        data[0]['path'] = '../../../etc/passwd'
        self.assertTrue(validate_catalog(data))

    def test_cli_outputs_machine_readable_contract(self):
        run = subprocess.run([sys.executable, '-m', 'sintergpu', 'list', '--json'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertGreaterEqual(len(json.loads(run.stdout)), 17)

    def test_gpu_unavailable_is_not_reported_as_pass(self):
        run = subprocess.run([sys.executable, '-m', 'sintergpu', 'test', 'missing'], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)


if __name__ == '__main__':
    unittest.main()
