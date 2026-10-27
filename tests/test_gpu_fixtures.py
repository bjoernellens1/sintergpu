import os
import unittest


@unittest.skipUnless(os.getenv('SINTERGPU_GPU_TESTS') == '1', 'GPU execution explicitly requested only')
class GPUFixtureTests(unittest.TestCase):
    def test_locus_parameter_array_and_pitched_resize(self):
        from sintergpu.catalog import get_operation
        from sintergpu.validation import run
        report = run([get_operation('image.resize_nearest')], backend='vulkan')
        self.assertEqual(report['status'], 'passed')
        self.assertGreater(len(report['cases']), 0)


if __name__ == '__main__':
    unittest.main()
