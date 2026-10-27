import unittest,copy
from reference import execute,validate,f4
from cases import cases
class ReferenceTests(unittest.TestCase):
 def test_every_kernel_has_valid_fixture(self):
  import json
  from pathlib import Path
  names={x['name']for x in json.loads((Path(__file__).resolve().parents[2]/'gpu/kernels.json').read_text())['kernels']}
  self.assertEqual(names,{x['name']for x in cases()})
  for c in cases():
   with self.subTest(c['label']):execute(c['name'],c['params'],c['inputs'],c['output'],c['state'])
 def test_known_colors_and_geometry(self):
  c=cases()[1];o,_=execute(c['name'],c['params'],c['inputs'],c['output'],c['state']);self.assertEqual(o[0],f4(0,0,0,1));self.assertEqual(o[1],f4(1,1,1,1))
  c=next(c for c in cases()if c['label']=='optical-backproject');o,_=execute(c['name'],c['params'],c['inputs'],c['output'],c['state']);self.assertEqual(o[5],f4(0,0,2,1))
  c=next(c for c in cases()if c['label']=='nearest-depth-collision');_,s=execute(c['name'],c['params'],c['inputs'],c['output'],c['state']);from reference import number;self.assertEqual(number(s[0]),1)
 def test_missing_calibration_and_bounds_fail_explicitly(self):
  for name in('depth_register','undistort','backproject'):
   c=copy.deepcopy(next(c for c in cases()if c['name']==name));c['params'][28]=0
   with self.assertRaisesRegex(ValueError,'calibration'):execute(c['name'],c['params'],c['inputs'],c['output'],c['state'])
  c=copy.deepcopy(cases()[0]);c['inputs'][0]=[]
  with self.assertRaisesRegex(ValueError,'extent'):execute(c['name'],c['params'],c['inputs'],c['output'],c['state'])
 def test_overflow_epoch_policy(self):
  c=next(c for c in cases()if c['label']=='voxel-capacity-overflow');_,s=execute(c['name'],c['params'],c['inputs'],c['output'],c['state']);self.assertEqual(s[-3:],[0,2,1])
if __name__=='__main__':unittest.main()
