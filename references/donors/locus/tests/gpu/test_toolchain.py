import importlib.util,json,tempfile,unittest,copy
from pathlib import Path
from unittest.mock import patch
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('build_shaders',root/'tools/build-shaders.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class ToolchainTests(unittest.TestCase):
 def test_missing_compiler(self):
  with self.assertRaisesRegex(ValueError,'compiler unavailable'):module.build('/nonexistent/slangc','spirv-val','/tmp/not-written')
 def test_wrong_version(self):
  with patch.object(module.shutil,'which',return_value='/fake'),patch.object(module.subprocess,'check_output',return_value='2026.18'):
   with self.assertRaisesRegex(ValueError,'wrong Slang'):module.build('fake','fake','/tmp/not-written')
 def test_missing_metal_toolchain(self):
  with patch.object(module.shutil,'which',side_effect=lambda x:None if x=='xcrun'else'/fake'),patch.object(module.subprocess,'check_output',return_value='2026.19'):
   with self.assertRaisesRegex(ValueError,'Metal compiler required'):module.build('fake','fake','/tmp/not-written',True)
 def test_failed_target_compile_is_not_masked(self):
  import subprocess
  with patch.object(module.shutil,'which',return_value='/fake'),patch.object(module.subprocess,'check_output',return_value='2026.19'),patch.object(module,'run',side_effect=subprocess.CalledProcessError(1,['slangc'])):
   with tempfile.TemporaryDirectory()as d:
    with self.assertRaises(subprocess.CalledProcessError):module.build('fake','fake',d)
    self.assertFalse((Path(d)/'build.json').exists())
 def test_layout_mutations_rejected(self):
  # Positive reflection from a real compile; fake metadata cannot establish a pass.
  path=root/'build/shaders/yuv_to_rgb.spv.json'
  if not path.exists():self.skipTest('run shader compiler before reflection mutation tests')
  original=json.loads(path.read_text());manifest=json.loads((root/'gpu/kernels.json').read_text());module.validate_reflection(path,manifest)
  for mutate in[lambda j:j['parameters'][0]['binding'].update(index=7),lambda j:j['parameters'][-1]['type']['elementType']['fields'][0]['binding'].update(size=64),lambda j:j['entryPoints'][0].update(threadGroupSize=[16,1,1]),lambda j:j['parameters'][0]['type'].update(baseShape='texture2D'),lambda j:j['parameters'][3]['type'].update(access='read'),lambda j:j['parameters'][-1]['type']['elementType']['fields'][0]['type']['elementType']['elementType'].update(scalarType='uint32')]:
   j=copy.deepcopy(original);mutate(j)
   with tempfile.TemporaryDirectory()as d:
    p=Path(d)/'reflection.json';p.write_text(json.dumps(j))
    with self.assertRaises(ValueError):module.validate_reflection(p,manifest)
if __name__=='__main__':unittest.main()
