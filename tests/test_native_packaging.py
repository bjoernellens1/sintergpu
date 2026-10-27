"""Optional real static-link/relocation checks; never initializes Vulkan/GPU."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
native_dependencies=pytest.mark.skipif(os.environ.get('SINTERGPU_NATIVE_PACKAGE_TESTS')!='1',reason='native packaging requires explicit compiler/volk test dependencies')


def invoke(args,*,cwd):
    result=subprocess.run(list(map(str,args)),cwd=cwd,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    return result.stdout


@native_dependencies
@pytest.mark.parametrize('build_kind',['plain','namespace'])
def test_radix_install_relocates_without_exporting_callers_volk(tmp_path,build_kind):
    cmake=os.environ.get('SINTERGPU_CMAKE','cmake')
    volk=Path(os.environ['SINTERGPU_TEST_VOLK']).resolve()
    pin=json.loads((ROOT/'tests/native/volk.json').read_text())
    for name,digest in pin['files'].items():assert hashlib.sha256((volk/name).read_bytes()).hexdigest()==digest
    compiler=Path(os.environ['SINTERGPU_SLANGC']).resolve()
    provider=tmp_path/'provider';provider.mkdir();build=tmp_path/'build';prefix=tmp_path/'prefix'
    target='volk' if build_kind=='plain' else 'host_volk'
    alias='' if build_kind=='plain' else 'add_library(volk::volk ALIAS host_volk)'
    (provider/'CMakeLists.txt').write_text(f'''cmake_minimum_required(VERSION 3.25)
project(native_package_provider LANGUAGES C CXX)
find_package(Vulkan REQUIRED)
add_library({target} STATIC "{volk}/volk.c")
{alias}
target_include_directories({target} PUBLIC "{volk}")
target_link_libraries({target} PUBLIC Vulkan::Headers ${{CMAKE_DL_LIBS}})
file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/volk-archive.txt" CONTENT "$<TARGET_FILE:{target}>")
set(SINTERGPU_BUILD_RADIX ON CACHE BOOL "" FORCE)
add_subdirectory("{ROOT}" sintergpu)
''')
    invoke([cmake,'-S',provider,'-B',build,f'-DCMAKE_INSTALL_PREFIX={prefix}',f'-DSINTERGPU_SLANGC={compiler}'],cwd=tmp_path)
    invoke([cmake,'--build',build,'--parallel','2'],cwd=tmp_path)
    invoke([cmake,'--install',build],cwd=tmp_path)
    relocated=tmp_path/'relocated';prefix.rename(relocated)
    exports=list(relocated.rglob('sintergpuTargets.cmake'));assert len(exports)==1
    exported=exports[0].read_text();assert str(ROOT) not in exported and str(provider) not in exported and str(build) not in exported
    installed_archive=list(relocated.rglob('*sintergpu_radix*'));assert len(installed_archive)==1
    archive_sha=hashlib.sha256(installed_archive[0].read_bytes()).hexdigest()
    volk_archive=Path((build/'volk-archive.txt').read_text())
    volk_archive_name=volk_archive.name;volk_archive_bytes=volk_archive.read_bytes()
    generated={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (build/'sintergpu/vrdx/generated').glob('*.h')}
    assert len(generated)==4
    # The original producer paths disappear before any consumer config/link.
    provider.rename(tmp_path/'hidden-provider');build.rename(tmp_path/'hidden-build')
    # Separate dependency prefixes exercise find_dependency and old CMake's
    # PACKAGE_PREFIX_DIR clobbering, plus either external volk target spelling.
    for mode in ('existing-plain','existing-namespace','package-plain','package-namespace'):
        consumer=tmp_path/mode;consumer.mkdir()
        dependency_target='volk::volk' if mode.endswith('namespace') else 'volk'
        dependency_prefix=tmp_path/(mode+'-dependency');dependency_prefix.mkdir()
        alias='add_library(volk::volk ALIAS caller_volk)' if dependency_target=='volk::volk' else 'add_library(volk ALIAS caller_volk)'
        if mode.startswith('existing'):
            setup=f'''find_package(Vulkan REQUIRED)
add_library(caller_volk STATIC "{volk}/volk.c")
{alias}
target_include_directories(caller_volk PUBLIC "{volk}")
target_link_libraries(caller_volk PUBLIC Vulkan::Headers ${{CMAKE_DL_LIBS}})
'''
        else:
            (dependency_prefix/volk_archive_name).write_bytes(volk_archive_bytes)
            for name in pin['files']:shutil.copyfile(volk/name,dependency_prefix/name)
            (dependency_prefix/'volkConfig.cmake').write_text(f'''add_library({dependency_target} STATIC IMPORTED)
set_target_properties({dependency_target} PROPERTIES IMPORTED_LOCATION "{dependency_prefix/volk_archive_name}" INTERFACE_INCLUDE_DIRECTORIES "{dependency_prefix}" INTERFACE_LINK_LIBRARIES "${{CMAKE_DL_LIBS}}")
set(PACKAGE_PREFIX_DIR "{dependency_prefix}")
''')
            setup=''
        (consumer/'CMakeLists.txt').write_text(f'''cmake_minimum_required(VERSION 3.25)
project(native_package_consumer LANGUAGES C CXX)
{setup}
find_package(sintergpu CONFIG REQUIRED)
find_package(sintergpu CONFIG REQUIRED)
if(NOT EXISTS "${{SINTERGPU_KERNEL_DIR}}/image/camera_color.slang" OR NOT EXISTS "${{SINTERGPU_DATA_DIR}}/timings.json")
  message(FATAL_ERROR "SinterGPU relocated resources missing")
endif()
add_executable(link_probe main.cpp)
target_link_libraries(link_probe PRIVATE sintergpu::radix)
''')
        (consumer/'main.cpp').write_text('#include <vk_radix_sort.h>\nint main() { auto* volatile create = &vrdxCreateSorter; return create == nullptr; }\n')
        invoke([cmake,'-S',consumer,'-B',consumer/'build',f'-DCMAKE_PREFIX_PATH={relocated};{dependency_prefix}'],cwd=tmp_path)
        invoke([cmake,'--build',consumer/'build','--parallel','2'],cwd=tmp_path)
        invoke([consumer/'build/link_probe'],cwd=tmp_path)
    report={'build_kind':build_kind,'compiler_sha256':hashlib.sha256(compiler.read_bytes()).hexdigest(),
            'volk_revision':pin['revision'],'consumer_modes':4,'scope':'Real archive compilation, relocated install, static linking and CPU-only link probe; no Vulkan initialization/GPU execution'}
    sources=['VERSION','CMakeLists.txt','cmake/SinterGPURadix.cmake','cmake/sintergpuConfig.cmake.in',
             'third_party/vrdx/src/vk_radix_sort.cc','third_party/vrdx/include/vk_radix_sort.h']
    sources += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'third_party/vrdx/src/shader').glob('*.slang'))]
    report.update(library_version=(ROOT/'VERSION').read_text().strip(),cmake_version=invoke([cmake,'--version'],cwd=tmp_path).splitlines()[0],
                  cxx_compiler=invoke(['c++','--version'],cwd=tmp_path).splitlines()[0],
                  archive_sha256=archive_sha,generated_shader_headers_sha256=generated,
                  volk_files_sha256=pin['files'],source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    output=os.environ.get('SINTERGPU_NATIVE_TEST_EVIDENCE')
    if output:
        evidence=Path(output);evidence.mkdir(parents=True,exist_ok=True)
        (evidence/(build_kind+'.json')).write_text(json.dumps(report,indent=2)+'\n')


def test_both_compiler_ci_lanes_run_native_install_and_link_checks():
    text=(ROOT/'.github/workflows/ci.yml').read_text()
    compiler=text.split('  compiler:',1)[1].split('  reflection-compatibility:',1)[0]
    assert 'SINTERGPU_NATIVE_PACKAGE_TESTS' in compiler
    assert 'tests/test_native_packaging.py' in compiler
    assert 'tests/native/volk.json' in compiler
    assert 'native-packaging-${{ matrix.version }}' in compiler
