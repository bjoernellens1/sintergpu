# The unchanged MIT vrdx implementation is optional. Generate shader headers in
# the build directory, rather than upstream's source-tree generated/ location.
find_package(Vulkan REQUIRED)
if(NOT TARGET volk AND NOT TARGET volk::volk)
    find_package(volk CONFIG REQUIRED)
endif()
if(TARGET volk::volk)
    set(_sintergpu_volk volk::volk)
elseif(TARGET volk)
    set(_sintergpu_volk volk)
else()
    message(FATAL_ERROR "Optional radix requires a caller-provided volk target")
endif()
find_program(SINTERGPU_SLANGC NAMES slangc REQUIRED)
set(_sintergpu_vrdx "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vrdx")
set(_sintergpu_generated "${CMAKE_CURRENT_BINARY_DIR}/vrdx/generated")
file(MAKE_DIRECTORY "${_sintergpu_generated}")
set(_sintergpu_headers)
foreach(_variant IN ITEMS upsweep spine downsweep downsweep_key_value)
    set(_shader "${_variant}")
    set(_defines)
    if(_variant STREQUAL "downsweep_key_value")
        set(_shader downsweep)
        set(_defines -DKEY_VALUE)
    endif()
    set(_output "${_sintergpu_generated}/${_variant}_slang.h")
    add_custom_command(OUTPUT "${_output}"
        COMMAND "${SINTERGPU_SLANGC}" ${_defines} -target spirv
            -fvk-t-shift 0 0 -fvk-u-shift 0 0
            -source-embed-name "${_variant}_slang" -source-embed-style text
            -o "${_output}" "${_sintergpu_vrdx}/src/shader/${_shader}.slang"
        DEPENDS "${_sintergpu_vrdx}/src/shader/${_shader}.slang"
                "${_sintergpu_vrdx}/src/shader/constants.slang"
        VERBATIM)
    list(APPEND _sintergpu_headers "${_output}")
endforeach()
add_library(sintergpu_radix STATIC "${_sintergpu_vrdx}/src/vk_radix_sort.cc" ${_sintergpu_headers})
add_library(sintergpu::radix ALIAS sintergpu_radix)
set_target_properties(sintergpu_radix PROPERTIES EXPORT_NAME radix POSITION_INDEPENDENT_CODE ON)
target_compile_features(sintergpu_radix PUBLIC cxx_std_17)
target_include_directories(sintergpu_radix PUBLIC
    $<BUILD_INTERFACE:${_sintergpu_vrdx}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/sintergpu/radix>
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vrdx" ${Vulkan_INCLUDE_DIRS})
# A caller-local source target need not be installed or exported by SinterGPU.
# The installed config resolves either supported spelling to a private bridge.
target_link_libraries(sintergpu_radix PUBLIC Vulkan::Headers
    $<BUILD_INTERFACE:${_sintergpu_volk}>
    $<INSTALL_INTERFACE:sintergpu::radix_volk>)
install(TARGETS sintergpu_radix EXPORT sintergpuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES "${_sintergpu_vrdx}/include/vk_radix_sort.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sintergpu/radix)
install(FILES "${_sintergpu_vrdx}/LICENSE" DESTINATION ${CMAKE_INSTALL_DATADIR}/sintergpu/licenses RENAME vrdx-LICENSE)
