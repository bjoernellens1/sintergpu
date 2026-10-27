cmake_minimum_required(VERSION 3.25)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" SINTERGPU_VERSION LIMIT_COUNT 1)
project(sintergpu VERSION ${SINTERGPU_VERSION} LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
option(SINTERGPU_BUILD_RADIX "Build optional native Vulkan radix sort (requires caller-provided volk and Slang compiler)" OFF)

add_library(sintergpu_kernels INTERFACE)
add_library(sintergpu::kernels ALIAS sintergpu_kernels)
set_target_properties(sintergpu_kernels PROPERTIES EXPORT_NAME kernels)
target_compile_features(sintergpu_kernels INTERFACE cxx_std_20)
target_include_directories(sintergpu_kernels INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# Applications can include only selected shader modules. No runtime is linked.
set(SINTERGPU_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/sintergpu/kernels")
install(TARGETS sintergpu_kernels EXPORT sintergpuTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY src/sintergpu/kernels/ DESTINATION ${CMAKE_INSTALL_DATADIR}/sintergpu/kernels)
install(DIRECTORY LICENSES/ DESTINATION ${CMAKE_INSTALL_DATADIR}/sintergpu/licenses)
install(FILES LICENSE NOTICE VERSION DESTINATION ${CMAKE_INSTALL_DATADIR}/sintergpu)
install(EXPORT sintergpuTargets NAMESPACE sintergpu:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sintergpu)
if(SINTERGPU_BUILD_RADIX)
    include(cmake/SinterGPURadix.cmake)
endif()
configure_package_config_file(cmake/sintergpuConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sintergpuConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sintergpu)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/sintergpuConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sintergpuConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/sintergpuConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sintergpu)
