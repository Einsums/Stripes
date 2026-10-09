#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.24)

project(
  Stripes
  VERSION 1.0.0
  DESCRIPTION "Portable SIMD vectors with runtime instruction-set dispatch"
  LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Windows finds a DLL beside the executable that loads it, so put the library, the tests and the
# examples in one directory there.
if(WIN32 AND PROJECT_IS_TOP_LEVEL AND NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ── Options ──────────────────────────────────────────────────────────────────
#
# Stripes dispatches through compile-time feature macros (`__AVX512F__`,
# `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC`, ...), so what a translation unit gets
# depends on its target flags. Kernels compiled through
# stripes_add_dispatch_sources() get one copy per rung and pick at run time;
# everything else sees the toolchain baseline unless one of the two pins below
# raises it. Either pin makes the binary unportable to older CPUs of the same
# family, and collapses every dispatch ladder to one native copy.
option(
  STRIPES_WITH_DISPATCH
  "Compile dispatched kernels once per instruction-set rung (x86-64 baseline/v2/v3/v4, aarch64 sme) and pick the best at run time. OFF compiles one copy at the toolchain baseline."
  ON
)
option(STRIPES_NATIVE_ARCH "Compile code using Stripes with -march=native / -mcpu=native" OFF)
set(STRIPES_TARGET_CPU
    ""
    CACHE STRING "Compile code using Stripes for this CPU (apple-m4, znver4, sapphirerapids, ...); overrides STRIPES_NATIVE_ARCH"
)
option(STRIPES_SHARED "Build the runtime library shared, so a process holds one dispatch choice however many libraries use it" ON)
option(STRIPES_WITH_TESTS "Build the unit tests" ${PROJECT_IS_TOP_LEVEL})
option(STRIPES_WITH_EXAMPLES "Build the examples" ${PROJECT_IS_TOP_LEVEL})
option(STRIPES_WITH_BENCHMARKS "Build the kernel benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(STRIPES_WITH_CUDA_TESTS "Build the CUDA tests of the scalar forms (needs a CUDA compiler; a device to run them)" OFF)

include(StripesDispatch)

# ── The runtime library ──────────────────────────────────────────────────────
#
# Everything but CPU detection and rung selection is header-only.
if(STRIPES_SHARED)
  add_library(stripes SHARED src/RuntimeFeatures.cpp)
else()
  add_library(stripes STATIC src/RuntimeFeatures.cpp)
  target_compile_definitions(stripes PUBLIC STRIPES_STATIC)
endif()
add_library(Stripes::stripes ALIAS stripes)

target_compile_features(stripes PUBLIC cxx_std_20)
target_include_directories(
  stripes PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(stripes PRIVATE STRIPES_EXPORTS)
set_target_properties(
  stripes
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             VERSION ${PROJECT_VERSION}
             SOVERSION ${PROJECT_VERSION_MAJOR}
             EXPORT_NAME stripes
)

# The architecture pin travels with the target to everything that links it, and nothing else.
# `-mcpu=` is the aarch64 spelling and `-march=` the x86 one; MSVC's /arch: vocabulary is
# different, so an MSVC build sets it in CMAKE_CXX_FLAGS instead.
set(_stripes_arch_flag "")
if(NOT "${STRIPES_TARGET_CPU}" STREQUAL "")
  set(_stripes_cpu "${STRIPES_TARGET_CPU}")
elseif(STRIPES_NATIVE_ARCH)
  set(_stripes_cpu native)
endif()
if(DEFINED _stripes_cpu)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    set(_stripes_arch_flag "-mcpu=${_stripes_cpu}")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
    set(_stripes_arch_flag "-march=${_stripes_cpu}")
  endif()
endif()
if(NOT "${_stripes_arch_flag}" STREQUAL "")
  if(MSVC)
    message(WARNING "Stripes: ${_stripes_arch_flag} is GCC/Clang syntax and MSVC ignores it; "
                    "set /arch:AVX512 or similar in CMAKE_CXX_FLAGS instead."
    )
  else()
    target_compile_options(stripes INTERFACE ${_stripes_arch_flag})
    message(STATUS "Stripes: applying ${_stripes_arch_flag} to everything that links Stripes::stripes")
  endif()
endif()

# The launcher the rung tests run under (stripes_add_rung_tests): installed, so a project testing
# its own dispatched kernels can use it too.
# It compiles the runtime in rather than linking the library, so an installed copy runs without
# finding libstripes (no RPATH, no DLL search path).
add_executable(stripes_rung_guard tests/rung_guard/stripes_rung_guard.cpp src/RuntimeFeatures.cpp)
target_include_directories(stripes_rung_guard PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(stripes_rung_guard PRIVATE cxx_std_20)
target_compile_definitions(stripes_rung_guard PRIVATE STRIPES_STATIC)
set_target_properties(stripes_rung_guard PROPERTIES EXPORT_NAME rung_guard)
add_executable(Stripes::rung_guard ALIAS stripes_rung_guard)

# ── Install ──────────────────────────────────────────────────────────────────
install(
  TARGETS stripes stripes_rung_guard
  EXPORT StripesTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(_stripes_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/Stripes)
install(
  EXPORT StripesTargets
  NAMESPACE Stripes::
  DESTINATION ${_stripes_cmake_dir}
)
# The build tree is a usable package too, for a project that fetches Stripes and exports its own
# build tree: the targets, and beside them the files the config includes.
export(
  EXPORT StripesTargets
  NAMESPACE Stripes::
  FILE ${CMAKE_CURRENT_BINARY_DIR}/StripesTargets.cmake
)
configure_file(cmake/StripesDispatch.cmake ${CMAKE_CURRENT_BINARY_DIR}/StripesDispatch.cmake COPYONLY)
configure_file(cmake/StripesCheckRungObjects.cmake ${CMAKE_CURRENT_BINARY_DIR}/StripesCheckRungObjects.cmake COPYONLY)
configure_package_config_file(
  cmake/StripesConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/StripesConfig.cmake INSTALL_DESTINATION ${_stripes_cmake_dir}
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/StripesConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/StripesConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/StripesConfigVersion.cmake
              cmake/StripesDispatch.cmake cmake/StripesCheckRungObjects.cmake DESTINATION ${_stripes_cmake_dir}
)

# ── Tests, examples, benchmarks ──────────────────────────────────────────────
if(STRIPES_WITH_TESTS
   OR STRIPES_WITH_BENCHMARKS
   OR STRIPES_WITH_EXAMPLES
   OR STRIPES_WITH_CUDA_TESTS
)
  enable_testing()
endif()
if(STRIPES_WITH_TESTS OR STRIPES_WITH_BENCHMARKS)
  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    # Catch2 compiles its string_view support only at C++17 and above, and the tests use it, so the
    # fetched copy is built at the tests' standard rather than the compiler's default (C++14 under
    # clang-cl, which left StringMaker<std::string_view> undefined at link time).
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.8.1
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
  endif()
endif()
# Intel's icx/icpx default to -fp-model=fast, which reassociates floating-point arithmetic, a mode the
# library does not support (the SSE2 rounding emulation relies on an add and subtract staying
# apart). Stripes' own tests, benchmarks and examples are built precise, so they check the library
# under the semantics it promises. Set here, after Catch2, it reaches only the directories below and
# never a project that uses Stripes.
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  if(CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fp:precise>)
  else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fp-model=precise>)
  endif()
endif()
if(STRIPES_WITH_TESTS)
  add_subdirectory(tests/unit)
endif()
if(STRIPES_WITH_BENCHMARKS)
  add_subdirectory(tests/performance)
endif()
if(STRIPES_WITH_EXAMPLES)
  add_subdirectory(examples)
endif()
if(STRIPES_WITH_CUDA_TESTS)
  enable_language(CUDA)
  add_subdirectory(tests/gpu)
endif()
