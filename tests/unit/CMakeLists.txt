#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(StripesTests
    VecBasic
    VecInteger
    VecHalf
    VecInt8
    VecMath
    VecMask
    VecReduce
    VecPartial
    VecConvert
    StreamStore
    Shuffle
    GatherScatter
    ComplexVec
    GenericKernel
    VecWide
    VecExp
    VecErf
    RuntimeFeatures
)

# Every suite but RuntimeFeatures calls the headers directly, so
# what it exercises is fixed when the test itself is compiled, not by runtime
# dispatch. Built once at the ambient flags these read one rung on every
# machine; the helper adds a copy per higher rung so the wider paths are
# covered too. RuntimeFeatures tests CPU detection, which does not widen.
set(StripesWidthDependentTests
    VecBasic
    VecInteger
    VecHalf
    VecInt8
    VecMath
    VecMask
    VecReduce
    VecPartial
    VecConvert
    StreamStore
    Shuffle
    GatherScatter
    ComplexVec
    GenericKernel
    VecWide
    VecExp
    VecErf
)

# A plain Catch2 main: Stripes has no runtime of its own to start.
foreach(test ${StripesTests})
  add_executable(${test}_test ${test}.cpp)
  target_link_libraries(${test}_test PRIVATE Stripes::stripes Catch2::Catch2WithMain)
  # GenericKernel compares the vector and scalar instantiations of one kernel bit for bit. GCC
  # contracts a separate multiply and add into an FMA by default (-ffp-contract=fast), even in ISO
  # mode and in intrinsic code, and decides per instantiation, so with it on the comparison would
  # test the compiler's contraction choices rather than the library's operations.
  if(test STREQUAL "GenericKernel" AND NOT MSVC)
    target_compile_options(${test}_test PRIVATE -ffp-contract=off)
  endif()
  add_test(NAME unit.${test} COMMAND ${test}_test)
  # Catch2 exits 4 when every case it ran was skipped (ComplexVec in the scalar fallback build, which
  # has no CVec), so ctest reports Skipped rather than a failure.
  set_tests_properties(unit.${test} PROPERTIES SKIP_RETURN_CODE 4)
endforeach()

# The ladder tests pin the sme gate against what the rung's flags enable on
# this compiler, the same answer RuntimeFeatures.cpp was built with.
stripes_rung_enables(sme _sme_sve _sme_sve2)
target_compile_definitions(
  RuntimeFeatures_test PRIVATE STRIPES_SME_RUNG_ENABLES_SVE=$<BOOL:${_sme_sve}>
                               STRIPES_SME_RUNG_ENABLES_SVE2=$<BOOL:${_sme_sve2}>
)

# A copy of each width-dependent suite per higher rung (see StripesWidthDependentTests).
foreach(test ${StripesWidthDependentTests})
  stripes_add_rung_compiled_tests(TARGET ${test}_test NAME unit.${test})
endforeach()

# IsaNamespace_test links a baseline translation unit with one compiled at the v3
# rung's flags, to check that the headers' instruction-set namespace keeps their
# entities apart. Only x86 has a rung whose lane count differs from the baseline's.
if(STRIPES_WITH_DISPATCH
   AND NOT STRIPES_NATIVE_ARCH
   AND "${STRIPES_TARGET_CPU}" STREQUAL ""
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
)
  stripes_rung_flags(v3 _isa_wide_flags _isa_wide_ordinal _isa_wide_ok "IsaNamespace_test")
  if(_isa_wide_ok)
    add_executable(IsaNamespace_test IsaNamespace.cpp IsaNamespaceWide.cpp)
    target_link_libraries(IsaNamespace_test PRIVATE Stripes::stripes Catch2::Catch2WithMain)
    set_source_files_properties(IsaNamespaceWide.cpp PROPERTIES COMPILE_OPTIONS "${_isa_wide_flags}")
    add_test(NAME unit.IsaNamespace COMMAND IsaNamespace_test)
  endif()
endif()

# selected_arch() chooses once per process, so each ArchOverride case is its own test process.
add_executable(ArchOverride_test ArchOverride.cpp)
target_link_libraries(ArchOverride_test PRIVATE Stripes::stripes Catch2::Catch2WithMain)
add_test(NAME unit.ArchOverride.override COMMAND ArchOverride_test "[override]")
add_test(NAME unit.ArchOverride.env COMMAND ArchOverride_test "[env]")
set_tests_properties(unit.ArchOverride.env PROPERTIES ENVIRONMENT "STRIPES_ARCH=baseline")
add_test(NAME unit.ArchOverride.override_env COMMAND ArchOverride_test "[override-env]")
set_tests_properties(unit.ArchOverride.override_env PROPERTIES ENVIRONMENT "STRIPES_ARCH=bogus")
add_test(NAME unit.ArchOverride.handler COMMAND ArchOverride_test "[handler]")
