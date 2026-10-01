#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

#:
#: .. cmake:command:: stripes_add_dispatch_sources
#:
#:    Generate per-instruction-set translation units for a runtime dispatch
#:    ladder (see ``Stripes/RuntimeFeatures.hpp``).
#:
#:    .. code-block:: cmake
#:
#:       stripes_add_dispatch_sources(<out_var>
#:         IMPL <impl-file>
#:         [RUNGS <rung>...]        # subset of: baseline v2 v3 v4 (default: all)
#:       )
#:
#:    For each rung, a thin wrapper ``.cpp`` is generated into the current
#:    binary directory that
#:
#:    1. defines ``STRIPES_ARCH_NS`` to ``arch_<rung>`` (the namespace
#:       the implementation file must wrap its arch-dependent code in),
#:    2. defines ``STRIPES_DISPATCH_RUNG`` to the rung's ordinal
#:       (0 = baseline ... 3 = v4), and
#:    3. includes the implementation file,
#:
#:    and is given the compiler flags of that rung (``-march=x86-64-v2/-v3/-v4``
#:    for GCC/Clang, ``/arch:AVX2``/``/arch:AVX512`` for the MSVC driver).
#:    Because the Stripes headers key off compiler-defined feature macros, the
#:    same implementation source widens ``Vec<T>``/``native_lanes``/all
#:    operations to each rung's register width without source changes.
#:
#:    The generated source list is returned in ``<out_var>`` for passing to
#:    a target's sources. The rungs actually generated are
#:    returned in ``<out_var>_RUNGS``, and matching compile definitions of
#:    the form ``STRIPES_HAS_RUNG_<RUNG>=1`` are returned in
#:    ``<out_var>_DEFINITIONS`` so dispatch-table code can declare exactly
#:    the namespaces that exist (add them to the consuming target with
#:    ``target_compile_definitions``). When the ``sme`` rung is generated they
#:    also carry ``STRIPES_SME_RUNG_ENABLES_SVE`` and ``_SVE2``: whether this
#:    compiler's SME flags switch on non-streaming SVE as well, which
#:    ``STRIPES_LADDER`` passes to ``select()`` so a core with SME but no SVE
#:    is not handed code that needs it. Give the definitions to the
#:    translation unit that calls ``select()``.
#:
#:    Single-TU mode: the requested ladder collapses to one ``native`` rung
#:    (namespace ``arch_native``, ambient compiler flags, definition
#:    ``STRIPES_HAS_RUNG_NATIVE``) when any of the following holds,
#:    because a ladder would be meaningless or unreachable:
#:
#:    * ``STRIPES_WITH_DISPATCH`` is OFF,
#:    * the target processor is not x86-64 (the v2/v3/v4 rungs are x86
#:      levels; on aarch64 the toolchain baseline already includes NEON),
#:    * ``STRIPES_NATIVE_ARCH`` or ``STRIPES_TARGET_CPU`` pins the
#:      whole SIMD interface to a specific CPU (the pin raises every TU's
#:      baseline, so a runtime ladder below it can never be selected).
#:
#:    Independent of that, individual rungs are dropped (degrading toward
#:    the always-present baseline) when their flag is unusable: the true
#:    MSVC driver has no flag for ``v2`` (``v3``/``v4`` map to
#:    ``/arch:AVX2`` and ``/arch:AVX512``; clang-cl reaches ``v2`` via
#:    ``/clang:-march=x86-64-v2`` and Intel icx accepts the GCC spelling),
#:    and every rung flag is probed with ``check_cxx_compiler_flag`` first,
#:    so compilers predating ``-march=x86-64-vN`` (GCC < 11, Clang < 12)
#:    build baseline-only instead of failing to configure.
#:
#:    The implementation file is compiled once per surviving rung, so keep
#:    everything arch-independent out of it; heavy shared code belongs in a
#:    regular TU. Every copy must define its own names: the implementation's
#:    in ``STRIPES_ARCH_NS``, the Stripes headers' in the instruction-set
#:    namespace they open themselves (``STRIPES_ISA_NS``). A template
#:    from another library instantiated here (a ``std::vector`` member, an fmt
#:    formatter) is emitted by every copy under one name, and the linker keeps
#:    one of them for all callers.
include(CheckCXXCompilerFlag)

# The switches the functions below read. Stripes' own CMakeLists declares them as options; a project
# that only includes this file (through find_package) gets the same defaults and may set them first.
if(NOT DEFINED STRIPES_WITH_DISPATCH)
  set(STRIPES_WITH_DISPATCH ON)
endif()
if(NOT DEFINED STRIPES_NATIVE_ARCH)
  set(STRIPES_NATIVE_ARCH OFF)
endif()
if(NOT DEFINED STRIPES_TARGET_CPU)
  set(STRIPES_TARGET_CPU "")
endif()

#:
#: .. cmake:command:: stripes_rung_flags
#:
#:    Resolve one dispatch rung to the flags that raise a translation unit to
#:    it, its ladder ordinal, and whether this toolchain can reach it at all.
#:
#:    .. code-block:: cmake
#:
#:       stripes_rung_flags(<rung> <out_flags> <out_ordinal> <out_ok> <context>)
#:
#:    ``<out_ok>`` is FALSE when the driver has no spelling for the rung (the
#:    true MSVC driver has none for ``v2`` or ``sme``) or when it rejects the
#:    spelling it does have (compilers predating ``-march=x86-64-vN``). The
#:    caller drops that rung rather than failing to configure. ``<context>``
#:    only prefixes the diagnostic.
#:
#:    Both the per-rung kernel TUs and the per-rung compiled tests resolve
#:    flags through here, so a new rung or a new driver spelling is added in
#:    one place.
function(stripes_rung_flags rung out_flags out_ordinal out_ok context)
  # True MSVC driver (cl.exe): no flag exists for the v2 level. clang-cl
  # takes the GCC spelling through the /clang: escape hatch, and Intel's
  # icx-cl (IntelLLVM) accepts the GCC spelling directly.
  set(_msvc_true_driver FALSE)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
    set(_msvc_true_driver TRUE)
  endif()

  set(_flags "")
  set(_ordinal 0)
  set(_ok TRUE)

  if(rung STREQUAL "native" OR rung STREQUAL "baseline")
    set(_ordinal 0)
  elseif(rung STREQUAL "v2")
    set(_ordinal 1)
    if(_msvc_true_driver)
      message(STATUS "${context}: MSVC cl has no x86-64-v2 flag; dropping the v2 rung")
      set(_ok FALSE)
    elseif(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # clang-cl
      set(_flags "/clang:-march=x86-64-v2")
    else() # GCC/Clang/IntelLLVM (icx accepts the GCC spelling on Windows too)
      set(_flags "-march=x86-64-v2")
    endif()
  elseif(rung STREQUAL "v3")
    set(_ordinal 2)
    if(MSVC)
      set(_flags "/arch:AVX2")
    else()
      set(_flags "-march=x86-64-v3")
    endif()
  elseif(rung STREQUAL "v4")
    set(_ordinal 3)
    if(MSVC)
      set(_flags "/arch:AVX512")
    else()
      set(_flags "-march=x86-64-v4")
    endif()
  elseif(rung STREQUAL "sme")
    set(_ordinal 4)
    if(_msvc_true_driver)
      message(STATUS "${context}: MSVC cl has no SME flag; dropping the sme rung")
      set(_ok FALSE)
    elseif(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # clang-cl
      set(_flags "/clang:-march=armv8.6-a+sme2+sme-f64f64")
    else() # GCC/Clang/AppleClang
      set(_flags "-march=armv8.6-a+sme2+sme-f64f64")
    endif()
  else()
    message(FATAL_ERROR "stripes_rung_flags: unknown rung '${rung}' (expected baseline/native/v2/v3/v4/sme)")
  endif()

  # Old compilers (pre -march=x86-64-vN: GCC < 11, Clang < 12) or drivers
  # that reject a spelling drop the rung instead of breaking the build -
  # the ladder degrades toward baseline, which always exists.
  if(_ok AND NOT "${_flags}" STREQUAL "")
    string(TOUPPER "${rung}" _rung_upper)
    check_cxx_compiler_flag("${_flags}" STRIPES_RUNG_FLAG_${_rung_upper})
    if(NOT STRIPES_RUNG_FLAG_${_rung_upper})
      message(STATUS "${context}: compiler rejects '${_flags}'; dropping the ${rung} rung")
      set(_ok FALSE)
    endif()
  endif()

  set(${out_flags} "${_flags}" PARENT_SCOPE)
  set(${out_ordinal} "${_ordinal}" PARENT_SCOPE)
  set(${out_ok} "${_ok}" PARENT_SCOPE)
endfunction()

#:
#: .. cmake:command:: stripes_rung_enables
#:
#:    Report whether a rung's flags also switch on non-streaming SVE or SVE2.
#:
#:    .. code-block:: cmake
#:
#:       stripes_rung_enables(<rung> <out_sve> <out_sve2>)
#:
#:    A rung's translation units may use anything their flags enable, so the
#:    runtime gate for the rung has to require all of it. For ``sme`` that is
#:    more than SME on some compilers: GCC before 15 makes ``+sme`` imply
#:    ``+sve2``, and its autovectorizer then emits non-streaming SVE that
#:    faults on a core with SME but no SVE (Apple M4). This preprocesses an
#:    empty TU at the rung's flags and reports ``__ARM_FEATURE_SVE`` and
#:    ``__ARM_FEATURE_SVE2``. Both are FALSE off aarch64 or when the rung is
#:    dropped.
function(stripes_rung_enables rung out_sve out_sve2)
  set(_sve FALSE)
  set(_sve2 FALSE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(_flags "")
    set(_ordinal 0)
    set(_rung_ok FALSE)
    stripes_rung_flags(${rung} _flags _ordinal _rung_ok "stripes_rung_enables(${rung})")
    if(_rung_ok AND NOT "${_flags}" STREQUAL "")
      include(CheckCXXSourceCompiles)
      string(TOUPPER "${rung}" _rung_upper)
      set(CMAKE_REQUIRED_FLAGS "${_flags}")
      set(CMAKE_REQUIRED_QUIET ON)
      foreach(_feature SVE SVE2)
        check_cxx_source_compiles(
          "#if !defined(__ARM_FEATURE_${_feature})\n#error not enabled\n#endif\nint main() { return 0; }"
          STRIPES_RUNG_${_rung_upper}_ENABLES_${_feature}
        )
      endforeach()
      if(STRIPES_RUNG_${_rung_upper}_ENABLES_SVE)
        set(_sve TRUE)
      endif()
      if(STRIPES_RUNG_${_rung_upper}_ENABLES_SVE2)
        set(_sve2 TRUE)
      endif()
    endif()
  endif()
  set(${out_sve} "${_sve}" PARENT_SCOPE)
  set(${out_sve2} "${_sve2}" PARENT_SCOPE)
endfunction()

function(stripes_add_dispatch_sources out_var)
  set(options)
  set(one_value_args IMPL)
  set(multi_value_args RUNGS)
  cmake_parse_arguments(_simd "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT _simd_IMPL)
    message(FATAL_ERROR "stripes_add_dispatch_sources: IMPL is required")
  endif()
  if(NOT _simd_RUNGS)
    set(_simd_RUNGS baseline v2 v3 v4)
  endif()

  get_filename_component(_impl_abs "${_simd_IMPL}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT EXISTS "${_impl_abs}")
    message(FATAL_ERROR "stripes_add_dispatch_sources: IMPL file not found: ${_impl_abs}")
  endif()
  get_filename_component(_impl_name "${_impl_abs}" NAME_WE)

  # Decide which architecture's ladder applies.
  set(_is_x86 FALSE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_is_x86 TRUE)
  endif()
  set(_is_aarch64 FALSE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(_is_aarch64 TRUE)
  endif()
  set(_pinned FALSE)
  if(STRIPES_NATIVE_ARCH OR NOT "${STRIPES_TARGET_CPU}" STREQUAL "")
    set(_pinned TRUE)
  endif()
  # Single-TU mode: no ladder. The wrapper compiles at the ambient flags in
  # the arch_native namespace, and consumers get STRIPES_HAS_RUNG_NATIVE
  # instead of the per-rung definitions.
  #
  # On aarch64 the x86 rungs (baseline/v2/v3/v4) collapse to `native`, but a
  # requested `sme` rung survives alongside it: NEON is the toolchain
  # baseline (native), and SME2 is the one optional aarch64 rung.
  if(_pinned OR NOT STRIPES_WITH_DISPATCH)
    set(_simd_RUNGS native)
  elseif(_is_aarch64)
    set(_arm_rungs native)
    if("sme" IN_LIST _simd_RUNGS)
      list(APPEND _arm_rungs sme)
    endif()
    set(_simd_RUNGS ${_arm_rungs})
  elseif(_is_x86)
    list(REMOVE_ITEM _simd_RUNGS sme)
    if(NOT _simd_RUNGS)
      set(_simd_RUNGS native)
    endif()
  else()
    set(_simd_RUNGS native)
  endif()

  set(_sources)
  set(_definitions)
  set(_generated_rungs)
  foreach(_rung IN LISTS _simd_RUNGS)
    # Declared before the call: stripes_rung_flags returns through
    # PARENT_SCOPE, so without these the names would carry the previous
    # iteration's values if it ever returned without setting them.
    set(_flags "")
    set(_ordinal 0)
    set(_rung_ok FALSE)
    stripes_rung_flags(${_rung} _flags _ordinal _rung_ok "stripes_add_dispatch_sources(${_impl_name})")
    if(NOT _rung_ok)
      continue()
    endif()

    set(_wrapper "${CMAKE_CURRENT_BINARY_DIR}/simd_dispatch/${_impl_name}_${_rung}.cpp")
    file(
      CONFIGURE
      OUTPUT "${_wrapper}"
      CONTENT
        "// Generated by stripes_add_dispatch_sources - do not edit.
#define STRIPES_ARCH_NS arch_${_rung}
#define STRIPES_DISPATCH_RUNG ${_ordinal}
#include \"${_impl_abs}\"
"
      @ONLY
      NEWLINE_STYLE UNIX
    )

    # stripes_rung_flags already probed the spelling, so anything that
    # reaches here is a flag this driver accepts.
    if(NOT "${_flags}" STREQUAL "")
      set_source_files_properties("${_wrapper}" PROPERTIES COMPILE_OPTIONS "${_flags}")
    endif()

    # A rung TU is compiled at its own -march, so it can never share the
    # project's baseline precompiled header: GCC rejects it with
    #   warning: ... .gch: created and used with differing settings of '-march='
    # and the TU re-parses the headers anyway. Opting out makes that explicit
    # and silences a warning that is inherent rather than fixable.
    set_source_files_properties("${_wrapper}" PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

    list(APPEND _sources "${_wrapper}")
    string(TOUPPER "${_rung}" _rung_upper)
    list(APPEND _definitions "STRIPES_HAS_RUNG_${_rung_upper}=1")
    list(APPEND _generated_rungs "${_rung}")
  endforeach()

  # What this compiler's sme flags switch on besides SME, for select() to require at run time
  # (STRIPES_LADDER passes it; see stripes::SmeRungRequires). Probed here, with the compiler that
  # builds these copies, because Stripes itself may have been built by another.
  if("sme" IN_LIST _generated_rungs)
    stripes_rung_enables(sme _sme_sve _sme_sve2)
    foreach(_feature SVE SVE2)
      string(TOLOWER "${_feature}" _lower)
      if(_sme_${_lower})
        list(APPEND _definitions "STRIPES_SME_RUNG_ENABLES_${_feature}=1")
      else()
        list(APPEND _definitions "STRIPES_SME_RUNG_ENABLES_${_feature}=0")
      endif()
    endforeach()
  endif()

  set(${out_var}
      "${_sources}"
      PARENT_SCOPE
  )
  set(${out_var}_RUNGS
      "${_generated_rungs}"
      PARENT_SCOPE
  )
  set(${out_var}_DEFINITIONS
      "${_definitions}"
      PARENT_SCOPE
  )
endfunction()

# The rung guard's options for <rung>: on the sme rung, what this compiler's sme flags need besides
# SME, so a host that select() would quietly drop to baseline reports the test Skipped instead.
function(_stripes_rung_guard_options rung out_var)
  set(_options)
  if(rung STREQUAL "sme")
    stripes_rung_enables(sme _sve _sve2)
    if(_sve)
      list(APPEND _options --require-sve)
    endif()
    if(_sve2)
      list(APPEND _options --require-sve2)
    endif()
  endif()
  set(${out_var}
      "${_options}"
      PARENT_SCOPE
  )
endfunction()

#:
#: .. cmake:command:: stripes_add_rung_tests
#:
#:    Register an existing test executable once more per dispatch rung, forcing
#:    the rung its dispatch tables select through ``STRIPES_ARCH``:
#:
#:    .. code-block:: cmake
#:
#:       stripes_add_rung_tests(TARGET <executable> NAME <test-prefix>
#:                              [ARGS <arg>...] [OUT_TESTS <var>])
#:
#:    creates ``<test-prefix>.baseline``, ``.v2``, ``.v3`` and ``.v4`` on x86
#:    (``.baseline`` and ``.sme`` on aarch64), each running ``<executable>
#:    <arg>...``. Each runs through the ``Stripes::rung_guard`` launcher, which
#:    exits 77 (registered as ``SKIP_RETURN_CODE``) when the host cannot run
#:    the rung, so ctest reports Skipped instead of silently passing at a
#:    clamped lower rung. ``OUT_TESTS`` receives the test names, for a caller
#:    that sets further properties. ``STRIPES_ARCH`` is appended to the tests'
#:    ``ENVIRONMENT``, so set any environment of your own before it with
#:    ``APPEND`` too. No-op when the build is single-TU (dispatch OFF, a
#:    compile-time CPU pin, or another architecture), where only arch_native
#:    exists.
function(stripes_add_rung_tests)
  cmake_parse_arguments(_rt "" "TARGET;NAME;OUT_TESTS" "ARGS" ${ARGN})
  if(NOT _rt_TARGET OR NOT _rt_NAME)
    message(FATAL_ERROR "stripes_add_rung_tests: TARGET and NAME are required")
  endif()
  set(_tests)
  if(STRIPES_WITH_DISPATCH
     AND NOT STRIPES_NATIVE_ARCH
     AND "${STRIPES_TARGET_CPU}" STREQUAL ""
  )
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
      set(_rungs baseline v2 v3 v4)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
      set(_rungs baseline sme)
    else()
      set(_rungs)
    endif()
    foreach(_rung IN LISTS _rungs)
      set(_test "${_rt_NAME}.${_rung}")
      _stripes_rung_guard_options(${_rung} _guard_options)
      add_test(NAME ${_test} COMMAND Stripes::rung_guard ${_guard_options} ${_rung} $<TARGET_FILE:${_rt_TARGET}> ${_rt_ARGS})
      set_property(
        TEST ${_test}
        APPEND
        PROPERTY ENVIRONMENT "STRIPES_ARCH=${_rung}"
      )
      set_tests_properties(${_test} PROPERTIES SKIP_RETURN_CODE 77)
      list(APPEND _tests ${_test})
    endforeach()
  endif()
  if(_rt_OUT_TESTS)
    set(${_rt_OUT_TESTS}
        "${_tests}"
        PARENT_SCOPE
    )
  endif()
endfunction()

#:
#: .. cmake:command:: stripes_add_rung_compiled_tests
#:
#:    Build a copy of a test executable per dispatch rung, each compiled AT
#:    that rung, and register each under the ``Stripes::rung_guard`` launcher:
#:
#:    .. code-block:: cmake
#:
#:       stripes_add_rung_compiled_tests(TARGET <executable> NAME <test-prefix>
#:                                       [ARGS <arg>...] [OUT_TESTS <var>])
#:
#:    The copies are named ``<executable>_<rung>`` and take the base target's
#:    sources, libraries, include directories, compile definitions, options
#:    and features, plus the rung's flags; the tests are
#:    ``<test-prefix>.<rung>``. Configure the base target fully before this
#:    call, since the copies are made from what it has then.
#:
#:    This is the counterpart to ``stripes_add_rung_tests`` for code that
#:    resolves at COMPILE time rather than through a runtime dispatch table.
#:    ``STRIPES_ARCH`` moves the rung a dispatch table selects, so it
#:    covers kernels whose rungs were compiled into a library. It cannot
#:    reach an inline function in a header, which is frozen at whatever width
#:    its own translation unit was compiled for: a test that calls Stripes
#:    directly, built once at the ambient flags, exercises one rung no matter
#:    what the environment says and no matter what the host CPU supports.
#:
#:    That gap is not hypothetical. It is why a wrong AVX-512 ``Vec<double>``
#:    transpose survived in ``Shuffle.hpp`` with a correctness test sitting
#:    directly on top of it: ``__AVX512F__`` was never defined while that test
#:    was compiled, so the test read the AVX2 branch on every machine.
#:
#:    The ambient build already covers the ``baseline`` rung (x86) and NEON
#:    (aarch64), so only the rungs above it get a copy: ``v2``, ``v3`` and
#:    ``v4`` on x86, ``sme`` on aarch64. Rungs this toolchain cannot spell are
#:    dropped, as everywhere else on the ladder. No-op when the build is
#:    single-TU (dispatch OFF or a compile-time CPU pin).
function(stripes_add_rung_compiled_tests)
  cmake_parse_arguments(_ct "" "TARGET;NAME;OUT_TESTS" "ARGS" ${ARGN})
  if(NOT _ct_TARGET OR NOT _ct_NAME)
    message(FATAL_ERROR "stripes_add_rung_compiled_tests: TARGET and NAME are required")
  endif()
  set(_tests)
  set(_rungs)
  if(STRIPES_WITH_DISPATCH
     AND NOT STRIPES_NATIVE_ARCH
     AND "${STRIPES_TARGET_CPU}" STREQUAL ""
  )
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
      set(_rungs v2 v3 v4)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
      set(_rungs sme)
    endif()
  endif()

  get_target_property(_dir ${_ct_TARGET} SOURCE_DIR)
  get_target_property(_sources ${_ct_TARGET} SOURCES)
  set(_absolute)
  foreach(_source IN LISTS _sources)
    get_filename_component(_source "${_source}" ABSOLUTE BASE_DIR "${_dir}")
    list(APPEND _absolute "${_source}")
  endforeach()

  foreach(_rung IN LISTS _rungs)
    set(_flags "")
    set(_ordinal 0)
    set(_rung_ok FALSE)
    stripes_rung_flags(${_rung} _flags _ordinal _rung_ok "stripes_add_rung_compiled_tests(${_ct_TARGET})")
    if(NOT _rung_ok)
      continue()
    endif()

    set(_target "${_ct_TARGET}_${_rung}")
    add_executable(${_target} ${_absolute})
    foreach(_property LINK_LIBRARIES INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS COMPILE_FEATURES)
      get_target_property(_value ${_ct_TARGET} ${_property})
      if(_value)
        set_property(TARGET ${_target} PROPERTY ${_property} "${_value}")
      endif()
    endforeach()
    target_compile_options(${_target} PRIVATE ${_flags})
    # A TU compiled at its own -march cannot share a baseline precompiled
    # header; GCC warns and re-parses anyway.
    set_target_properties(${_target} PROPERTIES DISABLE_PRECOMPILE_HEADERS ON)

    # The binary IS the rung here, so there is no STRIPES_ARCH to set. The
    # guard still runs first, because a v4-compiled binary would take SIGILL on
    # a host without AVX-512; it exits 77 and ctest says Skipped.
    set(_test "${_ct_NAME}.${_rung}")
    _stripes_rung_guard_options(${_rung} _guard_options)
    add_test(NAME ${_test} COMMAND Stripes::rung_guard ${_guard_options} ${_rung} $<TARGET_FILE:${_target}> ${_ct_ARGS})
    set_tests_properties(${_test} PROPERTIES SKIP_RETURN_CODE 77)
    list(APPEND _tests ${_test})
  endforeach()
  if(_ct_OUT_TESTS)
    set(${_ct_OUT_TESTS}
        "${_tests}"
        PARENT_SCOPE
    )
  endif()
endfunction()

#:
#: .. cmake:command:: stripes_add_rung_objects_test
#:
#:    Register a test that every per-rung object of ``<target>`` keeps its
#:    weak symbols to itself.
#:
#:    .. code-block:: cmake
#:
#:       stripes_add_rung_objects_test(NAME <test> TARGET <target>)
#:
#:    Each rung's copy of an implementation file is compiled at that rung's
#:    flags, and a weak symbol it defines under a name the other copies share
#:    (a ``std::vector`` member, an fmt formatter) is merged at link time: one
#:    copy, possibly the AVX-512 one, serves every caller. The test runs ``nm``
#:    over the target's objects from ``stripes_add_dispatch_sources`` and
#:    fails on any weak symbol outside the rung's ``arch_<rung>`` namespace or
#:    the Stripes headers' ``isa_<features>`` namespace. ``<target>`` must be
#:    an OBJECT library. The test is registered only for ELF toolchains (GCC or
#:    Clang on Linux), where ``nm`` reports symbol binding portably.
function(stripes_add_rung_objects_test)
  cmake_parse_arguments(_ot "" "NAME;TARGET" "" ${ARGN})
  if(NOT _ot_NAME OR NOT _ot_TARGET)
    message(FATAL_ERROR "stripes_add_rung_objects_test: NAME and TARGET are required")
  endif()
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
     OR NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
     OR NOT CMAKE_NM
  )
    return()
  endif()
  add_test(NAME ${_ot_NAME}
           COMMAND ${CMAKE_COMMAND} -DNM=${CMAKE_NM} "-DOBJECTS=$<JOIN:$<TARGET_OBJECTS:${_ot_TARGET}>,|>" -P
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/StripesCheckRungObjects.cmake
  )
endfunction()
