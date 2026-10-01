#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(example_programs
    SIMDBasics
    PlatformInfo
    Saxpy
    DotProduct
    ComplexAxpy
    ColumnSum
    BlockTranspose
    ClampAndSanitize
    IntegerBits
    QuantizedGemv
    StreamingScale
    RuntimeDispatch
    GemmMicroKernel
    OperatorsAndFunctions
)

# RuntimeDispatchKernel.cpp is not a program: it is compiled once per instruction-set rung, each
# copy in its own arch_<rung> namespace, and RuntimeDispatch picks one at run time. The x86 rungs
# only; the SME rung would buy a streaming loop nothing. On other targets this is one native copy.
include(StripesDispatch)
stripes_add_dispatch_sources(StripesExampleKernelRungs IMPL RuntimeDispatchKernel.cpp RUNGS baseline v2 v3 v4)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  add_executable(example_${example_program} ${sources})
  target_link_libraries(example_${example_program} PRIVATE Stripes::stripes)

  # OperatorsAndFunctions shows that a * b - p is zero where fmadd(a, b, -p) is not, which holds
  # only while the compiler does not fuse the expression itself. GCC does by default under GNU
  # extensions, even through the intrinsics; Clang fuses only within one source expression.
  if(example_program STREQUAL "OperatorsAndFunctions")
    target_compile_options(example_OperatorsAndFunctions PRIVATE $<$<CXX_COMPILER_ID:GNU>:-ffp-contract=off>)
  endif()

  if(example_program STREQUAL "RuntimeDispatch")
    target_sources(example_RuntimeDispatch PRIVATE ${StripesExampleKernelRungs})
    # Tells STRIPES_LADDER which arch_<rung> copies exist.
    set_source_files_properties(
      RuntimeDispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${StripesExampleKernelRungs_DEFINITIONS}"
    )
  endif()

  add_test(NAME example.${example_program} COMMAND example_${example_program})
endforeach()
