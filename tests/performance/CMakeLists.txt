#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# BenchmarkKernels times the SIMD kernels on every dispatch rung the machine supports in one run:
# the kernels (BenchmarkKernelsImpl.cpp) are compiled once per rung, the driver once.
include(StripesDispatch)
stripes_add_dispatch_sources(StripesBenchmarkRungs IMPL BenchmarkKernelsImpl.cpp RUNGS baseline v2 v3 v4)

# The per-rung kernels as an object library, so the rung-objects check can read their symbols.
add_library(BenchmarkKernelsRungs OBJECT ${StripesBenchmarkRungs})
target_link_libraries(BenchmarkKernelsRungs PRIVATE Stripes::stripes)
stripes_add_rung_objects_test(NAME unit.BenchmarkKernelsRungObjects TARGET BenchmarkKernelsRungs)

# Timings print to stdout; run it by hand, pinned to one core on an idle machine:
#   taskset -c 2 ./BenchmarkKernels
add_executable(BenchmarkKernels BenchmarkKernels.cpp $<TARGET_OBJECTS:BenchmarkKernelsRungs>)
target_link_libraries(BenchmarkKernels PRIVATE Stripes::stripes Catch2::Catch2WithMain)
target_compile_definitions(BenchmarkKernels PRIVATE ${StripesBenchmarkRungs_DEFINITIONS})
