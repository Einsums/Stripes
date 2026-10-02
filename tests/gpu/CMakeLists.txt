#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# The scalar forms on a CUDA device against the same file's host code, bit for bit. The comparison
# holds only where neither side contracts a kernel's own multiply and add into an FMA (see
# GenericKernel in tests/unit), so both are built without contraction: nvcc's --fmad=false for the
# device and -ffp-contract=off for its host compiler, or -ffp-contract=off for both under clang.
set(_no_contraction
    $<$<CUDA_COMPILER_ID:NVIDIA>:--fmad=false>
    $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<NOT:$<BOOL:${MSVC}>>>:-Xcompiler=-ffp-contract=off>
    $<$<CUDA_COMPILER_ID:Clang>:-ffp-contract=off>
)

add_executable(ScalarKernels_test ScalarKernels.cu)
target_link_libraries(ScalarKernels_test PRIVATE Stripes::stripes)
set_target_properties(ScalarKernels_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_compile_options(ScalarKernels_test PRIVATE ${_no_contraction})
add_test(NAME gpu.ScalarKernels COMMAND ScalarKernels_test)
# 77: no CUDA device, so only the build was checked.
set_tests_properties(gpu.ScalarKernels PROPERTIES SKIP_RETURN_CODE 77)

# Vec.hpp in a CUDA translation unit must stop with its #error. The object is left out of the default
# build, and the test builds it and expects that to fail.
add_library(VecInCuda OBJECT EXCLUDE_FROM_ALL VecInCuda.cu)
target_link_libraries(VecInCuda PRIVATE Stripes::stripes)
set_target_properties(VecInCuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
add_test(NAME gpu.VecInCudaFails COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target VecInCuda --config $<CONFIG>)
set_tests_properties(gpu.VecInCudaFails PROPERTIES PASS_REGULAR_EXPRESSION "Stripes vectors do not exist in a CUDA or HIP translation unit")
